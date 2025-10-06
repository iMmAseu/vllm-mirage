from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - typing aid only
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from deploy_qwen3 import operator_modifier

try:  # pragma: no cover - optional dependency during template development
    import mirage as mi
except ImportError:  # pragma: no cover - template still functions without Mirage
    mi = None  # type: ignore[assignment]


@dataclass
class ShapeDebugRecord:
    """Profiling metadata captured per forward invocation."""

    name: str
    input_shapes: Sequence[Optional[torch.Size]]
    output_shapes: Sequence[Optional[torch.Size]]


class ShapeLoggingModule(nn.Module):
    """Wraps a module, logging tensor shapes while delegating to the original."""

    def __init__(self, wrapped: nn.Module, name: str, logger: Optional[logging.Logger] = None) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.name = name
        self.logger = logger or logging.getLogger(__name__)

    def _collect_shapes(self, values: Iterable[object]) -> Sequence[Optional[torch.Size]]:
        shapes = []
        for value in values:
            if torch is not None and isinstance(value, torch.Tensor):
                shapes.append(value.shape)
            else:
                shapes.append(None)
        return tuple(shapes)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        debug_record = None
        if self.logger.isEnabledFor(logging.DEBUG):
            debug_record = ShapeDebugRecord(
                name=self.name,
                input_shapes=self._collect_shapes(args) + self._collect_shapes(kwargs.values()),
                output_shapes=(),
            )
            self.logger.debug("[%s] inputs=%s", debug_record.name, debug_record.input_shapes)

        outputs = self.wrapped(*args, **kwargs)

        if debug_record is not None:
            if isinstance(outputs, tuple):
                output_shapes = self._collect_shapes(outputs)
            else:
                output_shapes = self._collect_shapes((outputs,))
            debug_record = ShapeDebugRecord(
                name=debug_record.name,
                input_shapes=debug_record.input_shapes,
                output_shapes=output_shapes,
            )
            self.logger.debug("[%s] outputs=%s", debug_record.name, debug_record.output_shapes)

        return outputs


def _wrap_module_attr(parent: object, attr_name: str, wrapper_name: str, logger: logging.Logger) -> None:
    """Helper that swaps ``parent.attr`` with a shape-logging wrapper."""

    if parent is None:
        return

    module = getattr(parent, attr_name, None)
    if module is None or nn is None or not isinstance(module, nn.Module):
        logger.debug("Skip wrapping missing attribute '%s' on %s", attr_name, type(parent).__name__)
        return

    if isinstance(module, ShapeLoggingModule):  # already wrapped
        return

    setattr(parent, attr_name, ShapeLoggingModule(module, wrapper_name, logger))

class MirageAttentionProjection(nn.Module):
    """Mirage-backed replacement for torch ``Linear`` attention projections."""

    def __init__(self, wrapped: nn.Linear, name: str, logger: logging.Logger) -> None:
        super().__init__()
        self.name = name
        self.logger = logger
        self.use_bias = wrapped.bias is not None
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.original_dtype = wrapped.weight.dtype
        self.original_device = wrapped.weight.device

        self.register_buffer("weight", wrapped.weight.detach().clone())
        if self.use_bias and wrapped.bias is not None:
            self.register_buffer("bias", wrapped.bias.detach().clone())
        else:
            self.bias = None  # type: ignore[assignment]

        self.fallback = wrapped
        self._kernel_cache: dict[tuple[int, torch.dtype, torch.device], Any] = {}

    def _collect_shapes(self, values: Iterable[object]) -> Sequence[Optional[torch.Size]]:
        shapes = []
        for value in values:
            if torch is not None and isinstance(value, torch.Tensor):
                shapes.append(value.shape)
            else:
                shapes.append(None)
        return tuple(shapes)

    def _to_mirage_dtype(self, torch_dtype: torch.dtype) -> "mi.dtype":
        if torch_dtype == torch.float16:
            return mi.float16
        if torch_dtype == torch.bfloat16:
            return mi.bfloat16
        if torch_dtype == torch.float32:
            return mi.float32
        raise ValueError(f"Unsupported dtype for Mirage attention projection: {torch_dtype}")

    def _ensure_kernel(
        self, batch: int, torch_dtype: torch.dtype, device: torch.device
    ) -> Optional["mi.KNGraph"]:
        if mi is None:
            return None

        key = (batch, torch_dtype, device)
        kernel = self._kernel_cache.get(key)
        if kernel is not None:
            return kernel

        try:
            kernel = self._build_mirage_kernel(batch, torch_dtype, device)
        except Exception as exc:  # pragma: no cover - runtime diagnostic path
            self.logger.warning("Failed to build Mirage kernel for %s: %s", self.name, exc)
            return None

        self._kernel_cache[key] = kernel
        return kernel

    def _build_mirage_kernel(
        self, batch: int, torch_dtype: torch.dtype, device: torch.device
    ) -> Any:
        mi_dtype = self._to_mirage_dtype(torch_dtype)
        graph = mi.new_kernel_graph()
        x = graph.new_input((batch, self.in_features), dtype=mi_dtype)
        w = graph.new_input((self.in_features, self.out_features), dtype=mi_dtype)
        out = graph.matmul(x, w)
        graph.mark_output(out)

        sample_x = torch.empty((batch, self.in_features), dtype=torch_dtype, device=device)
        sample_w = torch.empty((self.in_features, self.out_features), dtype=torch_dtype, device=device)
        graph.compile(inputs=[sample_x, sample_w])
        return graph

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        debug_record = None
        if self.logger.isEnabledFor(logging.DEBUG):
            debug_record = ShapeDebugRecord(
                name=self.name,
                input_shapes=self._collect_shapes((inputs,)),
                output_shapes=(),
            )
            self.logger.debug("[%s] inputs=%s", debug_record.name, debug_record.input_shapes)

        outputs: Optional[torch.Tensor] = None
        if mi is not None and inputs.is_cuda:
            try:
                outputs = self._invoke_mirage_kernel(inputs)
            except Exception as exc:  # pragma: no cover - runtime diagnostic path
                self.logger.warning(
                    "Mirage kernel execution failed for %s; falling back to torch.Linear: %s",
                    self.name,
                    exc,
                )
                outputs = None

        if outputs is None:
            if self.fallback is None:
                raise RuntimeError(f"Fallback module for {self.name} was not initialized")
            outputs = self.fallback(inputs)

        if debug_record is not None:
            debug_record = ShapeDebugRecord(
                name=debug_record.name,
                input_shapes=debug_record.input_shapes,
                output_shapes=self._collect_shapes((outputs,)),
            )
            self.logger.debug("[%s] outputs=%s", debug_record.name, debug_record.output_shapes)

        return outputs

    def _invoke_mirage_kernel(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        torch_dtype = inputs.dtype
        device = inputs.device

        kernel = self._ensure_kernel(batch, torch_dtype, device)
        if kernel is None:
            raise RuntimeError("Mirage kernel unavailable")

        inp = inputs.contiguous()
        weight = self.weight.to(device=device, dtype=torch_dtype).transpose(0, 1).contiguous()

        outputs = kernel.cuda_call(inputs=[inp, weight])
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("Mirage kernel returned no outputs")

        result = outputs[0]
        if self.use_bias and self.bias is not None:
            bias = self.bias.to(device=device, dtype=torch_dtype).view(1, -1)
            result = result + bias

        return result


@operator_modifier("qwen3_attention_template")
def wrap_attention_modules(model: "torch.nn.Module") -> None:
    """Template for wrapping Qwen3 attention projections with a shape logger."""

    logger = logging.getLogger("qwen3.operator_template.attn")
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        logger.warning("Unable to locate transformer layers; skip attention template.")
        return

    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            wrapper_name = f"layer{idx}.{attr}"
            module = getattr(attn, attr, None)
            if module is None or nn is None or not isinstance(module, nn.Linear):
                logger.debug("Fallback to shape logger for %s due to unexpected module type", wrapper_name)
                _wrap_module_attr(attn, attr, wrapper_name, logger)
                continue
            setattr(attn, attr, MirageAttentionProjection(module, wrapper_name, logger))


@operator_modifier("qwen3_mlp_template")
def wrap_mlp_modules(model: "torch.nn.Module") -> None:
    """Template for wrapping Qwen3 MLP projections with a shape logger."""

    logger = logging.getLogger("qwen3.operator_template.mlp")
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        logger.warning("Unable to locate transformer layers; skip MLP template.")
        return

    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for attr in ("gate_up_proj", "down_proj"):
            wrapper_name = f"layer{idx}.{attr}"
            _wrap_module_attr(mlp, attr, wrapper_name, logger)


@operator_modifier("qwen3_rotary_template")
def wrap_rotary_embeddings(model: "torch.nn.Module") -> None:
    """Template for wrapping rotary embedding application with a shape logger."""

    logger = logging.getLogger("qwen3.operator_template.rotary")
    transformer = getattr(model, "model", None)
    rotary = getattr(transformer, "rotary_emb", None)
    if rotary is None or nn is None or isinstance(rotary, ShapeLoggingModule):
        return

    _wrap_module_attr(transformer, "rotary_emb", "rotary_emb", logger)


# Additional templates can be registered below. Each should focus on
# isolating a logical operator bundle (e.g. attention, MLP, embedding lookup)
# and rely on ``ShapeLoggingModule`` for shape inspection until a Mirage-backed
# implementation is substituted.
