#!/usr/bin/env python3
"""Utilities for deploying Qwen3 0.6B with vLLM and customizing operators."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import torch

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - handled at runtime
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]

# Type alias for operator modifier callables.
OperatorModifier = Callable[["torch.nn.Module"], None]


@dataclass
class DeploymentConfig:
    """Settings for bringing up a Qwen3 model with vLLM."""

    model: str = "Qwen/Qwen3-0.6B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.75
    max_num_seqs: Optional[int] = 16
    enforce_eager: bool = False
    dtype: Optional[str] = None
    trust_remote_code: bool = True
    max_model_len: Optional[int] = None
    speculative_model: Optional[str] = None
    disable_custom_all_reduce: bool = False


@dataclass
class PerformanceResult:
    latency_s: float
    total_tokens: int
    tokens_per_second: float
    prompt_count: int
    completions: List[str]


class OperatorModifierRegistry:
    """Collects model-operator modifiers to be applied post load."""

    def __init__(self) -> None:
        self._modifiers: Dict[str, OperatorModifier] = {}

    def register(self, name: str, modifier: OperatorModifier) -> None:
        if name in self._modifiers:
            raise ValueError(f"modifier '{name}' already registered")
        self._modifiers[name] = modifier

    def replace(self, name: str, modifier: OperatorModifier) -> None:
        self._modifiers[name] = modifier

    def apply(self, model: "torch.nn.Module", logger: logging.Logger) -> None:
        for name, modifier in self._modifiers.items():
            logger.debug("Applying operator modifier '%s'", name)
            modifier(model)

    def names(self) -> Sequence[str]:
        return tuple(self._modifiers.keys())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._modifiers)


operator_registry = OperatorModifierRegistry()


def operator_modifier(name: str) -> Callable[[OperatorModifier], OperatorModifier]:
    """Decorator-style helper to register operator modifiers."""

    def decorator(func: OperatorModifier) -> OperatorModifier:
        operator_registry.register(name, func)
        return func

    return decorator


def resolve_torch_model(llm: "LLM") -> Optional["torch.nn.Module"]:
    """Attempt to retrieve the underlying torch.nn.Module from a vLLM LLM."""

    # vLLM organizes its components with private attributes; we navigate them
    # defensively because public accessors are not yet available.
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return None

    executor = getattr(engine, "model_executor", None)
    runner = getattr(executor, "driver_worker", None)
    runner = getattr(runner, "model_runner", runner)
    model = getattr(runner, "model", None)

    return model


def apply_operator_modifiers(llm: "LLM", logger: Optional[logging.Logger] = None) -> bool:
    """Apply all registered operator modifiers to the loaded torch model."""

    if logger is None:
        logger = logging.getLogger(__name__)

    if len(operator_registry) == 0:
        logger.debug("No operator modifiers registered; skipping.")
        return True

    model = resolve_torch_model(llm)
    if model is None:
        logger.warning("Unable to access underlying torch model; operator modifiers not applied.")
        return False

    operator_registry.apply(model, logger)
    return True


def ensure_vllm_imported() -> None:
    if LLM is None or SamplingParams is None:
        raise ImportError(
            "vLLM is required but not installed. Install with `pip install vllm` before running this script."
        )


def build_llm(config: DeploymentConfig) -> "LLM":
    ensure_vllm_imported()

    init_kwargs = {
        "model": config.model,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "trust_remote_code": config.trust_remote_code,
        "enforce_eager": config.enforce_eager,
        "disable_custom_all_reduce": config.disable_custom_all_reduce,
    }
    if config.max_num_seqs is not None:
        init_kwargs["max_num_seqs"] = config.max_num_seqs
    if config.dtype:
        init_kwargs["dtype"] = config.dtype
    if config.max_model_len:
        init_kwargs["max_model_len"] = config.max_model_len
    if config.speculative_model:
        init_kwargs["speculative_model"] = config.speculative_model

    llm = LLM(**init_kwargs)
    return llm


def extract_completions(request_outputs: Iterable) -> List[str]:
    """Normalize vLLM request outputs into plain text completions."""

    completions: List[str] = []
    for request_output in request_outputs:
        outputs = getattr(request_output, "outputs", [])
        if not outputs:
            completions.append("")
            continue
        first = outputs[0]
        text = getattr(first, "text", "")
        completions.append(text)
    return completions


def count_generated_tokens(request_outputs: Iterable) -> int:
    tokens = 0
    for request_output in request_outputs:
        outputs = getattr(request_output, "outputs", [])
        if not outputs:
            continue
        first = outputs[0]
        token_ids = getattr(first, "token_ids", None)
        if token_ids is None:
            token_ids = getattr(first, "token_ids_tensor", [])
        if hasattr(token_ids, "__len__"):
            tokens += len(token_ids)
    return tokens


def run_generation(
    llm: "LLM",
    prompts: Sequence[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: Optional[logging.Logger] = None,
) -> PerformanceResult:
    ensure_vllm_imported()

    if logger is None:
        logger = logging.getLogger(__name__)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    start_time = time.perf_counter()
    request_outputs = llm.generate(prompts, sampling_params)
    latency = time.perf_counter() - start_time

    completions = extract_completions(request_outputs)
    total_tokens = count_generated_tokens(request_outputs)
    tokens_per_second = total_tokens / latency if latency > 0 else 0.0

    logger.info(
        "Generated %s tokens across %s prompts in %.3f s (%.2f tokens/s)",
        total_tokens,
        len(prompts),
        latency,
        tokens_per_second,
    )

    return PerformanceResult(
        latency_s=latency,
        total_tokens=total_tokens,
        tokens_per_second=tokens_per_second,
        prompt_count=len(prompts),
        completions=completions,
    )


def read_prompts(path: Optional[Path]) -> List[str]:
    if path is None:
        return [
            "Write a short story about an AI assistant helping with deployment.",
            "Explain what makes the Qwen3 0.6B model notable in two sentences.",
        ]
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file {path} is empty")
    return [line for line in text.splitlines() if line.strip()]


def configure_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    return logging.getLogger("deploy_qwen3")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Qwen3 0.6B with vLLM and measure performance.")
    parser.add_argument("--prompt-file", type=Path, help="Path to a newline-delimited prompt file.", default=None)
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p value.")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
        help="Fraction of GPU memory that vLLM may allocate for KV cache and weights.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help=(
            "Upper bound on the number of concurrent sequences for KV cache allocation."
            " Reduce this if CUDA OOM occurs during warmup."
        ),
    )
    parser.add_argument("--dtype", type=str, default=None, help="Computation dtype, e.g. float16/bfloat16.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override model context length.")
    parser.add_argument("--speculative-model", type=str, default=None, help="Optional speculative decoding model ID.")
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable vLLM custom all-reduce kernels (useful for custom operator experiments).",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Force eager execution mode.")
    parser.add_argument("--log-level", type=str, choices=["info", "debug"], default="info")
    parser.add_argument("--dry-run", action="store_true", help="Load the model and apply modifiers without generating.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.log_level == "debug")

    config = DeploymentConfig(
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        speculative_model=args.speculative_model,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )

    try:
        llm = build_llm(config)
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.error("Failed to load vLLM model: %s", exc)
        return 1

    applied = apply_operator_modifiers(llm, logger)
    if not applied:
        logger.warning("Operator modifiers were registered but could not be applied. Continue with caution.")

    if args.dry_run:
        logger.info("Dry run complete; model loaded and modifiers applied.")
        return 0

    prompts = read_prompts(args.prompt_file)
    try:
        result = run_generation(
            llm,
            prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.error("Generation failed: %s", exc)
        return 1

    for idx, completion in enumerate(result.completions):
        logger.info("Prompt %d completion:\n%s", idx + 1, completion.strip())

    logger.info(
        "Latency: %.3f s | Tokens: %d | Tokens/s: %.2f",
        result.latency_s,
        result.total_tokens,
        result.tokens_per_second,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
