import types
from pathlib import Path

import deploy_qwen3


class DummyModel:
    def __init__(self) -> None:
        self.modified = False


class DummyRunner:
    def __init__(self, model) -> None:
        self.model = model


class DummyDriver:
    def __init__(self, runner) -> None:
        self.model_runner = runner


class DummyExecutor:
    def __init__(self, driver) -> None:
        self.driver_worker = driver


class DummyEngine:
    def __init__(self, executor) -> None:
        self.model_executor = executor


class DummyLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.generate_called = 0

    def generate(self, prompts, _sampling_params):
        self.generate_called += 1
        return self.outputs


class DummySamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockOutput:
    def __init__(self, text: str, token_ids):
        self.outputs = [types.SimpleNamespace(text=text, token_ids=token_ids)]


class MockOutputNoTokens:
    def __init__(self, text: str):
        self.outputs = [types.SimpleNamespace(text=text, token_ids=None, token_ids_tensor=[1, 2, 3])]


def test_resolve_torch_model_success(monkeypatch):
    model = DummyModel()
    runner = DummyRunner(model)
    driver = DummyDriver(runner)
    executor = DummyExecutor(driver)
    engine = DummyEngine(executor)

    class LLMWithEngine:
        def __init__(self):
            self.llm_engine = engine

    result = deploy_qwen3.resolve_torch_model(LLMWithEngine())
    assert result is model


def test_apply_operator_modifiers(monkeypatch):
    model = DummyModel()
    runner = DummyRunner(model)
    driver = DummyDriver(runner)
    executor = DummyExecutor(driver)
    engine = DummyEngine(executor)

    class LLMWithEngine:
        def __init__(self):
            self.llm_engine = engine

    def modifier(mod):
        mod.modified = True

    registry = deploy_qwen3.OperatorModifierRegistry()
    registry.register("mark_modified", modifier)

    monkeypatch.setattr(deploy_qwen3, "operator_registry", registry)

    apply_result = deploy_qwen3.apply_operator_modifiers(LLMWithEngine())
    assert apply_result is True
    assert model.modified is True


def test_run_generation_collects_metrics(monkeypatch):
    outputs = [MockOutput("hello", [1, 2, 3]), MockOutputNoTokens("world")]
    llm = DummyLLM(outputs)

    perf_counter_values = iter([0.0, 0.5])
    monkeypatch.setattr(deploy_qwen3, "SamplingParams", DummySamplingParams)
    monkeypatch.setattr(deploy_qwen3, "ensure_vllm_imported", lambda: None)
    monkeypatch.setattr(deploy_qwen3.time, "perf_counter", lambda: next(perf_counter_values))

    result = deploy_qwen3.run_generation(llm, ["hi", "there"], 32, 0.8, 0.9)

    assert llm.generate_called == 1
    assert result.prompt_count == 2
    assert result.total_tokens == 6  # 3 tokens from first, 3 fallback tokens from second
    assert result.latency_s == 0.5
    assert result.tokens_per_second == 12.0
    assert result.completions == ["hello", "world"]


def test_read_prompts(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("first\n\nsecond\n", encoding="utf-8")
    prompts = deploy_qwen3.read_prompts(prompt_file)
    assert prompts == ["first", "second"]


def test_read_prompts_default():
    prompts = deploy_qwen3.read_prompts(None)
    assert len(prompts) >= 1
