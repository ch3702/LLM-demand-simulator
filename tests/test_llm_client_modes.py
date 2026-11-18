"""LLM 키·엔드포인트 설정 단위 테스트 (base_url·chat 모드·.env)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd  # noqa: PANDAS_OK — 레포 표준 데이터 파이프라인
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demand_sim import llm as llm_module
from demand_sim.llm import LLMConfig, LLMConfigError, content_for_chat, run_simulation_sync

SCRIPT_PATH = REPO_ROOT / "scripts" / "run_llm_simulations.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_llm_simulations", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestModeResolution:
    def test_auto_without_base_url_uses_responses(self):
        assert LLMConfig().resolved_mode() == "responses"

    def test_auto_with_openai_url_uses_responses(self):
        cfg = LLMConfig(base_url="https://api.openai.com/v1")
        assert cfg.resolved_mode() == "responses"

    def test_auto_with_local_server_uses_chat(self):
        cfg = LLMConfig(base_url="http://127.0.0.1:11434/v1")
        assert cfg.resolved_mode() == "chat"

    def test_explicit_mode_overrides_detection(self):
        cfg = LLMConfig(base_url="http://127.0.0.1:11434/v1", api_mode="responses")
        assert cfg.resolved_mode() == "responses"

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(LLMConfigError):
            LLMConfig(api_mode="banana").resolved_mode()


def _fake_plan() -> pd.DataFrame:
    return pd.DataFrame([{
        "persona_id": "p1",
        "article_id": 7,
        "price_sig": "sig-1",
        "prices_json": '{"a": 1}',
        "prompt": "구매 의향을 0~1로 답하세요.",
    }])


class _FakeChatCompletions:
    def __init__(self, sink):
        self.sink = sink

    def create(self, model, messages):
        self.sink.append({"mode": "chat", "model": model, "messages": messages})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="0.7"))])


class _FakeResponses:
    def __init__(self, sink):
        self.sink = sink

    def create(self, model, input):
        self.sink.append({"mode": "responses", "model": model, "input": input})
        return SimpleNamespace(output_text="0.8")


class _FakeClient:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeClient.last_kwargs = kwargs
        self.calls: list = []
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(self.calls))
        self.responses = _FakeResponses(self.calls)


class TestSyncRun:
    def test_chat_mode_uses_openai_compatible_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_module, "OpenAI", _FakeClient)
        output = tmp_path / "out.csv"
        cfg = LLMConfig(model="glm-5.3-flash:cloud", base_url="http://127.0.0.1:11434/v1")
        run_simulation_sync(_fake_plan(), "ollama", output, tmp_path, cfg)
        assert _FakeClient.last_kwargs == {
            "api_key": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
        }
        assert "0.7" in output.read_text(encoding="utf-8")

    def test_responses_mode_keeps_official_api(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_module, "OpenAI", _FakeClient)
        output = tmp_path / "out.csv"
        run_simulation_sync(_fake_plan(), "sk-test", output, tmp_path, LLMConfig(model="gpt-5-mini"))
        assert _FakeClient.last_kwargs == {"api_key": "sk-test"}
        assert "0.8" in output.read_text(encoding="utf-8")


class TestContent:
    def test_chat_content_without_image(self):
        content = content_for_chat("질문", None)
        assert content == [{"type": "text", "text": "질문"}]

    def test_chat_content_with_image_embeds_data_url(self, tmp_path):
        image = tmp_path / "7.jpg"
        image.write_bytes(b"fake-jpeg")
        content = content_for_chat("질문", image)
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


class TestDotenvAndKey:
    def test_load_dotenv_reads_values_without_overriding(self, tmp_path, monkeypatch):
        script = _load_script()
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\nWISE_LLM_BASE_URL=http://127.0.0.1:11434/v1\nPRESET=from-file\nQUOTED='값'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("PRESET", "from-env")
        monkeypatch.delenv("WISE_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("QUOTED", raising=False)
        script.load_dotenv(env_file)
        assert os.environ["WISE_LLM_BASE_URL"] == "http://127.0.0.1:11434/v1"
        assert os.environ["PRESET"] == "from-env"
        assert os.environ["QUOTED"] == "값"

    def test_key_falls_back_to_placeholder_for_local_endpoint(self, monkeypatch):
        script = _load_script()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        args = SimpleNamespace(api_key_env="OPENAI_API_KEY")
        assert script.resolve_api_key(args, "http://127.0.0.1:11434/v1") == "ollama"

    def test_missing_key_without_base_url_exits(self, monkeypatch):
        script = _load_script()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        args = SimpleNamespace(api_key_env="OPENAI_API_KEY")
        with pytest.raises(SystemExit):
            script.resolve_api_key(args, None)

    def test_base_url_env_wins_over_nothing_and_cli_wins_over_env(self, monkeypatch):
        script = _load_script()
        monkeypatch.setenv("WISE_LLM_BASE_URL", "http://env:11434/v1")
        assert script.resolve_base_url(None) == "http://env:11434/v1"
        assert script.resolve_base_url("http://cli:11434/v1") == "http://cli:11434/v1"
