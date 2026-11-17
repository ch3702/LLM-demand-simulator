from __future__ import annotations

import argparse
import asyncio  # noqa: ANYIO_OK — OpenAI AsyncOpenAI 경로에 필요
import os
import sys
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK — 레포 표준 데이터 파이프라인

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_IMAGES_DIR, DEFAULT_QUERY_PLAN_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import load_plan
from demand_sim.llm import LLMConfig, run_simulation_async, run_simulation_sync

DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"

# Set OpenAI key here if you run this file from an IDE
# Prefer setting OPENAI_API_KEY in the environment or .env (see .env.example)
OPENAI_API_KEY = ""


def load_dotenv(path: Path) -> None:
    """간단한 .env 로더 — 이미 설정된 환경변수는 덮어쓰지 않는다."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM purchase-probability simulations.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_QUERY_PLAN_DIR / "plan_top100_online.pkl", help="플랜 파일(.pkl 또는 .json)")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--model",
        default=os.environ.get("WISE_LLM_MODEL") or os.environ.get("WISE_OLLAMA_MODEL") or "gpt-5-mini",
    )
    parser.add_argument("--requests-per-minute", type=int, default=60)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--n-draws", type=int, default=1)
    parser.add_argument("--sync", action="store_true", help="Use synchronous requests instead of async.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI 호환 엔드포인트(예: 로컬 Ollama http://127.0.0.1:11434/v1). 미지정 시 WISE_LLM_BASE_URL 사용.",
    )
    parser.add_argument(
        "--api-mode",
        default="auto",
        choices=("auto", "responses", "chat"),
        help="auto: OpenAI 공식이면 responses, 호환 서버면 chat.",
    )
    return parser.parse_args()


def resolve_base_url(cli_value: str | None) -> str | None:
    return cli_value or os.environ.get("WISE_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")


def resolve_api_key(args: argparse.Namespace, base_url: str | None) -> str:
    api_key = os.environ.get(args.api_key_env) or OPENAI_API_KEY
    if api_key:
        return api_key
    if base_url:
        return "ollama"  # OpenAI 호환 로컬 서버는 키를 검증하지 않는다
    raise SystemExit(
        "LLM API 키가 없습니다. OPENAI_API_KEY 를 설정하거나, "
        "로컬 Ollama를 쓰려면 .env 에 WISE_LLM_BASE_URL=http://127.0.0.1:11434/v1 과 "
        "WISE_LLM_MODEL 을 지정하세요(.env.example 참고)."
    )


def main() -> None:
    load_dotenv(DOTENV_PATH)
    args = parse_args()
    base_url = resolve_base_url(args.base_url)
    api_key = resolve_api_key(args, base_url)
    plan = load_plan(args.plan)
    cfg = LLMConfig(
        model=args.model,
        requests_per_minute=args.requests_per_minute,
        max_concurrency=args.max_concurrency,
        n_draws=args.n_draws,
        base_url=base_url,
        api_mode=args.api_mode,
    )
    print(f"LLM 모드: {cfg.resolved_mode()} · 모델: {cfg.model} · 엔드포인트: {base_url or 'api.openai.com'}")
    if args.sync:
        run_simulation_sync(plan, api_key, args.output, args.images_dir, cfg)
    else:
        asyncio.run(run_simulation_async(plan, api_key, args.output, args.images_dir, cfg))


if __name__ == "__main__":
    main()
