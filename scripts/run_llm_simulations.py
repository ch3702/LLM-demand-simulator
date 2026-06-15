from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_IMAGES_DIR, DEFAULT_QUERY_PLAN_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.llm import LLMConfig, run_simulation_async, run_simulation_sync


# Set OpenAI key here if you run this file from an IDE
# Prefer setting OPENAI_API_KEY in the environment
OPENAI_API_KEY = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM purchase-probability simulations.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_QUERY_PLAN_DIR / "plan_top100_online.pkl")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--requests-per-minute", type=int, default=60)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--n-draws", type=int, default=1)
    parser.add_argument("--sync", action="store_true", help="Use synchronous requests instead of async.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env) or OPENAI_API_KEY
    if not api_key:
        raise SystemExit(
            f"Set {args.api_key_env} before running simulations, or fill in OPENAI_API_KEY at the top of this script."
        )
    plan = pd.read_pickle(args.plan)
    cfg = LLMConfig(
        model=args.model,
        requests_per_minute=args.requests_per_minute,
        max_concurrency=args.max_concurrency,
        n_draws=args.n_draws,
    )
    if args.sync:
        run_simulation_sync(plan, api_key, args.output, args.images_dir, cfg)
    else:
        asyncio.run(run_simulation_async(plan, api_key, args.output, args.images_dir, cfg))


if __name__ == "__main__":
    main()
