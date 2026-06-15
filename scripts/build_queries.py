from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_PERSONAS_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_QUERY_PLAN_DIR, DEFAULT_TOP_N_PRODUCTS
from demand_sim.preprocessing import build_query_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build product-persona LLM query plan.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_PRODUCTS)
    parser.add_argument("--sales", type=Path, default=None)
    parser.add_argument("--product-info", type=Path, default=None)
    parser.add_argument("--persona-prompts", type=Path, default=DEFAULT_PERSONAS_DIR / "persona_prompts.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-prices", type=int, default=10)
    parser.add_argument("--max-personas", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sales = args.sales or DEFAULT_PRODUCTS_DIR / f"sales_top{args.top_n}_online.csv"
    product_info = args.product_info or DEFAULT_PRODUCTS_DIR / f"product_info_top{args.top_n}_online.csv"
    output = args.output or DEFAULT_QUERY_PLAN_DIR / f"plan_top{args.top_n}_online.pkl"
    plan = build_query_plan(sales, product_info, args.persona_prompts, output, args.max_prices, args.max_personas)
    print(f"Wrote query plan with {len(plan)} rows to {output}.")


if __name__ == "__main__":
    main()
