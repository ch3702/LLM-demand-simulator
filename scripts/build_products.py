from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_DATA_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_TOP_N_PRODUCTS
from demand_sim.preprocessing import ProductPreprocessConfig, build_products


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build top-product daily demand and product metadata.")
    parser.add_argument("--transactions", type=Path, default=DEFAULT_DATA_DIR / "transactions_train.csv")
    parser.add_argument("--articles", type=Path, default=DEFAULT_DATA_DIR / "articles.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PRODUCTS_DIR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_PRODUCTS)
    parser.add_argument("--all-channels", action="store_true", help="Do not restrict to online channel.")
    parser.add_argument("--product-type", default="Trousers")
    parser.add_argument("--cutoff-date", default="2019-09-20")
    parser.add_argument("--min-distinct-prices", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProductPreprocessConfig(
        top_n=args.top_n,
        online_only=not args.all_channels,
        product_type=args.product_type,
        cutoff_date=args.cutoff_date,
        min_distinct_prices=args.min_distinct_prices,
    )
    sales, product_info = build_products(args.transactions, args.articles, args.output_dir, cfg)
    print(f"Wrote {len(sales)} demand rows for {sales['article_id'].nunique()} products.")
    print(f"Wrote product metadata for {len(product_info)} products.")


if __name__ == "__main__":
    main()

