from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_DATA_DIR, DEFAULT_PERSONAS_DIR
from demand_sim.preprocessing import PersonaPreprocessConfig, build_personas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build customer persona prompts from transaction history.")
    parser.add_argument("--transactions", type=Path, default=DEFAULT_DATA_DIR / "transactions_train.csv")
    parser.add_argument("--customers", type=Path, default=DEFAULT_DATA_DIR / "customers.csv")
    parser.add_argument("--articles", type=Path, default=DEFAULT_DATA_DIR / "articles.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PERSONAS_DIR)
    parser.add_argument("--train-start", default="2018-09-01")
    parser.add_argument("--train-end", default="2019-09-19")
    parser.add_argument("--n-personas", type=int, default=100)
    parser.add_argument("--top-taste-k", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PersonaPreprocessConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        n_personas=args.n_personas,
        top_taste_k=args.top_taste_k,
        chunksize=args.chunksize,
    )
    customer_features, personas = build_personas(args.transactions, args.customers, args.articles, args.output_dir, cfg)
    print(f"Wrote {len(customer_features)} customer feature rows.")
    print(f"Wrote {len(personas)} personas.")


if __name__ == "__main__":
    main()

