from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR


@dataclass(frozen=True)
class HotelInputConfig:
    start_date: str = "2026-11-15"
    end_date: str = "2026-12-31"
    peak_date: str = "2026-12-24"
    peak_bump: float = 1.8
    peak_sigma_days: float = 9.0
    season_start_level: float = 0.8
    weekend_multipliers: tuple[float, ...] = (1.0, 0.9, 0.95, 1.0, 1.25, 1.45, 1.2)
    price_elasticity: float = -1.2
    seed: int = 20251224
    suffix: str = "hotel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Grand Hyatt Christmas SKU sales and 2030 persona contract files.")
    parser.add_argument("--skus", type=Path, default=Path("data/hotel_skus.csv"))
    parser.add_argument("--personas", type=Path, default=Path("data/hotel_personas.csv"))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--peak-date", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--products-dir", type=Path, default=DEFAULT_PRODUCTS_DIR)
    parser.add_argument("--personas-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "personas_hotel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {key: value for key, value in {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "peak_date": args.peak_date,
        "seed": args.seed,
        "suffix": args.suffix,
    }.items() if value is not None}
    cfg = HotelInputConfig(**overrides)

    skus = pd.read_csv(args.skus)
    personas = pd.read_csv(args.personas)
    required_sku_cols = {"article_id", "sku_code", "prod_name", "product_type_name", "colour_group_name", "detail_desc", "price_grid", "base_daily_demand"}
    missing = required_sku_cols.difference(skus.columns)
    if missing:
        raise ValueError(f"SKU file {args.skus} is missing columns: {sorted(missing)}")
    required_persona_cols = {"persona_id", "age_bin", "engagement_bin", "price_tier", "taste_bucket", "n_customers", "txn_q25", "txn_q75", "price_q25", "price_q75", "persona_context"}
    missing = required_persona_cols.difference(personas.columns)
    if missing:
        raise ValueError(f"Persona file {args.personas} is missing columns: {sorted(missing)}")

    sales = build_sales(skus, cfg)
    product_info = build_product_info(skus)
    cells = build_persona_cells(personas, f"{pd.Timestamp(cfg.peak_date).year}")

    args.products_dir.mkdir(parents=True, exist_ok=True)
    args.personas_dir.mkdir(parents=True, exist_ok=True)
    top_n = len(skus)
    sales_path = args.products_dir / f"sales_top{top_n}_{cfg.suffix}.csv"
    info_path = args.products_dir / f"product_info_top{top_n}_{cfg.suffix}.csv"
    sales.to_csv(sales_path, index=False)
    product_info.to_csv(info_path, index=False)
    cells.to_csv(args.personas_dir / "persona_cells.csv", index=False)
    cells[["persona_id", "n_customers", "persona_prompt", "age_bin", "engagement_bin", "price_tier", "taste_bucket"]].to_csv(
        args.personas_dir / "persona_prompts.csv", index=False
    )

    print(f"Wrote {len(sales)} proxy demand rows for {top_n} SKUs -> {sales_path}")
    print(f"Wrote product metadata -> {info_path}")
    print(f"Wrote {len(cells)} hotel personas -> {args.personas_dir / 'persona_prompts.csv'}")
    sampled = sales.groupby("article_id")["offer_price"].nunique()
    too_few = sampled[sampled < 3]
    if not too_few.empty:
        print(f"WARNING: SKUs with fewer than 3 sampled prices: {too_few.to_dict()}")


def build_sales(skus: pd.DataFrame, cfg: HotelInputConfig) -> pd.DataFrame:
    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    peak = pd.Timestamp(cfg.peak_date)
    ramp = cfg.season_start_level + (1.0 - cfg.season_start_level) * np.linspace(0.0, 1.0, len(dates))
    bump = cfg.peak_bump * np.exp(-((dates - peak).days.astype(float) ** 2) / (2.0 * cfg.peak_sigma_days ** 2))
    seasonal = ramp + bump
    weekday_mult = np.array([cfg.weekend_multipliers[d.weekday()] for d in dates])

    rng = np.random.default_rng(cfg.seed)
    frames: list[pd.DataFrame] = []
    for sku in skus.itertuples(index=False):
        grid = sorted(float(price) for price in str(sku.price_grid).split("|") if str(price).strip())
        if len(grid) < 2:
            raise ValueError(f"SKU {sku.article_id} needs at least 2 prices in price_grid.")
        weights = np.array([0.8 ** idx for idx in range(len(grid))])
        weights /= weights.sum()
        picked = rng.choice(len(grid), size=len(dates), p=weights)
        reference = grid[0]
        price_effect = (np.array([grid[i] for i in picked]) / reference) ** cfg.price_elasticity
        mean_demand = sku.base_daily_demand * seasonal * weekday_mult * price_effect
        demand = rng.poisson(np.clip(mean_demand, 0.0, None))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "article_id": int(sku.article_id),
                    "offer_price": [grid[i] for i in picked],
                    "demand": demand.astype(int),
                }
            )
        )
    sales = pd.concat(frames, ignore_index=True)
    sales["offer_price"] = sales["offer_price"].round(2)
    return sales.sort_values(["article_id", "date", "offer_price"]).reset_index(drop=True)


def build_product_info(skus: pd.DataFrame) -> pd.DataFrame:
    info = skus.copy()
    if "graphical_appearance_name" not in info.columns:
        info["graphical_appearance_name"] = ""
    info["article_id"] = info["article_id"].astype(int)
    return info.sort_values("article_id").reset_index(drop=True)


def build_persona_cells(personas: pd.DataFrame, season_label: str) -> pd.DataFrame:
    cells = personas.sort_values("persona_id").reset_index(drop=True).copy()
    cells["persona_prompt"] = cells.apply(_persona_prompt, axis=1, season_label=season_label)
    return cells


def _persona_prompt(row: pd.Series, season_label: str) -> str:
    return (
        "You are a customer of Grand Hyatt Seoul during the "
        f"{season_label} Christmas season.\n"
        f"{str(row['persona_context']).strip()}\n"
        f"Your age group is {row['age_bin']}. "
        f"During the holiday season you typically visit the hotel about {float(row['txn_q25']):g} to {float(row['txn_q75']):g} times per month, "
        f"and your typical spend per visit is between {float(row['price_q25']):.0f} and {float(row['price_q75']):.0f} KRW. "
        f"The categories you are most interested in are: {row['taste_bucket']}.\n\n"
        "Task: Given a product and a list of prices in KRW (Korean won), return the probability you would buy the product at each price. "
        "Output JSON exactly: {\"prices\": [...], \"p_buy\": [...], \"reason\": \"<=30 words\"}."
    )


if __name__ == "__main__":
    main()
