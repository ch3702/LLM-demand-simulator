from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales
from demand_sim.io import load_pickle, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate illustrative demand and pricing figures from a saved pricing ground-truth model.")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--article-id", type=int, default=562245001)
    parser.add_argument("--pricing-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "pricing")
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument("--product-info", type=Path, default=DEFAULT_PRODUCTS_DIR / "product_info_top100_online.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "illustrative use case")
    parser.add_argument("--cvar-level", type=float, default=0.25)
    parser.add_argument(
        "--demand-prices",
        type=float,
        nargs="+",
        default=None,
        help="Optional specific prices for demand-distribution plots. Defaults to min, middle, and max observed prices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.pricing_dir / f"split_{args.split:03d}" / "models" / "ground_truth_q_star.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved ground-truth model: {model_path}")
    model = load_pickle(model_path)

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    rows, _persona_ids = build_prompting_design_rows(sales, probabilities)
    product_rows = (
        rows[rows["article_id"] == args.article_id]
        .drop_duplicates(subset=["article_id", "offer_price"])
        .sort_values("offer_price")
        .reset_index(drop=True)
    )
    if product_rows.empty:
        raise ValueError(f"No prompting rows found for article_id={args.article_id}.")

    product_info = load_product_info(args.product_info, args.article_id)
    objective_values = pricing_objective_table(product_rows, model, args.cvar_level)
    demand_values = demand_distribution_table(product_rows, model, args.demand_prices)

    output_stem = f"split_{args.split:03d}_product_{args.article_id}"
    objective_values.to_csv(args.output_dir / f"pricing_values_{output_stem}.csv", index=False)
    demand_values.to_csv(args.output_dir / f"demand_distribution_values_{output_stem}.csv", index=False)
    plot_demand_distributions(
        demand_values=demand_values,
        output_dir=args.output_dir,
        output_stem=output_stem,
    )
    plot_pricing_curves(
        objective_values=objective_values,
        cvar_level=args.cvar_level,
        output_dir=args.output_dir,
        output_stem=output_stem,
    )
    write_json(
        {
            "split": args.split,
            "article_id": args.article_id,
            "product": product_info,
            "ground_truth_model": str(model_path),
            "selected_exposure_n": int(model.exposure_n),
            "cvar_level": args.cvar_level,
            "n_candidate_prices": int(len(product_rows)),
            "price_min": float(product_rows["offer_price"].min()),
            "price_max": float(product_rows["offer_price"].max()),
            "demand_plot_prices": sorted(demand_values["offer_price"].drop_duplicates().astype(float).tolist()),
        },
        args.output_dir / "metadata.json",
    )
    print(f"Wrote illustrative use-case figures and CSVs to {args.output_dir}.")


def load_product_info(path: Path, article_id: int) -> dict:
    if not path.exists():
        return {}
    info = pd.read_csv(path)
    matches = info[info["article_id"].astype(int) == int(article_id)]
    if matches.empty:
        return {}
    record = matches.iloc[0].to_dict()
    return {key: clean_json_value(value) for key, value in record.items()}


def clean_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def pricing_objective_table(rows: pd.DataFrame, model, cvar_level: float) -> pd.DataFrame:
    out = rows[["article_id", "offer_price"]].copy().reset_index(drop=True)
    price = out["offer_price"].to_numpy(float)
    purchase_prob = model.purchase_probability(rows)
    out["purchase_prob"] = purchase_prob
    out["expected_demand"] = model.exposure_n * purchase_prob
    out["expected_revenue"] = price * out["expected_demand"].to_numpy(float)
    out[f"cvar_{cvar_level:g}"] = revenue_cvar(price, model.exposure_n, purchase_prob, cvar_level)
    return out


def demand_distribution_table(rows: pd.DataFrame, model, demand_prices: list[float] | None = None) -> pd.DataFrame:
    price_grid = rows["offer_price"].to_numpy(float)
    if demand_prices is None:
        selected_indices = sorted(set([0, len(price_grid) // 2, len(price_grid) - 1]))
    else:
        selected_indices = []
        for price in demand_prices:
            matches = np.where(np.isclose(price_grid, float(price), rtol=0.0, atol=1e-8))[0]
            if len(matches) == 0:
                available = ", ".join(f"{value:g}" for value in price_grid)
                raise ValueError(f"Price {price:g} is not available for this product. Available prices: {available}")
            selected_indices.append(int(matches[0]))
        selected_indices = sorted(set(selected_indices))
    selected = rows.iloc[selected_indices].copy().reset_index(drop=True)
    purchase_prob = model.purchase_probability(selected)
    support = np.arange(0, model.exposure_n + 1, dtype=int)
    records: list[dict] = []
    for price, q in zip(selected["offer_price"].to_numpy(float), purchase_prob):
        pmf = binom.pmf(support, model.exposure_n, float(np.clip(q, 1e-12, 1.0 - 1e-12)))
        for demand, prob in zip(support, pmf):
            records.append(
                {
                    "article_id": int(selected["article_id"].iloc[0]),
                    "offer_price": float(price),
                    "demand": int(demand),
                    "probability": float(prob),
                    "purchase_prob": float(q),
                }
            )
    return pd.DataFrame(records)


def revenue_cvar(price: np.ndarray, exposure_n: int, purchase_prob: np.ndarray, level: float) -> np.ndarray:
    support = np.arange(0, exposure_n + 1, dtype=float)
    values = price[:, None] * support[None, :]
    q = np.clip(np.asarray(purchase_prob, dtype=float), 1e-12, 1.0 - 1e-12)
    pmf = binom.pmf(support[None, :], exposure_n, q[:, None])
    weights = np.minimum(pmf, np.maximum(level - np.cumsum(pmf, axis=1) + pmf, 0.0))
    return np.sum(weights * values, axis=1) / level


def plot_demand_distributions(demand_values: pd.DataFrame, output_dir: Path, output_stem: str) -> None:
    cutoff = demand_values.groupby("demand", as_index=False)["probability"].max()
    positive = cutoff[cutoff["probability"] > 1e-4]
    x_max = int(max(positive["demand"].max() if not positive.empty else 10, 10))
    plot_data = demand_values[demand_values["demand"] <= x_max].copy()
    y_max = float(plot_data["probability"].max() * 1.08)

    for price, group in plot_data.groupby("offer_price", sort=True):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        demand = group["demand"].to_numpy(float)
        probability = group["probability"].to_numpy(float)
        smooth_x = np.linspace(float(demand.min()), float(demand.max()), 400)
        smooth_y = PchipInterpolator(demand, probability)(smooth_x)
        ax.bar(
            demand,
            probability,
            width=0.86,
            color="#9ecae1",
            edgecolor="#6baed6",
            linewidth=0.35,
            alpha=0.75,
        )
        ax.plot(smooth_x, smooth_y, color="#2171b5", linewidth=1.9)
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(output_dir / f"demand_distribution_{output_stem}_price_{format_price(price)}.pdf")
        plt.close(fig)


def plot_pricing_curves(objective_values: pd.DataFrame, cvar_level: float, output_dir: Path, output_stem: str) -> None:
    cvar_col = f"cvar_{cvar_level:g}"
    y_max = float(objective_values[["expected_revenue", cvar_col]].to_numpy(float).max() * 1.08)
    for column, label, suffix in [
        ("expected_revenue", "Expected revenue", "expected_revenue"),
        (cvar_col, rf"$\mathrm{{CVaR}}_{{{cvar_level:g}}}$", f"cvar_{format_price(cvar_level)}"),
    ]:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        x = objective_values["offer_price"].to_numpy(float)
        y = objective_values[column].to_numpy(float)
        opt_idx = int(np.argmax(y))
        opt_price = float(x[opt_idx])
        opt_value = float(y[opt_idx])
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.35, color="#2171b5")
        ax.scatter([opt_price], [opt_value], color="#d62728", s=28, zorder=4)
        ax.vlines(opt_price, 0, opt_value, color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Price")
        ax.set_ylabel("")
        ax.set_ylim(0, y_max)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(output_dir / f"pricing_curve_{suffix}_{output_stem}.pdf")
        plt.close(fig)


def format_price(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
