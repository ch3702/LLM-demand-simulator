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
from scipy.stats import binom
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCTS_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import build_prompting_design_rows, load_probability_rows, load_sales
from demand_sim.io import load_pickle, save_pickle, write_json
from demand_sim.models.llm_mix_cal import LLMMixCalModel, fit_llm_mix_cal


M = 10
TRAIN_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
EXPOSURE_N_VALUES = [100, 150, 200, 250]
CVaR_LEVEL = 0.25
FIT_EXISTING_FRACTIONS = True
REGENERATE_EXISTING_SYNTHETIC = True
OVERWRITE_EXISTING_FRACTIONS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pricing sample efficiency for llm-mix-cal.")
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "pricing")
    parser.add_argument("--n-splits", type=int, default=M)
    parser.add_argument("--train-fractions", type=float, nargs="+", default=TRAIN_FRACTIONS)
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.60, 0.25, 0.15])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--exposure-n-values", type=int, nargs="+", default=EXPOSURE_N_VALUES)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--logit-calibration-iters", type=int, default=6)
    parser.add_argument("--cvar-level", type=float, default=CVaR_LEVEL)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Refresh aggregate CSVs and plots from existing split outputs without refitting.",
    )
    parser.add_argument(
        "--fit-existing-fractions",
        action=argparse.BooleanOptionalAction,
        default=FIT_EXISTING_FRACTIONS,
        help="Fit requested train fractions for existing splits, reusing saved ground truth and synthetic data.",
    )
    parser.add_argument(
        "--regenerate-existing-synthetic",
        action=argparse.BooleanOptionalAction,
        default=REGENERATE_EXISTING_SYNTHETIC,
        help="Regenerate D2/D3 for existing splits from saved ground truth models before fitting requested fractions.",
    )
    parser.add_argument(
        "--overwrite-existing-fractions",
        action=argparse.BooleanOptionalAction,
        default=OVERWRITE_EXISTING_FRACTIONS,
        help="Overwrite requested fraction outputs when fitting existing splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_existing:
        write_aggregate_outputs(args.output_dir, args.cvar_level)
        print(f"Refreshed pricing aggregate outputs in {args.output_dir} without refitting.")
        return

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    prompting_rows, persona_ids = build_prompting_design_rows(sales, probabilities)
    product_ids = np.array(sorted(prompting_rows["article_id"].unique()), dtype=int)

    if args.fit_existing_fractions or args.regenerate_existing_synthetic:
        fit_existing_fractions(args, prompting_rows, persona_ids)
        print(f"Fit requested train fractions for existing pricing splits in {args.output_dir}.")
        return

    start_split = next_split_index(args.output_dir)
    split_records: list[dict] = []

    for split_idx in tqdm(range(start_split, start_split + args.n_splits), desc="pricing splits"):
        split_dir = args.output_dir / f"split_{split_idx:03d}"
        model_dir = split_dir / "models"
        synthetic_dir = split_dir / "synthetic"
        model_dir.mkdir(parents=True, exist_ok=True)
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        j1_ids, j2_ids, j3_ids = split_products_three_way(product_ids, args.split_ratio, args.seed + split_idx)
        j1_rows = prompting_rows[prompting_rows["article_id"].isin(j1_ids)].copy()
        j2_rows = prompting_rows[prompting_rows["article_id"].isin(j2_ids)].copy()
        j3_rows = prompting_rows[prompting_rows["article_id"].isin(j3_ids)].copy()

        pd.Series(j1_ids, name="article_id").to_csv(split_dir / "products_j1.csv", index=False)
        pd.Series(j2_ids, name="article_id").to_csv(split_dir / "products_j2.csv", index=False)
        pd.Series(j3_ids, name="article_id").to_csv(split_dir / "products_j3.csv", index=False)

        ground_truth = fit_llm_mix_cal(
            j1_rows,
            persona_ids,
            args.exposure_n_values,
            fit_objective="truncated",
            calibration_iters=args.logit_calibration_iters,
            solver=args.solver,
        )
        save_model_outputs(ground_truth, model_dir, "ground_truth_q_star")

        rng = np.random.default_rng(args.seed + 100_000 + split_idx)
        d2_full = synthetic_demand_rows(j2_rows, ground_truth, rng)
        d3 = synthetic_demand_rows(j3_rows, ground_truth, rng)
        d2_full.to_csv(synthetic_dir / "synthetic_d2_full.csv", index=False)
        d3.to_csv(synthetic_dir / "synthetic_d3.csv", index=False)

        split_records.append(
            {
                "split": split_idx,
                "n_j1_products": int(len(j1_ids)),
                "n_j2_products": int(len(j2_ids)),
                "n_j3_products": int(len(j3_ids)),
                "n_j1_rows": int(len(j1_rows)),
                "n_d2_rows": int(len(d2_full)),
                "n_d3_rows": int(len(d3)),
                "ground_truth_selected_exposure_n": int(ground_truth.exposure_n),
                "ground_truth_objective_value": float(ground_truth.objective_value),
            }
        )

        fraction_summaries: list[pd.DataFrame] = []
        for train_frac in tqdm(args.train_fractions, desc=f"split {split_idx:03d} train fractions", leave=False):
            frac_label = format_fraction(train_frac)
            frac_dir = split_dir / f"frac_{frac_label}"
            frac_model_dir = frac_dir / "models"
            frac_dir.mkdir(parents=True, exist_ok=True)
            frac_model_dir.mkdir(parents=True, exist_ok=True)

            d2_subset = subsample_rows(d2_full, train_frac, args.seed + 200_000 + 997 * split_idx)
            d2_subset.to_csv(frac_dir / "synthetic_d2_train.csv", index=False)

            q_hat = fit_llm_mix_cal(
                d2_subset,
                persona_ids,
                args.exposure_n_values,
                fit_objective="naive",
                calibration_iters=args.logit_calibration_iters,
                solver=args.solver,
            )
            save_model_outputs(q_hat, frac_model_dir, "q_hat")

            by_product = evaluate_pricing_regret(
                grid_rows=d3,
                ground_truth=ground_truth,
                policy=q_hat,
                split_idx=split_idx,
                train_frac=train_frac,
                n_train_rows=len(d2_subset),
                cvar_level=args.cvar_level,
            )
            summary = summarize_pricing_regret(by_product)
            by_product.to_csv(frac_dir / "pricing_by_product.csv", index=False)
            summary.to_csv(frac_dir / "pricing_summary.csv", index=False)
            fraction_summaries.append(summary)

        if fraction_summaries:
            pd.concat(fraction_summaries, ignore_index=True).to_csv(split_dir / "pricing_summary.csv", index=False)

        combine_split_summary(args.output_dir, split_records).to_csv(args.output_dir / "split_summary.csv", index=False)
        write_aggregate_outputs(args.output_dir, args.cvar_level)

    split_summary = combine_split_summary(args.output_dir, split_records)
    split_summary.to_csv(args.output_dir / "split_summary.csv", index=False)
    write_aggregate_outputs(args.output_dir, args.cvar_level)
    write_json(
        {
            "sales": str(args.sales),
            "responses": str(args.responses),
            "n_splits_requested": args.n_splits,
            "total_completed_splits": int(len(split_summary)),
            "new_split_start": int(start_split),
            "new_split_end": int(start_split + args.n_splits - 1) if args.n_splits > 0 else int(start_split - 1),
            "split_ratio": args.split_ratio,
            "train_fractions": args.train_fractions,
            "seed": args.seed,
            "ground_truth_fit_objective": "truncated",
            "q_hat_fit_objective": "naive",
            "exposure_n_values": args.exposure_n_values,
            "cvar_level": args.cvar_level,
        },
        args.output_dir / "metadata.json",
    )
    print(f"Wrote pricing sample-efficiency outputs to {args.output_dir}. Total completed splits: {len(split_summary)}.")


def validate_args(args: argparse.Namespace) -> None:
    if args.n_splits < 0:
        raise SystemExit("--n-splits must be nonnegative.")
    if len(args.split_ratio) != 3 or any(value <= 0 for value in args.split_ratio):
        raise SystemExit("--split-ratio must contain three positive values.")
    if not np.isclose(sum(args.split_ratio), 1.0):
        raise SystemExit("--split-ratio must sum to 1.")
    if any(frac <= 0 or frac > 1 for frac in args.train_fractions):
        raise SystemExit("--train-fractions must be in (0, 1].")
    if not 0 < args.cvar_level <= 1:
        raise SystemExit("--cvar-level must be in (0, 1].")


def split_products_three_way(product_ids: np.ndarray, ratios: list[float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(product_ids, copy=True)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_j1 = int(round(ratios[0] * n_total))
    n_j2 = int(round(ratios[1] * n_total))
    n_j1 = min(max(n_j1, 1), n_total - 2)
    n_j2 = min(max(n_j2, 1), n_total - n_j1 - 1)
    j1 = np.sort(shuffled[:n_j1])
    j2 = np.sort(shuffled[n_j1 : n_j1 + n_j2])
    j3 = np.sort(shuffled[n_j1 + n_j2 :])
    return j1, j2, j3


def synthetic_demand_rows(rows: pd.DataFrame, model: LLMMixCalModel, rng: np.random.Generator) -> pd.DataFrame:
    synthetic = rows.copy().reset_index(drop=True)
    synthetic["observed_demand"] = synthetic["demand"].astype(int)
    purchase_prob = model.purchase_probability(synthetic)
    synthetic["ground_truth_purchase_prob"] = purchase_prob
    synthetic["demand"] = rng.binomial(model.exposure_n, purchase_prob).astype(int)
    return synthetic


def subsample_rows(rows: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    if np.isclose(train_frac, 1.0):
        return rows.copy().reset_index(drop=True)
    n_rows = max(1, int(np.ceil(train_frac * len(rows))))
    sampled = rows.sample(n=n_rows, replace=False, random_state=seed)
    return sampled.sort_index().reset_index(drop=True)


def fit_existing_fractions(args: argparse.Namespace, prompting_rows: pd.DataFrame, persona_ids: list[str]) -> None:
    split_indices = existing_split_indices(args.output_dir)
    if not split_indices:
        raise RuntimeError(f"No existing split directories found under {args.output_dir}.")

    for split_idx in tqdm(split_indices, desc="existing pricing splits"):
        split_dir = args.output_dir / f"split_{split_idx:03d}"
        model_path = split_dir / "models" / "ground_truth_q_star.pkl"
        d2_path = split_dir / "synthetic" / "synthetic_d2_full.csv"
        d3_path = split_dir / "synthetic" / "synthetic_d3.csv"
        missing = [str(path) for path in [model_path, d2_path, d3_path] if not path.exists()]
        if missing and not args.regenerate_existing_synthetic:
            raise RuntimeError(f"Split {split_idx:03d} is missing required files: {missing}")

        ground_truth = load_pickle(model_path)
        if args.regenerate_existing_synthetic:
            j2_ids = pd.read_csv(split_dir / "products_j2.csv")["article_id"].astype(int)
            j3_ids = pd.read_csv(split_dir / "products_j3.csv")["article_id"].astype(int)
            j2_rows = prompting_rows[prompting_rows["article_id"].isin(j2_ids)].copy()
            j3_rows = prompting_rows[prompting_rows["article_id"].isin(j3_ids)].copy()
            rng = np.random.default_rng(args.seed + 100_000 + split_idx)
            d2_full = synthetic_demand_rows(j2_rows, ground_truth, rng)
            d3 = synthetic_demand_rows(j3_rows, ground_truth, rng)
            d2_path.parent.mkdir(parents=True, exist_ok=True)
            d2_full.to_csv(d2_path, index=False)
            d3.to_csv(d3_path, index=False)
        else:
            d2_full = pd.read_csv(d2_path)
            d3 = pd.read_csv(d3_path)
        fraction_summaries: list[pd.DataFrame] = []

        for train_frac in tqdm(args.train_fractions, desc=f"split {split_idx:03d} train fractions", leave=False):
            frac_label = format_fraction(train_frac)
            frac_dir = split_dir / f"frac_{frac_label}"
            summary_path = frac_dir / "pricing_summary.csv"
            if summary_path.exists() and not args.overwrite_existing_fractions:
                fraction_summaries.append(pd.read_csv(summary_path))
                continue

            frac_model_dir = frac_dir / "models"
            frac_dir.mkdir(parents=True, exist_ok=True)
            frac_model_dir.mkdir(parents=True, exist_ok=True)

            d2_subset = subsample_rows(d2_full, train_frac, args.seed + 200_000 + 997 * split_idx)
            d2_subset.to_csv(frac_dir / "synthetic_d2_train.csv", index=False)
            q_hat = fit_llm_mix_cal(
                d2_subset,
                persona_ids,
                args.exposure_n_values,
                fit_objective="naive",
                calibration_iters=args.logit_calibration_iters,
                solver=args.solver,
            )
            save_model_outputs(q_hat, frac_model_dir, "q_hat")

            by_product = evaluate_pricing_regret(
                grid_rows=d3,
                ground_truth=ground_truth,
                policy=q_hat,
                split_idx=split_idx,
                train_frac=train_frac,
                n_train_rows=len(d2_subset),
                cvar_level=args.cvar_level,
            )
            summary = summarize_pricing_regret(by_product)
            by_product.to_csv(frac_dir / "pricing_by_product.csv", index=False)
            summary.to_csv(summary_path, index=False)
            fraction_summaries.append(summary)

        if fraction_summaries:
            pd.concat(fraction_summaries, ignore_index=True).to_csv(split_dir / "pricing_summary.csv", index=False)
        write_aggregate_outputs(args.output_dir, args.cvar_level)

    write_aggregate_outputs(args.output_dir, args.cvar_level)


def evaluate_pricing_regret(
    grid_rows: pd.DataFrame,
    ground_truth: LLMMixCalModel,
    policy: LLMMixCalModel,
    split_idx: int,
    train_frac: float,
    n_train_rows: int,
    cvar_level: float,
) -> pd.DataFrame:
    grid = grid_rows.drop_duplicates(subset=["article_id", "offer_price"]).copy().reset_index(drop=True)
    gt_objectives = objective_table(grid, ground_truth, cvar_level, prefix="gt")
    policy_objectives = objective_table(grid, policy, cvar_level, prefix="policy")
    merged = gt_objectives.merge(policy_objectives, on=["article_id", "offer_price"], how="inner")
    records: list[dict] = []
    for article_id, product_rows in merged.groupby("article_id", sort=True):
        for objective in ["expected_revenue", f"cvar_{format_metric_level(cvar_level)}"]:
            gt_col = f"gt_{objective}"
            policy_col = f"policy_{objective}"
            oracle = product_rows.loc[product_rows[gt_col].idxmax()]
            chosen = product_rows.loc[product_rows[policy_col].idxmax()]
            best_value = float(oracle[gt_col])
            value_at_policy = float(chosen[gt_col])
            absolute_regret = max(0.0, best_value - value_at_policy)
            relative_performance = value_at_policy / best_value if best_value > 0 else np.nan
            records.append(
                {
                    "split": split_idx,
                    "train_frac": float(train_frac),
                    "n_train_rows": int(n_train_rows),
                    "objective": objective,
                    "article_id": int(article_id),
                    "n_candidate_prices": int(len(product_rows)),
                    "ground_truth_optimal_price": float(oracle["offer_price"]),
                    "policy_optimal_price": float(chosen["offer_price"]),
                    "ground_truth_best_value": best_value,
                    "ground_truth_value_at_policy_price": value_at_policy,
                    "absolute_regret": float(absolute_regret),
                    "relative_performance": float(relative_performance) if np.isfinite(relative_performance) else np.nan,
                    "price_match": bool(np.isclose(float(oracle["offer_price"]), float(chosen["offer_price"]))),
                    "policy_selected_exposure_n": int(policy.exposure_n),
                    "ground_truth_selected_exposure_n": int(ground_truth.exposure_n),
                }
            )
    return pd.DataFrame(records)


def objective_table(rows: pd.DataFrame, model: LLMMixCalModel, cvar_level: float, prefix: str) -> pd.DataFrame:
    out = rows[["article_id", "offer_price"]].copy().reset_index(drop=True)
    price = out["offer_price"].to_numpy(float)
    purchase_prob = model.purchase_probability(rows)
    out[f"{prefix}_purchase_prob"] = purchase_prob
    out[f"{prefix}_expected_revenue"] = price * model.exposure_n * purchase_prob
    out[f"{prefix}_cvar_{format_metric_level(cvar_level)}"] = revenue_cvar(
        price=price,
        exposure_n=model.exposure_n,
        purchase_prob=purchase_prob,
        level=cvar_level,
    )
    return out


def revenue_cvar(
    price: np.ndarray,
    exposure_n: int,
    purchase_prob: np.ndarray,
    level: float,
) -> np.ndarray:
    support = np.arange(0, exposure_n + 1, dtype=float)
    values = price[:, None] * support[None, :]
    q = np.clip(np.asarray(purchase_prob, dtype=float), 1e-12, 1.0 - 1e-12)
    pmf = binom.pmf(support[None, :], exposure_n, q[:, None])
    weights = np.minimum(pmf, np.maximum(level - np.cumsum(pmf, axis=1) + pmf, 0.0))
    return np.sum(weights * values, axis=1) / level


def summarize_pricing_regret(by_product: pd.DataFrame) -> pd.DataFrame:
    return (
        by_product.groupby(["split", "train_frac", "n_train_rows", "objective"], as_index=False)
        .agg(
            n_products=("article_id", "nunique"),
            mean_regret=("absolute_regret", "mean"),
            median_regret=("absolute_regret", "median"),
            std_regret=("absolute_regret", "std"),
            mean_relative_performance=("relative_performance", "mean"),
            price_match_rate=("price_match", "mean"),
            mean_candidate_prices=("n_candidate_prices", "mean"),
            selected_exposure_n=("policy_selected_exposure_n", lambda values: int(pd.Series(values).mode().iloc[0])),
        )
    )


def save_model_outputs(model: LLMMixCalModel, model_dir: Path, name: str) -> None:
    save_pickle(model, model_dir / f"{name}.pkl")
    model.alpha_table().to_csv(model_dir / f"{name}_alpha.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": name,
                "selected_exposure_n": int(model.exposure_n),
                "fit_objective": model.fit_objective,
                "fit_objective_value": float(model.objective_value),
                "status": model.status,
                "real_alpha_mass": float(model.alpha[:-1].sum()),
                "dummy_alpha_mass": float(model.alpha[-1]),
                "logit_cal_intercept": float(model.intercept),
                "logit_cal_slope": float(model.slope),
            }
        ]
    ).to_csv(model_dir / f"{name}_fit_summary.csv", index=False)


def write_aggregate_outputs(output_dir: Path, cvar_level: float) -> None:
    summaries = collect_pricing_summaries(output_dir)
    if summaries.empty:
        return
    summaries.to_csv(output_dir / "pricing_summary_all.csv", index=False)
    mean_summary = aggregate_pricing_summary(summaries)
    mean_summary.to_csv(output_dir / "pricing_summary_mean.csv", index=False)
    write_pricing_pivot(mean_summary, output_dir)
    plot_sample_efficiency(mean_summary, output_dir / "sample_efficiency_expected_revenue.pdf", "expected_revenue")
    plot_sample_efficiency(mean_summary, output_dir / f"sample_efficiency_cvar_{format_file_level(cvar_level)}.pdf", f"cvar_{format_metric_level(cvar_level)}")


def collect_pricing_summaries(output_dir: Path) -> pd.DataFrame:
    frames = []
    for split_idx in existing_split_indices(output_dir):
        split_dir = output_dir / f"split_{split_idx:03d}"
        for path in sorted(split_dir.glob("frac_*/pricing_summary.csv")):
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_pricing_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["train_frac", "objective"], as_index=False)
        .agg(
            n_splits=("split", "nunique"),
            n_train_rows_mean=("n_train_rows", "mean"),
            mean_regret=("mean_regret", "mean"),
            std_mean_regret=("mean_regret", "std"),
            se_mean_regret=("mean_regret", lambda values: float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0),
            ci95_halfwidth_mean_regret=("mean_regret", lambda values: float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0),
            median_regret=("median_regret", "mean"),
            mean_relative_performance=("mean_relative_performance", "mean"),
            price_match_rate=("price_match_rate", "mean"),
            mean_candidate_prices=("mean_candidate_prices", "mean"),
        )
    )


def write_pricing_pivot(mean_summary: pd.DataFrame, output_dir: Path) -> None:
    regret_table = mean_summary.pivot(index="train_frac", columns="objective", values="mean_regret").reset_index()
    regret_table.columns.name = None
    regret_table.to_csv(output_dir / "pricing_mean_regret_by_fraction.csv", index=False)

    ratio_table = mean_summary.pivot(
        index="train_frac",
        columns="objective",
        values="mean_relative_performance",
    ).reset_index()
    ratio_table.columns.name = None
    ratio_table.to_csv(output_dir / "pricing_mean_performance_ratio_by_fraction.csv", index=False)


def plot_sample_efficiency(summary: pd.DataFrame, path: Path, objective: str) -> None:
    plot_data = summary[summary["objective"] == objective].sort_values("train_frac")
    if plot_data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = plot_data["train_frac"].to_numpy(float)
    y = plot_data["mean_relative_performance"].to_numpy(float)
    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.set_xlabel(r"Fraction of data used for training $\hat{Q}$")
    ax.set_ylabel("Mean performance ratio")
    ax.set_xlim(0.0, 0.85)
    ax.set_ylim(0.75, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def combine_split_summary(output_dir: Path, new_records: list[dict]) -> pd.DataFrame:
    frames = []
    existing_path = output_dir / "split_summary.csv"
    if existing_path.exists():
        frames.append(pd.read_csv(existing_path))
    if new_records:
        frames.append(pd.DataFrame(new_records))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["split"], keep="last").sort_values("split").reset_index(drop=True)


def next_split_index(output_dir: Path) -> int:
    existing = existing_split_indices(output_dir)
    return max(existing) + 1 if existing else 0


def existing_split_indices(output_dir: Path) -> list[int]:
    indices = []
    for path in output_dir.glob("split_*"):
        if not path.is_dir():
            continue
        try:
            indices.append(int(path.name.removeprefix("split_")))
        except ValueError:
            continue
    return sorted(indices)


def format_fraction(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def format_metric_level(value: float) -> str:
    return f"{value:g}"


def format_file_level(value: float) -> str:
    return format_metric_level(value).replace(".", "p")


if __name__ == "__main__":
    main()
