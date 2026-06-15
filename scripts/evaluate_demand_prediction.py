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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import (
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_MAX_DEMAND_SUPPORT,
    DEFAULT_MAX_PERSONAS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRODUCTS_DIR,
    DEFAULT_RESPONSES_DIR,
)
from demand_sim.data import (
    build_prompting_design_rows,
    load_persona_embeddings,
    load_probability_rows,
    load_product_embeddings,
    load_sales,
)
from demand_sim.evaluation import evaluate_binomial_model, evaluate_rounded_gaussian_model
from demand_sim.io import load_pickle, save_pickle, write_json
from demand_sim.metrics import zt_binomial_ppf
from demand_sim.models.emb import fit_emb
from demand_sim.models.gaussian import fit_gaussian
from demand_sim.models.llm_mix import fit_llm_mix
from demand_sim.models.llm_mix_cal import fit_llm_mix_cal


M = 7
MODEL_FILENAMES = {
    "llm-mix": "llm-mix",
    "llm-mix-cal": "llm-mix-cal",
    "emb": "emb",
    "gaussian": "gaussian",
}
MODEL_DISPLAY_NAMES = {
    "llm-mix": "LLM",
    "llm-mix-cal": "LLM-cal",
    "emb": "emb",
    "gaussian": "normal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run product-level train/test splits for distributional evaluation.")
    parser.add_argument("--sales", type=Path, default=DEFAULT_PRODUCTS_DIR / "sales_top100_online.csv")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_online_top100.csv")
    parser.add_argument("--product-embeddings", type=Path, default=DEFAULT_EMBEDDINGS_DIR / "siglip2_product_embeddings.csv")
    parser.add_argument("--persona-embeddings", type=Path, default=DEFAULT_EMBEDDINGS_DIR / "siglip2_persona_embeddings.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "demand_prediction")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llm-mix", "llm-mix-cal", "emb", "gaussian"],
        help="Models to fit. Choices: llm-mix, llm-mix-cal, emb, gaussian.",
    )
    parser.add_argument("--n-splits", type=int, default=M)
    parser.add_argument(
        "--reevaluate-existing",
        action="store_true",
        help="Reload saved split models and recompute evaluation metrics without refitting.",
    )
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--fit-objective", choices=["truncated", "naive"], default="truncated")
    parser.add_argument("--exposure-n-values", type=int, nargs="+", default=[100, 150, 200, 250])
    parser.add_argument("--max-personas", type=int, default=DEFAULT_MAX_PERSONAS)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--support-max", type=int, default=DEFAULT_MAX_DEMAND_SUPPORT)
    parser.add_argument("--logit-calibration-iters", type=int, default=6)
    parser.add_argument("--embedding-outer-iters", type=int, default=8)
    parser.add_argument("--embedding-adam-steps", type=int, default=150)
    parser.add_argument("--embedding-l2", type=float, default=1.0)
    parser.add_argument("--gaussian-ridge", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_models = normalize_model_names(args.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sales = load_sales(args.sales)
    probabilities = load_probability_rows(args.responses)
    prompting_rows, persona_ids = build_prompting_design_rows(sales, probabilities)
    product_embeddings, product_cols = load_product_embeddings(args.product_embeddings)
    persona_embeddings, embedding_persona_ids, persona_cols = load_persona_embeddings(args.persona_embeddings, args.max_personas)
    embedded_rows = prompting_rows.merge(product_embeddings, on="article_id", how="inner")
    product_ids = np.array(sorted(embedded_rows["article_id"].unique()), dtype=int)

    if args.reevaluate_existing:
        reevaluate_existing_splits(
            args=args,
            requested_models=requested_models,
            prompting_rows=prompting_rows,
            embedded_rows=embedded_rows,
        )
        print(f"Recomputed existing split evaluations in {args.output_dir} without refitting models.")
        return

    split_records: list[dict] = []
    start_split = next_split_index(args.output_dir)

    for split_idx in range(start_split, start_split + args.n_splits):
        split_dir = args.output_dir / f"split_{split_idx:03d}"
        model_dir = split_dir / "models"
        eval_dir = split_dir / "evaluation"
        plot_dir = split_dir / "plots"
        model_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        train_ids, test_ids = split_products(product_ids, args.train_frac, args.seed + split_idx)
        train_prompting = prompting_rows[prompting_rows["article_id"].isin(train_ids)].copy()
        test_prompting = prompting_rows[prompting_rows["article_id"].isin(test_ids)].copy()
        train_embedded = embedded_rows[embedded_rows["article_id"].isin(train_ids)].copy()
        test_embedded = embedded_rows[embedded_rows["article_id"].isin(test_ids)].copy()

        split_records.append(
            {
                "split": split_idx,
                "n_train_products": int(len(train_ids)),
                "n_test_products": int(len(test_ids)),
                "n_train_rows": int(len(train_prompting)),
                "n_test_rows": int(len(test_prompting)),
            }
        )
        pd.Series(train_ids, name="article_id").to_csv(split_dir / "train_products.csv", index=False)
        pd.Series(test_ids, name="article_id").to_csv(split_dir / "test_products.csv", index=False)

        models = fit_split_models(
            requested_models=requested_models,
            train_prompting=train_prompting,
            train_embedded=train_embedded,
            product_cols=product_cols,
            persona_embeddings=persona_embeddings,
            embedding_persona_ids=embedding_persona_ids,
            persona_cols=persona_cols,
            persona_ids=persona_ids,
            exposure_n_values=args.exposure_n_values,
            fit_objective=args.fit_objective,
            solver=args.solver,
            logit_calibration_iters=args.logit_calibration_iters,
            embedding_outer_iters=args.embedding_outer_iters,
            embedding_adam_steps=args.embedding_adam_steps,
            embedding_l2=args.embedding_l2,
            gaussian_ridge=args.gaussian_ridge,
        )
        for model_name, model in models.items():
            save_pickle(model, model_dir / f"{MODEL_FILENAMES[model_name]}.pkl")
            if hasattr(model, "alpha_table"):
                model.alpha_table().to_csv(model_dir / f"{MODEL_FILENAMES[model_name]}_alpha.csv", index=False)
        fit_summary = summarize_fit_models(models, split_idx)
        fit_summary.to_csv(split_dir / "model_fit_summary.csv", index=False)
        split_calibration_frames: list[pd.DataFrame] = []

        for sample_name, prompt_rows, emb_rows in [
            ("train", train_prompting, train_embedded),
            ("test", test_prompting, test_embedded),
        ]:
            sample_eval_dir = eval_dir / sample_name
            for model_name, model in models.items():
                rows = emb_rows if model_name in {"emb", "gaussian"} else prompt_rows
                if model_name == "gaussian":
                    pair_scores, summary = evaluate_rounded_gaussian_model(
                        rows=rows,
                        model=model,
                        output_dir=sample_eval_dir,
                        support_max=args.support_max,
                        split_label=f"split_{split_idx:03d}_{sample_name}",
                    )
                    calibration = gaussian_quantile_calibration(
                        rows=rows,
                        model=model,
                        split_idx=split_idx,
                        sample=sample_name,
                        support_max=args.support_max,
                    )
                else:
                    pair_scores, summary = evaluate_binomial_model(
                        rows=rows,
                        model=model,
                        output_dir=sample_eval_dir,
                        split_label=f"split_{split_idx:03d}_{sample_name}",
                        seed=args.seed + 1000 * split_idx,
                    )
                    calibration = binomial_quantile_calibration(
                        rows=rows,
                        model=model,
                        split_idx=split_idx,
                        sample=sample_name,
                    )
                pair_scores.insert(0, "sample", sample_name)
                summary.insert(0, "sample", sample_name)
                split_calibration_frames.append(calibration)
        split_calibration = pd.concat(split_calibration_frames, ignore_index=True)
        split_calibration.to_csv(eval_dir / "quantile_calibration.csv", index=False)
        plot_quantile_calibration(split_calibration, plot_dir / "quantile_calibration_train.pdf", sample="train")
        plot_quantile_calibration(split_calibration, plot_dir / "quantile_calibration_test.pdf", sample="test")

    pair_scores_all, summary_all, calibration_all, fit_summary_all = collect_split_outputs(args.output_dir)
    summary_mean = aggregate_summary(summary_all)
    calibration_mean = aggregate_calibration(calibration_all)
    pair_scores_all.to_csv(args.output_dir / "pair_scores_all_splits.csv", index=False)
    summary_all.to_csv(args.output_dir / "summary_all_splits.csv", index=False)
    calibration_all.to_csv(args.output_dir / "quantile_calibration_all_splits.csv", index=False)
    fit_summary_all.to_csv(args.output_dir / "model_fit_summary_all_splits.csv", index=False)
    summary_mean.to_csv(args.output_dir / "summary_mean_by_model.csv", index=False)
    calibration_mean.to_csv(args.output_dir / "quantile_calibration_mean_by_model.csv", index=False)
    write_metric_pivot_tables(summary_mean, args.output_dir)
    split_summary = combine_split_summary(args.output_dir, split_records)
    split_summary.to_csv(args.output_dir / "split_summary.csv", index=False)
    plot_quantile_calibration(calibration_mean, args.output_dir / "quantile_calibration_mean_by_model_train.pdf", sample="train")
    plot_quantile_calibration(calibration_mean, args.output_dir / "quantile_calibration_mean_by_model_test.pdf", sample="test")
    write_json(
        {
            "sales": str(args.sales),
            "responses": str(args.responses),
            "product_embeddings": str(args.product_embeddings),
            "persona_embeddings": str(args.persona_embeddings),
            "n_splits": args.n_splits,
            "total_completed_splits": int(len(split_summary)),
            "new_split_start": int(start_split),
            "new_split_end": int(start_split + args.n_splits - 1) if args.n_splits > 0 else int(start_split - 1),
            "train_frac": args.train_frac,
            "seed": args.seed,
            "fit_objective": args.fit_objective,
            "exposure_n_values": args.exposure_n_values,
            "support_max": args.support_max,
            "models": requested_models,
        },
        args.output_dir / "metadata.json",
    )
    print(f"Wrote product-split evaluation outputs to {args.output_dir}. Total completed splits: {len(split_summary)}.")


def fit_split_models(
    requested_models: list[str],
    train_prompting: pd.DataFrame,
    train_embedded: pd.DataFrame,
    product_cols: list[str],
    persona_embeddings: pd.DataFrame,
    embedding_persona_ids: list[str],
    persona_cols: list[str],
    persona_ids: list[str],
    exposure_n_values: list[int],
    fit_objective: str,
    solver: str | None,
    logit_calibration_iters: int,
    embedding_outer_iters: int,
    embedding_adam_steps: int,
    embedding_l2: float,
    gaussian_ridge: float,
) -> dict[str, object]:
    models: dict[str, object] = {}
    if "llm-mix" in requested_models:
        models["llm-mix"] = fit_llm_mix(train_prompting, persona_ids, exposure_n_values, fit_objective, solver)
    if "llm-mix-cal" in requested_models:
        models["llm-mix-cal"] = fit_llm_mix_cal(
            train_prompting,
            persona_ids,
            exposure_n_values,
            fit_objective=fit_objective,
            calibration_iters=logit_calibration_iters,
            solver=solver,
        )
    if "emb" in requested_models:
        models["emb"] = fit_emb(
            train_embedded,
            product_cols,
            persona_embeddings,
            embedding_persona_ids,
            persona_cols,
            exposure_n_values,
            fit_objective=fit_objective,
            outer_iters=embedding_outer_iters,
            adam_steps=embedding_adam_steps,
            l2=embedding_l2,
            solver=solver,
        )
    if "gaussian" in requested_models:
        models["gaussian"] = fit_gaussian(
            train_embedded,
            product_cols,
            weight_decay=gaussian_ridge,
        )
    return models


def reevaluate_existing_splits(
    args: argparse.Namespace,
    requested_models: list[str],
    prompting_rows: pd.DataFrame,
    embedded_rows: pd.DataFrame,
) -> None:
    split_indices = existing_split_indices(args.output_dir)
    if not split_indices:
        raise RuntimeError(f"No existing split directories found under {args.output_dir}.")

    for split_idx in split_indices:
        split_dir = args.output_dir / f"split_{split_idx:03d}"
        model_dir = split_dir / "models"
        eval_dir = split_dir / "evaluation"
        plot_dir = split_dir / "plots"
        eval_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        train_ids = pd.read_csv(split_dir / "train_products.csv")["article_id"].astype(int).to_numpy()
        test_ids = pd.read_csv(split_dir / "test_products.csv")["article_id"].astype(int).to_numpy()
        train_prompting = prompting_rows[prompting_rows["article_id"].isin(train_ids)].copy()
        test_prompting = prompting_rows[prompting_rows["article_id"].isin(test_ids)].copy()
        train_embedded = embedded_rows[embedded_rows["article_id"].isin(train_ids)].copy()
        test_embedded = embedded_rows[embedded_rows["article_id"].isin(test_ids)].copy()

        models = {}
        for model_name in requested_models:
            model_path = model_dir / f"{MODEL_FILENAMES[model_name]}.pkl"
            if model_path.exists():
                models[model_name] = load_pickle(model_path)

        if not models:
            continue

        split_calibration_frames: list[pd.DataFrame] = []
        for sample_name, prompt_rows, emb_rows in [
            ("train", train_prompting, train_embedded),
            ("test", test_prompting, test_embedded),
        ]:
            sample_eval_dir = eval_dir / sample_name
            for model_name, model in models.items():
                rows = emb_rows if model_name in {"emb", "gaussian"} else prompt_rows
                if model_name == "gaussian":
                    pair_scores, summary = evaluate_rounded_gaussian_model(
                        rows=rows,
                        model=model,
                        output_dir=sample_eval_dir,
                        support_max=args.support_max,
                        split_label=f"split_{split_idx:03d}_{sample_name}",
                    )
                    calibration = gaussian_quantile_calibration(
                        rows=rows,
                        model=model,
                        split_idx=split_idx,
                        sample=sample_name,
                        support_max=args.support_max,
                    )
                else:
                    pair_scores, summary = evaluate_binomial_model(
                        rows=rows,
                        model=model,
                        output_dir=sample_eval_dir,
                        split_label=f"split_{split_idx:03d}_{sample_name}",
                        seed=args.seed + 1000 * split_idx,
                    )
                    calibration = binomial_quantile_calibration(
                        rows=rows,
                        model=model,
                        split_idx=split_idx,
                        sample=sample_name,
                    )
                pair_scores.insert(0, "sample", sample_name)
                summary.insert(0, "sample", sample_name)
                split_calibration_frames.append(calibration)

        if split_calibration_frames:
            split_calibration = pd.concat(split_calibration_frames, ignore_index=True)
            split_calibration.to_csv(eval_dir / "quantile_calibration.csv", index=False)
            plot_quantile_calibration(split_calibration, plot_dir / "quantile_calibration_train.pdf", sample="train")
            plot_quantile_calibration(split_calibration, plot_dir / "quantile_calibration_test.pdf", sample="test")

    pair_scores_all, summary_all, calibration_all, fit_summary_all = collect_split_outputs(args.output_dir)
    summary_mean = aggregate_summary(summary_all)
    calibration_mean = aggregate_calibration(calibration_all)
    pair_scores_all.to_csv(args.output_dir / "pair_scores_all_splits.csv", index=False)
    summary_all.to_csv(args.output_dir / "summary_all_splits.csv", index=False)
    calibration_all.to_csv(args.output_dir / "quantile_calibration_all_splits.csv", index=False)
    fit_summary_all.to_csv(args.output_dir / "model_fit_summary_all_splits.csv", index=False)
    summary_mean.to_csv(args.output_dir / "summary_mean_by_model.csv", index=False)
    calibration_mean.to_csv(args.output_dir / "quantile_calibration_mean_by_model.csv", index=False)
    write_metric_pivot_tables(summary_mean, args.output_dir)
    plot_quantile_calibration(calibration_mean, args.output_dir / "quantile_calibration_mean_by_model_train.pdf", sample="train")
    plot_quantile_calibration(calibration_mean, args.output_dir / "quantile_calibration_mean_by_model_test.pdf", sample="test")


def summarize_fit_models(models: dict[str, object], split_idx: int) -> pd.DataFrame:
    records = []
    for model_name, model in models.items():
        record = {"split": split_idx, "model": model_name}
        if hasattr(model, "exposure_n"):
            record["selected_exposure_n"] = int(model.exposure_n)
        else:
            record["selected_exposure_n"] = np.nan
        if hasattr(model, "objective_value"):
            record["fit_objective_value"] = float(model.objective_value)
        if hasattr(model, "fit_objective"):
            record["fit_objective"] = model.fit_objective
        if hasattr(model, "alpha"):
            record["real_alpha_mass"] = float(model.alpha[:-1].sum())
            record["dummy_alpha_mass"] = float(model.alpha[-1])
        if hasattr(model, "intercept"):
            record["logit_cal_intercept"] = float(model.intercept)
            record["logit_cal_slope"] = float(model.slope)
        if hasattr(model, "theta"):
            record["price_coefficient_scaled"] = float(model.theta[-1])
        if hasattr(model, "train_loss"):
            record["train_loss"] = float(model.train_loss)
            record["sigma"] = float(model.sigma)
            record["price_coefficient_scaled"] = float(model.beta[1])
        records.append(record)
    return pd.DataFrame(records)


def split_products(product_ids: np.ndarray, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    shuffled = np.array(product_ids, copy=True)
    rng.shuffle(shuffled)
    n_train = int(round(train_frac * len(shuffled)))
    n_train = min(max(n_train, 1), len(shuffled) - 1)
    train_ids = np.sort(shuffled[:n_train])
    test_ids = np.sort(shuffled[n_train:])
    return train_ids, test_ids


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


def collect_split_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    fit_summary_frames: list[pd.DataFrame] = []

    for split_idx in existing_split_indices(output_dir):
        split_dir = output_dir / f"split_{split_idx:03d}"
        for sample in ["train", "test"]:
            sample_dir = split_dir / "evaluation" / sample
            for pair_path in sorted(sample_dir.glob("*_pair_scores.csv")):
                pair = pd.read_csv(pair_path)
                if "sample" not in pair.columns:
                    pair.insert(0, "sample", sample)
                pair_frames.append(pair)
            for summary_path in sorted(sample_dir.glob("*_summary.csv")):
                summary = pd.read_csv(summary_path)
                if "sample" not in summary.columns:
                    summary.insert(0, "sample", sample)
                summary_frames.append(summary)

        calibration_path = split_dir / "evaluation" / "quantile_calibration.csv"
        if calibration_path.exists():
            calibration_frames.append(pd.read_csv(calibration_path))

        fit_summary_path = split_dir / "model_fit_summary.csv"
        if fit_summary_path.exists():
            fit_summary_frames.append(pd.read_csv(fit_summary_path))

    if not pair_frames or not summary_frames or not calibration_frames:
        raise RuntimeError(f"No completed split outputs found under {output_dir}.")

    pair_scores = pd.concat(pair_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    fit_summaries = pd.concat(fit_summary_frames, ignore_index=True) if fit_summary_frames else pd.DataFrame()
    return pair_scores, summaries, calibration, fit_summaries


def combine_split_summary(output_dir: Path, new_records: list[dict]) -> pd.DataFrame:
    frames = []
    existing_path = output_dir / "split_summary.csv"
    if existing_path.exists():
        frames.append(pd.read_csv(existing_path))
    if new_records:
        frames.append(pd.DataFrame(new_records))
    if not frames:
        return pd.DataFrame(columns=["split", "n_train_products", "n_test_products", "n_train_rows", "n_test_rows"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["split"], keep="last").sort_values("split").reset_index(drop=True)


def binomial_quantile_calibration(
    rows: pd.DataFrame,
    model,
    split_idx: int,
    sample: str,
    quantiles: np.ndarray | None = None,
) -> pd.DataFrame:
    quantiles = default_quantiles() if quantiles is None else quantiles
    eval_rows = rows[rows["demand"] > 0].copy()
    demand = eval_rows["demand"].to_numpy(int)
    purchase_prob = model.purchase_probability(eval_rows)
    records = []
    for quantile in quantiles:
        predicted = zt_binomial_ppf(float(quantile), model.exposure_n, purchase_prob)
        records.append(
            {
                "split": split_idx,
                "sample": sample,
                "model": model.name,
                "quantile": float(quantile),
                "empirical_coverage": float(np.mean(demand <= predicted)),
                "n_observations": int(len(demand)),
            }
        )
    return pd.DataFrame(records)


def gaussian_quantile_calibration(
    rows: pd.DataFrame,
    model,
    split_idx: int,
    sample: str,
    support_max: int,
    quantiles: np.ndarray | None = None,
) -> pd.DataFrame:
    quantiles = default_quantiles() if quantiles is None else quantiles
    eval_rows = rows[rows["demand"] > 0].copy().reset_index(drop=True)
    demand = eval_rows["demand"].to_numpy(int)
    support = np.arange(0, support_max + 1, dtype=int)
    pmf = model.pmf(eval_rows, support)
    positive_support = support[1:]
    positive_pmf = pmf[:, 1:] / np.clip(1.0 - pmf[:, [0]], 1e-12, None)
    cdf = np.cumsum(positive_pmf, axis=1)
    records = []
    for quantile in quantiles:
        idx = np.sum(cdf < float(quantile), axis=1)
        idx = np.clip(idx, 0, len(positive_support) - 1)
        predicted = positive_support[idx]
        records.append(
            {
                "split": split_idx,
                "sample": sample,
                "model": model.name,
                "quantile": float(quantile),
                "empirical_coverage": float(np.mean(demand <= predicted)),
                "n_observations": int(len(demand)),
            }
        )
    return pd.DataFrame(records)


def default_quantiles() -> np.ndarray:
    return np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=float)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["sample", "model", "metric"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"))
    )


def aggregate_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    return (
        calibration.groupby(["sample", "model", "quantile"], as_index=False)
        .agg(empirical_coverage=("empirical_coverage", "mean"), coverage_std=("empirical_coverage", "std"), n_observations=("n_observations", "sum"))
    )


def write_metric_pivot_tables(summary_mean: pd.DataFrame, output_dir: Path) -> None:
    for sample in ["train", "test"]:
        table = (
            summary_mean[summary_mean["sample"] == sample]
            .pivot(index="model", columns="metric", values="mean")
            .reset_index()
        )
        table.to_csv(output_dir / f"summary_mean_metrics_{sample}.csv", index=False)


def plot_quantile_calibration(calibration: pd.DataFrame, path: Path, sample: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = calibration[calibration["sample"] == sample].copy()
    if plot_data.empty:
        return
    if "coverage_std" not in plot_data.columns:
        plot_data = aggregate_calibration(plot_data)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for model_name, group in plot_data.groupby("model", sort=False):
        group = group.sort_values("quantile")
        ax.plot(
            group["quantile"],
            group["empirical_coverage"],
            marker="o",
            linewidth=1.8,
            label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
        )
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def normalize_model_names(model_names: list[str]) -> list[str]:
    normalized = []
    for model_name in model_names:
        if model_name not in MODEL_FILENAMES:
            raise SystemExit(f"Unknown model '{model_name}'. Use one of: {', '.join(MODEL_FILENAMES)}.")
        if model_name not in normalized:
            normalized.append(model_name)
    return normalized


if __name__ == "__main__":
    main()
