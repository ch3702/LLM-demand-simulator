from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import pair_level_zero_truncated_binomial_scores, pair_level_zero_truncated_pmf_scores, summarize_pair_scores
from .models.gaussian import GaussianModel


def evaluate_binomial_model(
    rows: pd.DataFrame,
    model,
    output_dir: Path,
    split_label: str = "evaluation",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a binomial-purchase-probability model on positive-demand rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = rows[rows["demand"] > 0].copy()
    purchase_prob = model.purchase_probability(eval_rows)
    zero_mass = np.exp(model.exposure_n * np.log1p(-purchase_prob))
    mean_prediction = model.mean_demand(eval_rows) / np.clip(1.0 - zero_mass, 1e-12, None)
    pair_scores = pair_level_zero_truncated_binomial_scores(
        rows=eval_rows,
        model_name=model.name,
        exposure_n=model.exposure_n,
        purchase_prob=purchase_prob,
        mean_prediction=mean_prediction,
        split_label=split_label,
        seed=seed,
    )
    summary = summarize_pair_scores(pair_scores)
    pair_scores.to_csv(output_dir / f"{model.name}_pair_scores.csv", index=False)
    summary.to_csv(output_dir / f"{model.name}_summary.csv", index=False)
    return pair_scores, summary


def evaluate_rounded_gaussian_model(
    rows: pd.DataFrame,
    model: GaussianModel,
    output_dir: Path,
    support_max: int = 400,
    split_label: str = "evaluation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate rounded Gaussian after conditioning its PMF on positive demand."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = rows[rows["demand"] > 0].copy().reset_index(drop=True)
    support = np.arange(0, support_max + 1, dtype=int)
    pmf = model.pmf(eval_rows, support)
    mean_prediction = model.mean_demand(eval_rows)
    pair_scores = pair_level_zero_truncated_pmf_scores(
        rows=eval_rows,
        model_name=model.name,
        support=support,
        pmf=pmf,
        mean_prediction=mean_prediction,
        split_label=split_label,
    )
    summary = summarize_pair_scores(pair_scores)
    pair_scores.to_csv(output_dir / f"{model.name}_pair_scores.csv", index=False)
    summary.to_csv(output_dir / f"{model.name}_summary.csv", index=False)
    return pair_scores, summary
