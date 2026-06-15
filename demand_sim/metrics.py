from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import binom


def zero_truncated_binomial_scores(
    demand: np.ndarray,
    exposure_n: int,
    purchase_prob: np.ndarray,
    interval_levels: tuple[float, ...] = (0.9, 0.95),
    seed: int = 7,
    n_crps_samples: int = 2000,
) -> dict[str, float]:
    demand = np.asarray(demand, dtype=int)
    purchase_prob = np.clip(np.asarray(purchase_prob, dtype=float), 1e-12, 1 - 1e-12)
    if np.any(demand <= 0):
        raise ValueError("Zero-truncated metrics require strictly positive observed demand.")
    in_support = demand <= exposure_n
    log_pmf = np.full(len(demand), -np.inf, dtype=float)
    if np.any(in_support):
        supported_demand = demand[in_support]
        supported_prob = purchase_prob[in_support]
        log_choose = gammaln(exposure_n + 1) - gammaln(supported_demand + 1) - gammaln(exposure_n - supported_demand + 1)
        log_pmf[in_support] = (
            log_choose
            + supported_demand * np.log(supported_prob)
            + (exposure_n - supported_demand) * np.log1p(-supported_prob)
        )
    zero_mass = np.exp(exposure_n * np.log1p(-purchase_prob))
    avg_nll = float(-np.mean(log_pmf - np.log1p(-zero_mass)))

    rng = np.random.default_rng(seed)
    crps_values = []
    for obs, q in zip(demand, purchase_prob):
        samples = sample_zero_truncated_binomial(exposure_n, float(q), n_crps_samples, rng)
        crps_values.append(empirical_crps_from_samples(samples, float(obs)))

    support = np.arange(1, exposure_n + 1, dtype=int)
    cdf_at_d = binom.cdf(demand, exposure_n, purchase_prob)
    cdf_below_d = binom.cdf(demand - 1, exposure_n, purchase_prob)
    zt_cdf_at_d = np.clip((cdf_at_d - zero_mass) / (1 - zero_mass), 0.0, 1.0)
    zt_cdf_below_d = np.clip((cdf_below_d - zero_mass) / (1 - zero_mass), 0.0, 1.0)
    pit = zt_cdf_below_d + rng.uniform(size=len(demand)) * (zt_cdf_at_d - zt_cdf_below_d)
    sorted_pit = np.sort(pit)
    n = len(sorted_pit)
    pit_ks = float(np.max(np.maximum(np.abs(np.arange(1, n + 1) / n - sorted_pit), np.abs(sorted_pit - np.arange(0, n) / n))))

    result = {"zt_avg_nll": avg_nll, "zt_avg_crps": float(np.mean(crps_values)), "zt_pit_ks": pit_ks}
    for level in interval_levels:
        alpha = 1.0 - level
        lower = zt_binomial_ppf(alpha / 2.0, exposure_n, purchase_prob)
        upper = zt_binomial_ppf(1.0 - alpha / 2.0, exposure_n, purchase_prob)
        width = upper - lower
        score = width + (2 / alpha) * (lower - demand) * (demand < lower) + (2 / alpha) * (demand - upper) * (demand > upper)
        result[f"zt_interval_score_{level:g}"] = float(np.mean(score))
    return result


def pair_level_zero_truncated_binomial_scores(
    rows: pd.DataFrame,
    model_name: str,
    exposure_n: int,
    purchase_prob: np.ndarray,
    mean_prediction: np.ndarray | None = None,
    split_label: str = "evaluation",
    seed: int = 7,
) -> pd.DataFrame:
    working = rows[["article_id", "offer_price", "demand"]].copy().reset_index(drop=True)
    working["purchase_prob"] = np.asarray(purchase_prob, dtype=float)
    if mean_prediction is None:
        mean_prediction = exposure_n * working["purchase_prob"].to_numpy(float)
    working["mean_prediction"] = np.asarray(mean_prediction, dtype=float)
    records: list[dict] = []
    for idx, ((article_id, offer_price), group) in enumerate(working.groupby(["article_id", "offer_price"], sort=True)):
        scores = zero_truncated_binomial_scores(
            group["demand"].to_numpy(int),
            exposure_n,
            group["purchase_prob"].to_numpy(float),
            seed=seed + idx,
        )
        records.append(
            {
                "split": split_label,
                "model": model_name,
                "article_id": int(article_id),
                "offer_price": float(offer_price),
                "n_observations": int(len(group)),
                **point_prediction_scores(group["demand"].to_numpy(float), group["mean_prediction"].to_numpy(float)),
                **scores,
            }
        )
    return pd.DataFrame(records)


def summarize_pair_scores(pair_scores: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in pair_scores.columns
        if col.startswith("zt_") or col in {"mae", "rmse"}
    ]
    rows: list[dict] = []
    for (split, model), group in pair_scores.groupby(["split", "model"], sort=False):
        weights = group["n_observations"].to_numpy(float)
        for metric in metric_cols:
            rows.append({"split": split, "model": model, "metric": metric, "value": float(np.average(group[metric], weights=weights))})
    return pd.DataFrame(rows)


def pair_level_zero_truncated_pmf_scores(
    rows: pd.DataFrame,
    model_name: str,
    support: np.ndarray,
    pmf: np.ndarray,
    mean_prediction: np.ndarray | None = None,
    split_label: str = "evaluation",
    seed: int = 7,
    n_crps_samples: int = 2000,
    interval_levels: tuple[float, ...] = (0.9, 0.95),
) -> pd.DataFrame:
    """Score arbitrary discrete predictive PMFs after conditioning on positive demand."""
    support = np.asarray(support, dtype=int)
    if support[0] != 0:
        raise ValueError("support must start at 0 so the PMF can be zero-truncated.")
    working = rows[["article_id", "offer_price", "demand"]].copy().reset_index(drop=True)
    if mean_prediction is not None:
        working["mean_prediction"] = np.asarray(mean_prediction, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    positive_pmf = pmf[:, 1:] / np.clip(1.0 - pmf[:, [0]], 1e-12, None)
    positive_support = support[1:]
    records: list[dict] = []
    rng = np.random.default_rng(seed)
    for group_idx, ((article_id, offer_price), group) in enumerate(working.groupby(["article_id", "offer_price"], sort=True)):
        idx = group.index.to_numpy()
        group_pmf = positive_pmf[idx]
        demand = group["demand"].to_numpy(int)
        demand_idx = np.searchsorted(positive_support, demand)
        demand_idx = np.clip(demand_idx, 0, len(positive_support) - 1)
        row_nll = -np.log(np.clip(group_pmf[np.arange(len(group)), demand_idx], 1e-12, None))
        cdf = np.cumsum(group_pmf, axis=1)
        crps = []
        pit = []
        interval_scores: dict[float, list[float]] = {level: [] for level in interval_levels}
        for row_pmf, row_cdf, observed in zip(group_pmf, cdf, demand):
            draws = positive_support[np.searchsorted(row_cdf, rng.uniform(size=n_crps_samples), side="left").clip(0, len(positive_support) - 1)]
            crps.append(empirical_crps_from_samples(draws, float(observed)))
            obs_idx = int(np.clip(np.searchsorted(positive_support, observed), 0, len(positive_support) - 1))
            lower = row_cdf[obs_idx - 1] if obs_idx > 0 else 0.0
            upper = row_cdf[obs_idx]
            pit.append(lower + rng.uniform() * max(upper - lower, 0.0))
            for level in interval_levels:
                alpha = 1.0 - level
                interval_lower = pmf_ppf(alpha / 2.0, positive_support, row_cdf)
                interval_upper = pmf_ppf(1.0 - alpha / 2.0, positive_support, row_cdf)
                score = (
                    interval_upper
                    - interval_lower
                    + (2.0 / alpha) * (interval_lower - observed) * (observed < interval_lower)
                    + (2.0 / alpha) * (observed - interval_upper) * (observed > interval_upper)
                )
                interval_scores[level].append(float(score))
        sorted_pit = np.sort(np.asarray(pit, dtype=float))
        n = len(sorted_pit)
        pit_ks = float(np.max(np.maximum(np.abs(np.arange(1, n + 1) / n - sorted_pit), np.abs(sorted_pit - np.arange(0, n) / n))))
        scores = {
            f"zt_interval_score_{level:g}": float(np.mean(values))
            for level, values in interval_scores.items()
        }
        records.append(
            {
                "split": split_label,
                "model": model_name,
                "article_id": int(article_id),
                "offer_price": float(offer_price),
                "n_observations": int(len(group)),
                **(
                    point_prediction_scores(group["demand"].to_numpy(float), group["mean_prediction"].to_numpy(float))
                    if "mean_prediction" in group
                    else {}
                ),
                "zt_avg_nll": float(np.mean(row_nll)),
                "zt_avg_crps": float(np.mean(crps)),
                "zt_pit_ks": pit_ks,
                **scores,
            }
        )
    return pd.DataFrame(records)


def point_prediction_scores(demand: np.ndarray, mean_prediction: np.ndarray) -> dict[str, float]:
    demand = np.asarray(demand, dtype=float)
    mean_prediction = np.asarray(mean_prediction, dtype=float)
    error = demand - mean_prediction
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def zt_binomial_ppf(tau: float, exposure_n: int, purchase_prob: np.ndarray) -> np.ndarray:
    zero_mass = np.exp(exposure_n * np.log1p(-purchase_prob))
    target = zero_mass + tau * (1 - zero_mass)
    return np.maximum(1, binom.ppf(target, exposure_n, purchase_prob)).astype(float)


def pmf_ppf(tau: float, support: np.ndarray, cdf: np.ndarray) -> float:
    idx = int(np.searchsorted(cdf, tau, side="left"))
    idx = int(np.clip(idx, 0, len(support) - 1))
    return float(support[idx])


def sample_zero_truncated_binomial(exposure_n: int, purchase_prob: float, size: int, rng: np.random.Generator) -> np.ndarray:
    q = float(np.clip(purchase_prob, 1e-12, 1 - 1e-12))
    zero_mass = (1 - q) ** exposure_n
    uniforms = zero_mass + rng.uniform(size=size) * (1 - zero_mass)
    return np.maximum(1, binom.ppf(uniforms, exposure_n, q)).astype(float)


def empirical_crps_from_samples(samples: np.ndarray, observed: float) -> float:
    samples = np.asarray(samples, dtype=float).ravel()
    first = float(np.mean(np.abs(samples - observed)))
    sorted_samples = np.sort(samples)
    n = sorted_samples.size
    weights = 2 * np.arange(1, n + 1) - n - 1
    pairwise_mean_abs = float(2.0 * np.sum(weights * sorted_samples) / (n * n))
    return first - 0.5 * pairwise_mean_abs
