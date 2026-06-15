from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from tqdm.auto import tqdm

from .llm_mix import (
    binomial_nll,
    choose_solver,
    fit_alpha_for_objective,
    truncated_binomial_nll,
)


@dataclass
class LLMMixCalModel:
    exposure_n: int
    alpha: np.ndarray
    persona_ids: list[str]
    intercept: float
    slope: float
    fit_objective: str
    objective_value: float
    status: str

    @property
    def name(self) -> str:
        return "llm-mix-cal"

    def purchase_probability(self, rows: pd.DataFrame) -> np.ndarray:
        q_raw = rows[self.persona_ids].to_numpy(float)
        q_cal = calibrate_probability_matrix(q_raw, self.intercept, self.slope)
        return np.clip(q_cal @ self.alpha[:-1], 1e-9, 1.0 - 1e-9)

    def mean_demand(self, rows: pd.DataFrame) -> np.ndarray:
        return self.exposure_n * self.purchase_probability(rows)

    def alpha_table(self) -> pd.DataFrame:
        return pd.DataFrame({"persona_id": self.persona_ids + ["DUMMY_NO_BUY"], "alpha": self.alpha})


def fit_llm_mix_cal(
    rows: pd.DataFrame,
    persona_ids: list[str],
    exposure_n_values: list[int],
    fit_objective: str = "truncated",
    calibration_iters: int = 6,
    solver: str | None = None,
) -> LLMMixCalModel:
    fit_rows = rows[rows["demand"] > 0].copy() if fit_objective == "truncated" else rows.copy()
    if fit_rows.empty:
        raise ValueError("No rows available for logit-calibrated fit.")
    q_raw = fit_rows[persona_ids].to_numpy(float)
    demand = fit_rows["demand"].to_numpy(int)
    solver_name = choose_solver(solver)
    best: LLMMixCalModel | None = None
    for exposure_n in tqdm(exposure_n_values, desc="fit llm-mix-cal"):
        if int(exposure_n) < int(demand.max()):
            continue
        intercept, slope = _initial_calibration(q_raw, demand, int(exposure_n))
        alpha = None
        objective = math.inf
        status = "not_started"
        for _ in range(calibration_iters):
            q_cal = calibrate_probability_matrix(q_raw, intercept, slope)
            alpha, objective, status = fit_alpha_for_objective(
                q_matrix=q_cal,
                demand=demand,
                exposure_n=int(exposure_n),
                fit_objective=fit_objective,
                solver=solver_name,
                initial_alpha=alpha,
            )
            intercept, slope, objective = _optimize_calibration(
                q_raw=q_raw,
                demand=demand,
                exposure_n=int(exposure_n),
                alpha=alpha,
                fit_objective=fit_objective,
                initial_intercept=intercept,
                initial_log_slope=float(np.log(slope)),
            )
        q_cal = calibrate_probability_matrix(q_raw, intercept, slope)
        alpha, objective, status = fit_alpha_for_objective(
            q_matrix=q_cal,
            demand=demand,
            exposure_n=int(exposure_n),
            fit_objective=fit_objective,
            solver=solver_name,
            initial_alpha=alpha,
        )
        candidate = LLMMixCalModel(
            exposure_n=int(exposure_n),
            alpha=alpha,
            persona_ids=persona_ids,
            intercept=float(intercept),
            slope=float(slope),
            fit_objective=fit_objective,
            objective_value=float(objective),
            status=status,
        )
        if best is None or candidate.objective_value < best.objective_value:
            best = candidate
    if best is None:
        raise RuntimeError("No logit-calibrated direct prompting model was fit.")
    return best


def calibrate_probability_matrix(q_matrix: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(np.asarray(q_matrix, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return np.clip(expit(intercept + slope * logits), 1e-9, 1.0 - 1e-9)


def _initial_calibration(q_raw: np.ndarray, demand: np.ndarray, exposure_n: int) -> tuple[float, float]:
    raw_mean = float(np.mean(np.clip(q_raw, 1e-6, 1.0 - 1e-6)))
    target = float(np.clip(np.mean(demand) / exposure_n, 1e-6, 1.0 - 1e-6))
    raw_logit = math.log(raw_mean / (1.0 - raw_mean))
    target_logit = math.log(target / (1.0 - target))
    return target_logit - raw_logit, 1.0


def _optimize_calibration(
    q_raw: np.ndarray,
    demand: np.ndarray,
    exposure_n: int,
    alpha: np.ndarray,
    fit_objective: str,
    initial_intercept: float,
    initial_log_slope: float,
) -> tuple[float, float, float]:
    def objective(params: np.ndarray) -> float:
        intercept = float(params[0])
        slope = float(np.exp(params[1]))
        q_cal = calibrate_probability_matrix(q_raw, intercept, slope)
        q = np.clip(q_cal @ alpha[:-1], 1e-9, 1.0 - 1e-9)
        if fit_objective == "truncated":
            return truncated_binomial_nll(demand, exposure_n, q)
        return binomial_nll(demand, exposure_n, q)

    result = minimize(
        objective,
        x0=np.array([initial_intercept, initial_log_slope], dtype=float),
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-5.0, 5.0)],
        options={"maxiter": 100},
    )
    intercept = float(result.x[0])
    slope = float(np.exp(result.x[1]))
    return intercept, slope, float(result.fun)
