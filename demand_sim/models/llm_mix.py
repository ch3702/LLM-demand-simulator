from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import gammaln
from tqdm.auto import tqdm


@dataclass
class LLMMixModel:
    exposure_n: int
    alpha: np.ndarray
    persona_ids: list[str]
    fit_objective: str
    objective_value: float
    status: str

    @property
    def name(self) -> str:
        return "llm-mix"

    def purchase_probability(self, rows: pd.DataFrame) -> np.ndarray:
        q_matrix = rows[self.persona_ids].to_numpy(float)
        return self.purchase_probability_from_matrix(q_matrix)

    def purchase_probability_from_matrix(self, q_matrix: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q_matrix, dtype=float) @ self.alpha[:-1], 1e-9, 1.0 - 1e-9)

    def mean_demand(self, rows: pd.DataFrame) -> np.ndarray:
        return self.exposure_n * self.purchase_probability(rows)

    def alpha_table(self) -> pd.DataFrame:
        return pd.DataFrame({"persona_id": self.persona_ids + ["DUMMY_NO_BUY"], "alpha": self.alpha})


def fit_llm_mix(
    rows: pd.DataFrame,
    persona_ids: list[str],
    exposure_n_values: list[int],
    fit_objective: str = "truncated",
    solver: str | None = None,
) -> LLMMixModel:
    fit_rows = _fit_rows(rows, fit_objective)
    q_matrix = fit_rows[persona_ids].to_numpy(float)
    demand = fit_rows["demand"].to_numpy(int)
    solver_name = choose_solver(solver)
    best: LLMMixModel | None = None
    last_alpha: np.ndarray | None = None
    for exposure_n in tqdm(exposure_n_values, desc="fit llm-mix"):
        if int(exposure_n) < int(demand.max()):
            continue
        alpha, objective, status = fit_alpha_for_objective(
            q_matrix=q_matrix,
            demand=demand,
            exposure_n=int(exposure_n),
            fit_objective=fit_objective,
            solver=solver_name,
            initial_alpha=last_alpha,
        )
        last_alpha = alpha
        candidate = LLMMixModel(
            exposure_n=int(exposure_n),
            alpha=alpha,
            persona_ids=persona_ids,
            fit_objective=fit_objective,
            objective_value=float(objective),
            status=status,
        )
        if best is None or candidate.objective_value < best.objective_value:
            best = candidate
    if best is None:
        raise RuntimeError("No llm-mix model was fit.")
    return best


def choose_solver(preferred: str | None = None) -> str:
    installed = cp.installed_solvers()
    if preferred and preferred in installed:
        return preferred
    for solver in ["ECOS", "CLARABEL", "SCS"]:
        if solver in installed:
            return solver
    raise RuntimeError("No supported CVXPY solver is installed.")


def fit_alpha_for_objective(
    q_matrix: np.ndarray,
    demand: np.ndarray,
    exposure_n: int,
    fit_objective: str,
    solver: str,
    initial_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str]:
    if fit_objective == "naive":
        return fit_alpha_for_n(q_matrix, demand, exposure_n, solver)
    if fit_objective == "truncated":
        return fit_truncated_alpha_for_n(q_matrix, demand, exposure_n, initial_alpha=initial_alpha)
    raise ValueError(f"Unknown fit objective: {fit_objective}")


def fit_alpha_for_n(q_matrix: np.ndarray, demand: np.ndarray, exposure_n: int, solver: str) -> tuple[np.ndarray, float, str]:
    n_personas = q_matrix.shape[1]
    alpha = cp.Variable(n_personas + 1, nonneg=True)
    q = q_matrix @ alpha[:-1]
    objective = -cp.sum(cp.multiply(demand, cp.log(q)) + cp.multiply(exposure_n - demand, cp.log(1 - q)))
    problem = cp.Problem(cp.Minimize(objective), [cp.sum(alpha) == 1, q >= 1e-12, q <= 1 - 1e-12])
    problem.solve(solver=solver, verbose=False)
    if alpha.value is None:
        raise RuntimeError(f"CVXPY failed for N={exposure_n} with status={problem.status}")
    fitted = np.maximum(np.asarray(alpha.value, dtype=float).ravel(), 0.0)
    fitted = fitted / fitted.sum()
    q_fit = np.clip(q_matrix @ fitted[:-1], 1e-12, 1 - 1e-12)
    return fitted, binomial_nll(demand, exposure_n, q_fit), str(problem.status)


def fit_truncated_alpha_for_n(
    q_matrix: np.ndarray,
    demand: np.ndarray,
    exposure_n: int,
    initial_alpha: np.ndarray | None = None,
    q_upper: float = 0.5,
) -> tuple[np.ndarray, float, str]:
    if np.any(demand <= 0):
        raise ValueError("Truncated fitting requires strictly positive observed demand.")
    n_personas = q_matrix.shape[1]
    alpha0 = _normalize_alpha(initial_alpha, q_matrix, q_upper) if initial_alpha is not None else _initial_alpha(q_matrix, q_upper)

    q_constraint = LinearConstraint(
        np.hstack([q_matrix, np.zeros((q_matrix.shape[0], 1))]),
        lb=np.zeros(q_matrix.shape[0], dtype=float),
        ub=np.full(q_matrix.shape[0], q_upper, dtype=float),
    )
    simplex = LinearConstraint(
        np.ones((1, n_personas + 1), dtype=float),
        lb=np.array([1.0], dtype=float),
        ub=np.array([1.0], dtype=float),
    )
    bounds = Bounds(np.zeros(n_personas + 1, dtype=float), np.ones(n_personas + 1, dtype=float))
    log_choose = gammaln(exposure_n + 1) - gammaln(demand + 1) - gammaln(exposure_n - demand + 1)

    def objective(alpha: np.ndarray) -> tuple[float, np.ndarray]:
        q = np.clip(q_matrix @ alpha[:-1], 1e-12, q_upper)
        log_one_minus_q = np.log1p(-q)
        log_zero = exposure_n * log_one_minus_q
        trunc_norm = log1mexp(log_zero)
        value = float(-np.sum(log_choose + demand * np.log(q) + (exposure_n - demand) * log_one_minus_q - trunc_norm))

        trunc_grad = exposure_n * np.exp((exposure_n - 1) * log_one_minus_q - trunc_norm)
        grad_q = -demand / q + (exposure_n - demand) / (1.0 - q) + trunc_grad
        grad = np.zeros(n_personas + 1, dtype=float)
        grad[:-1] = q_matrix.T @ grad_q
        return value, grad

    initial_guesses = [alpha0]
    if initial_alpha is None:
        base_weights = np.full(n_personas, 1.0 / n_personas, dtype=float)
        base_q = q_matrix @ base_weights
        mean_base_q = float(np.mean(base_q))
        max_base_q = float(np.max(base_q))
        target_q = float(np.mean(demand) / exposure_n)
        if mean_base_q > 0 and max_base_q > 0:
            target_mass = target_q / mean_base_q
            max_feasible_mass = q_upper / max_base_q
            for mass in [target_mass / 2, target_mass, target_mass * 2, 0.01, 0.02, 0.05, 0.1]:
                mass = float(np.clip(mass, 1e-6, min(1.0, max_feasible_mass)))
                guess = np.zeros(n_personas + 1, dtype=float)
                guess[:-1] = mass * base_weights
                guess[-1] = 1.0 - mass
                initial_guesses.append(guess)

    best_alpha = None
    best_objective = math.inf
    best_status = ""
    best_success = False
    failure_messages = []
    for guess in initial_guesses:
        result = minimize(
            fun=lambda alpha: objective(alpha)[0],
            x0=guess,
            jac=lambda alpha: objective(alpha)[1],
            method="SLSQP",
            bounds=bounds,
            constraints=[q_constraint, simplex],
            options={"disp": False, "ftol": 1e-9, "maxiter": 500},
        )
        for raw, status, success in [(guess, "initial guess", True), (result.x, result.message, result.success)]:
            alpha = _normalize_alpha(raw, q_matrix, q_upper)
            q = q_matrix @ alpha[:-1]
            value = truncated_binomial_nll(demand, exposure_n, q)
            feasible = (
                np.isfinite(value)
                and np.isclose(alpha.sum(), 1.0, atol=1e-6)
                and np.all(alpha >= -1e-8)
                and np.all(q <= q_upper + 1e-6)
            )
            if feasible and value < best_objective:
                best_alpha = alpha
                best_objective = value
                best_status = str(status)
                best_success = bool(success)
        if not result.success:
            failure_messages.append(f"status={result.status}, message={result.message}")

    for alpha, value, status, success in _fit_truncated_alpha_with_penalty_starts(
        q_matrix=q_matrix,
        demand=demand,
        exposure_n=exposure_n,
        q_upper=q_upper,
        starts=initial_guesses,
    ):
        if value < best_objective:
            best_alpha = alpha
            best_objective = value
            best_status = status
            best_success = success

    if best_alpha is None:
        joined = "; ".join(failure_messages) if failure_messages else "no feasible candidate"
        raise RuntimeError(f"SciPy failed for truncated N={exposure_n}: {joined}")
    status_prefix = "" if best_success else "approximate: "
    return best_alpha, best_objective, f"{status_prefix}{best_status}"


def binomial_nll(demand: np.ndarray, exposure_n: int, q: np.ndarray) -> float:
    q = np.clip(q, 1e-12, 1 - 1e-12)
    if np.any(demand > exposure_n):
        return math.inf
    log_choose = gammaln(exposure_n + 1) - gammaln(demand + 1) - gammaln(exposure_n - demand + 1)
    return float(-np.sum(log_choose + demand * np.log(q) + (exposure_n - demand) * np.log1p(-q)))


def truncated_binomial_nll(demand: np.ndarray, exposure_n: int, q: np.ndarray) -> float:
    q = np.clip(q, 1e-12, 0.5)
    if np.any(demand <= 0) or np.any(demand > exposure_n):
        return math.inf
    log_choose = gammaln(exposure_n + 1) - gammaln(demand + 1) - gammaln(exposure_n - demand + 1)
    log_zero = exposure_n * np.log1p(-q)
    return float(-np.sum(log_choose + demand * np.log(q) + (exposure_n - demand) * np.log1p(-q) - log1mexp(log_zero)))


def truncated_binomial_nll_gradient(
    demand: np.ndarray,
    exposure_n: int,
    q: np.ndarray,
    q_upper: float = 0.5,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    clipped = np.clip(q, 1e-12, q_upper)
    one_minus_q = 1.0 - clipped
    zero_mass = one_minus_q**exposure_n
    positive_mass = np.clip(1.0 - zero_mass, 1e-300, None)
    grad = (
        -demand / clipped
        + (exposure_n - demand) / one_minus_q
        + exposure_n * (one_minus_q ** (exposure_n - 1)) / positive_mass
    )
    grad = np.where((q <= 1e-12) | (q >= q_upper), 0.0, grad)
    excess = np.maximum(q - q_upper, 0.0)
    if np.any(excess > 0):
        grad = grad + 2.0e8 * excess / max(len(q), 1)
    return grad


def _fit_truncated_alpha_with_penalty_starts(
    q_matrix: np.ndarray,
    demand: np.ndarray,
    exposure_n: int,
    q_upper: float,
    starts: list[np.ndarray],
    n_random_starts: int = 10,
) -> list[tuple[np.ndarray, float, str, bool]]:
    n_personas = q_matrix.shape[1]
    rng = np.random.default_rng(17)
    all_starts = list(starts)
    for _ in range(n_random_starts):
        all_starts.append(_normalize_alpha(rng.dirichlet(np.ones(n_personas + 1)), q_matrix, q_upper))

    sum_penalty = 1.0e7
    q_penalty = 1.0e7

    def objective(alpha: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = np.asarray(alpha, dtype=float)
        q = q_matrix @ alpha[:-1]
        value = truncated_binomial_nll(demand, exposure_n, q)
        grad = np.zeros_like(alpha)
        grad[:-1] = q_matrix.T @ truncated_binomial_nll_gradient(demand, exposure_n, q, q_upper=q_upper)

        sum_error = float(alpha.sum() - 1.0)
        value += sum_penalty * sum_error * sum_error
        grad += 2.0 * sum_penalty * sum_error

        excess_q = np.maximum(q - q_upper, 0.0)
        if np.any(excess_q > 0):
            value += q_penalty * float(np.mean(excess_q**2))
            grad[:-1] += (2.0 * q_penalty / max(len(q), 1)) * (q_matrix.T @ excess_q)
        return value, grad

    candidates: list[tuple[np.ndarray, float, str, bool]] = []
    bounds = [(0.0, 1.0)] * (n_personas + 1)
    for start in all_starts:
        result = minimize(
            fun=lambda alpha: objective(alpha),
            x0=start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        for raw_alpha, status, success in [(start, "penalty initial guess", True), (result.x, result.message, result.success)]:
            alpha = _normalize_alpha(raw_alpha, q_matrix, q_upper)
            q = q_matrix @ alpha[:-1]
            value = truncated_binomial_nll(demand, exposure_n, q)
            feasible = (
                np.isfinite(value)
                and np.isclose(alpha.sum(), 1.0, atol=1e-6)
                and np.all(alpha >= -1e-8)
                and np.all(q <= q_upper + 1e-6)
            )
            if feasible:
                candidates.append((alpha, value, str(status), bool(success)))
    return candidates


def log1mexp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    cutoff = -math.log(2.0)
    mask = x <= cutoff
    out[mask] = np.log1p(-np.exp(x[mask]))
    out[~mask] = np.log(-np.expm1(x[~mask]))
    return out


def _fit_rows(rows: pd.DataFrame, fit_objective: str) -> pd.DataFrame:
    if fit_objective == "truncated":
        out = rows[rows["demand"] > 0].copy()
        if out.empty:
            raise ValueError("Truncated fitting requires at least one positive-demand row.")
        return out
    if fit_objective == "naive":
        return rows.copy()
    raise ValueError(f"Unknown fit objective: {fit_objective}")


def _initial_alpha(q_matrix: np.ndarray, q_upper: float) -> np.ndarray:
    n_personas = q_matrix.shape[1]
    weights = np.full(n_personas, 1.0 / n_personas)
    max_q = float(np.max(q_matrix @ weights))
    mass = min(1.0, q_upper / max(max_q, 1e-12))
    alpha = np.zeros(n_personas + 1)
    alpha[:-1] = mass * weights
    alpha[-1] = 1.0 - alpha[:-1].sum()
    return alpha


def _normalize_alpha(alpha: np.ndarray | None, q_matrix: np.ndarray, q_upper: float) -> np.ndarray:
    if alpha is None:
        return _initial_alpha(q_matrix, q_upper)
    out = np.maximum(np.asarray(alpha, dtype=float).ravel(), 0.0)
    if out.sum() <= 0:
        return _initial_alpha(q_matrix, q_upper)
    out = out / out.sum()
    max_q = float(np.max(q_matrix @ out[:-1]))
    if max_q > q_upper:
        scale = q_upper / max_q
        out[:-1] *= scale
        out[-1] = 1.0 - out[:-1].sum()
    return out
