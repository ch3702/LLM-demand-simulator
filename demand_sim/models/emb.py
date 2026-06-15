from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .llm_mix import fit_alpha_for_objective, choose_solver


@dataclass
class EmbModel:
    exposure_n: int
    alpha: np.ndarray
    theta: np.ndarray
    persona_ids: list[str]
    product_cols: list[str]
    persona_cols: list[str]
    product_scaler_mean: np.ndarray
    product_scaler_scale: np.ndarray
    persona_scaler_mean: np.ndarray
    persona_scaler_scale: np.ndarray
    price_mean: float
    price_scale: float
    persona_matrix_scaled: np.ndarray
    fit_objective: str
    objective_value: float

    @property
    def name(self) -> str:
        return "emb"

    def purchase_probability(self, rows: pd.DataFrame) -> np.ndarray:
        product = rows[self.product_cols].to_numpy(float)
        product_scaled = (product - self.product_scaler_mean) / self.product_scaler_scale
        price_scaled = (rows["offer_price"].to_numpy(float) - self.price_mean) / self.price_scale
        n_product = len(self.product_cols)
        n_persona = len(self.persona_cols)
        intercept = self.theta[0]
        product_beta = self.theta[1 : 1 + n_product]
        persona_beta = self.theta[1 + n_product : 1 + n_product + n_persona]
        price_beta = self.theta[-1]
        row_score = intercept + product_scaled @ product_beta + price_beta * price_scaled
        persona_score = self.persona_matrix_scaled @ persona_beta
        q_matrix = sigmoid_np(row_score[:, None] + persona_score[None, :])
        return np.clip(q_matrix @ self.alpha[:-1], 1e-9, 1.0 - 1e-9)

    def mean_demand(self, rows: pd.DataFrame) -> np.ndarray:
        return self.exposure_n * self.purchase_probability(rows)


def fit_emb(
    rows: pd.DataFrame,
    product_cols: list[str],
    persona_df: pd.DataFrame,
    persona_ids: list[str],
    persona_cols: list[str],
    exposure_n_values: list[int],
    fit_objective: str = "truncated",
    outer_iters: int = 8,
    adam_steps: int = 150,
    l2: float = 1.0,
    init_seed: int = 13,
    solver: str | None = None,
) -> EmbModel:
    fit_rows = rows[rows["demand"] > 0].copy() if fit_objective == "truncated" else rows.copy()
    if fit_rows.empty:
        raise ValueError("No rows available for embedding model fit.")
    arrays = _build_arrays(fit_rows, product_cols, persona_df, persona_cols)
    product_matrix, persona_matrix, price_scaled, demand = arrays[:4]
    product_scaler, persona_scaler, price_mean, price_scale = arrays[4:]
    solver_name = choose_solver(solver)
    best: EmbModel | None = None
    for exposure_n in tqdm(exposure_n_values, desc="fit emb"):
        if int(exposure_n) < int(demand.max()):
            continue
        rng = np.random.default_rng(init_seed + int(exposure_n))
        theta = _initial_theta(product_matrix.shape[1], persona_matrix.shape[1], demand, int(exposure_n), rng)
        alpha = None
        objective = math.inf
        for _ in range(outer_iters):
            q_matrix = _theta_to_q(theta, product_matrix, persona_matrix, price_scaled)
            alpha, objective, _status = fit_alpha_for_objective(
                q_matrix=q_matrix,
                demand=demand,
                exposure_n=int(exposure_n),
                fit_objective=fit_objective,
                solver=solver_name,
                initial_alpha=alpha,
            )
            theta = _optimize_theta(theta, alpha, product_matrix, persona_matrix, price_scaled, demand, int(exposure_n), fit_objective, adam_steps, l2)
        q_matrix = _theta_to_q(theta, product_matrix, persona_matrix, price_scaled)
        alpha, objective, _status = fit_alpha_for_objective(q_matrix, demand, int(exposure_n), fit_objective, solver_name, alpha)
        persona_matrix_scaled = persona_matrix
        candidate = EmbModel(
            exposure_n=int(exposure_n),
            alpha=alpha,
            theta=theta,
            persona_ids=persona_ids,
            product_cols=product_cols,
            persona_cols=persona_cols,
            product_scaler_mean=product_scaler.mean_,
            product_scaler_scale=product_scaler.scale_,
            persona_scaler_mean=persona_scaler.mean_,
            persona_scaler_scale=persona_scaler.scale_,
            price_mean=price_mean,
            price_scale=price_scale,
            persona_matrix_scaled=persona_matrix_scaled,
            fit_objective=fit_objective,
            objective_value=float(objective),
        )
        if best is None or candidate.objective_value < best.objective_value:
            best = candidate
    if best is None:
        raise RuntimeError("No embedding model was fit.")
    return best


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _build_arrays(rows: pd.DataFrame, product_cols: list[str], persona_df: pd.DataFrame, persona_cols: list[str]):
    product_scaler = StandardScaler()
    persona_scaler = StandardScaler()
    product_matrix = product_scaler.fit_transform(rows[product_cols].to_numpy(float)).astype(np.float64)
    persona_matrix = persona_scaler.fit_transform(persona_df[persona_cols].to_numpy(float)).astype(np.float64)
    price = rows["offer_price"].to_numpy(float)
    price_mean = float(price.mean())
    price_scale = float(price.std(ddof=0)) or 1.0
    price_scaled = ((price - price_mean) / price_scale).astype(np.float64)
    demand = rows["demand"].to_numpy(int)
    return product_matrix, persona_matrix, price_scaled, demand, product_scaler, persona_scaler, price_mean, price_scale


def _initial_theta(n_product: int, n_persona: int, demand: np.ndarray, exposure_n: int, rng: np.random.Generator) -> np.ndarray:
    theta = np.zeros(1 + n_product + n_persona + 1, dtype=np.float64)
    target = float(np.clip(np.mean(demand) / exposure_n, 1e-4, 0.25))
    theta[0] = math.log(target / (1.0 - target))
    theta[1 : 1 + n_product] = rng.normal(0.0, 0.01, size=n_product)
    start = 1 + n_product
    theta[start : start + n_persona] = rng.normal(0.0, 0.01, size=n_persona)
    theta[-1] = -0.1
    return theta


def _theta_to_q(theta: np.ndarray, product_matrix: np.ndarray, persona_matrix: np.ndarray, price_scaled: np.ndarray) -> np.ndarray:
    n_product = product_matrix.shape[1]
    n_persona = persona_matrix.shape[1]
    intercept = theta[0]
    product_beta = theta[1 : 1 + n_product]
    persona_beta = theta[1 + n_product : 1 + n_product + n_persona]
    price_beta = theta[-1]
    row_score = intercept + product_matrix @ product_beta + price_beta * price_scaled
    persona_score = persona_matrix @ persona_beta
    return sigmoid_np(row_score[:, None] + persona_score[None, :])


def _optimize_theta(
    theta0: np.ndarray,
    alpha: np.ndarray,
    product_matrix: np.ndarray,
    persona_matrix: np.ndarray,
    price_scaled: np.ndarray,
    demand: np.ndarray,
    exposure_n: int,
    fit_objective: str,
    steps: int,
    l2: float,
) -> np.ndarray:
    dtype = torch.float64
    product_t = torch.tensor(product_matrix, dtype=dtype)
    persona_t = torch.tensor(persona_matrix, dtype=dtype)
    price_t = torch.tensor(price_scaled, dtype=dtype)
    demand_t = torch.tensor(demand.astype(float), dtype=dtype)
    alpha_t = torch.tensor(alpha[:-1], dtype=dtype)
    theta = torch.tensor(theta0.copy(), dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.01)
    n_product = product_matrix.shape[1]
    n_persona = persona_matrix.shape[1]
    for _ in range(steps):
        optimizer.zero_grad()
        intercept = theta[0]
        product_beta = theta[1 : 1 + n_product]
        persona_beta = theta[1 + n_product : 1 + n_product + n_persona]
        price_beta = theta[-1]
        row_score = intercept + product_t @ product_beta + price_beta * price_t
        persona_score = persona_t @ persona_beta
        q_matrix = torch.sigmoid(row_score[:, None] + persona_score[None, :])
        q = torch.clamp(q_matrix @ alpha_t, 1e-9, 0.5 if fit_objective == "truncated" else 1.0 - 1e-9)
        log_prob = demand_t * torch.log(q) + (exposure_n - demand_t) * torch.log1p(-q)
        if fit_objective == "truncated":
            log_zero = exposure_n * torch.log1p(-q)
            log_prob = log_prob - torch.log1p(-torch.exp(log_zero))
        loss = -torch.mean(log_prob) + l2 * torch.mean(theta[1:] ** 2)
        loss.backward()
        optimizer.step()
    return theta.detach().cpu().numpy()
