from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


@dataclass
class GaussianModel:
    beta: np.ndarray
    sigma_value: float
    product_cols: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    train_loss: float
    price_mean: float = 0.0
    price_scale: float = 1.0

    @property
    def name(self) -> str:
        return "gaussian"

    @property
    def sigma(self) -> float:
        return float(getattr(self, "sigma_value", np.log1p(np.exp(getattr(self, "aux", 1.0))) + 1e-4))

    def design(self, rows: pd.DataFrame) -> np.ndarray:
        embedding = rows[self.product_cols].to_numpy(float)
        embedding_scaled = (embedding - self.scaler_mean) / self.scaler_scale
        price_mean = getattr(self, "price_mean", 0.0)
        price_scale = getattr(self, "price_scale", 1.0) or 1.0
        price_scaled = (rows["offer_price"].to_numpy(float).reshape(-1, 1) - price_mean) / price_scale
        return np.hstack(
            [
                np.ones((len(rows), 1), dtype=float),
                price_scaled,
                embedding_scaled,
            ]
        )

    def mean_demand(self, rows: pd.DataFrame) -> np.ndarray:
        return self.design(rows) @ self.beta

    def pmf(self, rows: pd.DataFrame, support: np.ndarray) -> np.ndarray:
        mu = self.mean_demand(rows)
        return rounded_gaussian_pmf(mu, self.sigma, support)


def fit_gaussian(
    rows: pd.DataFrame,
    product_cols: list[str],
    steps: int = 1500,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-3,
) -> GaussianModel:
    _ = steps, learning_rate
    x_train, scaler, price_mean, price_scale = build_design(rows, product_cols)
    y = rows["demand"].to_numpy(float)
    beta, sigma, loss = fit_gaussian_ridge(x_train, y, ridge=weight_decay)
    return GaussianModel(
        beta=beta,
        sigma_value=sigma,
        product_cols=product_cols,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        train_loss=loss,
        price_mean=price_mean,
        price_scale=price_scale,
    )


def build_design(
    rows: pd.DataFrame,
    product_cols: list[str],
    scaler: StandardScaler | None = None,
    price_mean: float | None = None,
    price_scale: float | None = None,
) -> tuple[np.ndarray, StandardScaler, float, float]:
    price = rows["offer_price"].to_numpy(float).reshape(-1, 1)
    embedding = rows[product_cols].to_numpy(float)
    if price_mean is None:
        price_mean = float(price.mean())
    if price_scale is None:
        price_scale = float(price.std(ddof=0)) or 1.0
    price_scaled = (price - price_mean) / price_scale
    if scaler is None:
        scaler = StandardScaler()
        embedding_scaled = scaler.fit_transform(embedding)
    else:
        embedding_scaled = scaler.transform(embedding)
    x = np.hstack([np.ones((len(rows), 1)), price_scaled, embedding_scaled]).astype(np.float64)
    return x, scaler, price_mean, price_scale


def fit_gaussian_ridge(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(max(ridge, 0.0))
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    residual = y - x @ beta
    sigma = float(np.sqrt(np.mean(residual**2)))
    sigma = max(sigma, 1e-4)
    loss = gaussian_nll_value(y=y, mu=x @ beta, sigma=sigma)
    return beta.astype(np.float64), sigma, loss


def gaussian_nll_value(y: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = float(max(sigma, 1e-12))
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2))


def rounded_gaussian_pmf(mu: np.ndarray, sigma: float, support: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=int)
    pmf = np.zeros((len(mu), len(support)), dtype=float)
    for idx, demand in enumerate(support):
        if demand == 0:
            pmf[:, idx] = norm.cdf(0.5, loc=mu, scale=sigma)
        else:
            pmf[:, idx] = norm.cdf(demand + 0.5, loc=mu, scale=sigma) - norm.cdf(demand - 0.5, loc=mu, scale=sigma)
    pmf = np.clip(pmf, 0.0, None)
    pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0.0, None)
    return pmf / np.clip(pmf.sum(axis=1, keepdims=True), 1e-12, None)
