"""evaluate_hotel_pricing 핵심 수식 테스트 (logit 적합·매출최적가·부트스트랩)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — 레포 표준 데이터 파이프라인
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_hotel_pricing import (  # noqa: E402
    PriceFit,
    bootstrap_optimal,
    elasticity_at,
    fit_logit,
    optimal_price,
)


def test_fit_logit_recovers_negative_slope():
    price = np.array([40000, 50000, 60000, 70000, 80000] * 12, dtype=float)
    truth = PriceFit(intercept=4.0, slope=-0.00008)
    p_buy = truth.p_buy(price)
    fit = fit_logit(price, p_buy, np.ones_like(price))
    assert fit.slope == pytest.approx(truth.slope, rel=0.05)
    assert fit.intercept == pytest.approx(truth.intercept, rel=0.05)


def test_optimal_price_respects_grid_bounds_and_revenue_shape():
    fit = PriceFit(intercept=3.0, slope=-0.00005)
    price, revenue = optimal_price(fit, 30000, 120000)
    assert 30000 <= price <= 120000
    assert revenue > 0
    lower = fit.p_buy(np.array([40000.0]))[0] * 40000
    assert revenue >= lower


def test_elasticity_is_negative_for_normal_goods():
    fit = PriceFit(intercept=3.0, slope=-0.00005)
    assert elasticity_at(fit, 50000) < 0


def test_bootstrap_returns_ordered_interval():
    rng = np.random.default_rng(7)
    rows = []
    for persona in range(6):
        for price in (40000, 50000, 60000, 70000, 80000):
            p = 1.0 / (1.0 + np.exp(-(3.5 - 0.00006 * price + rng.normal(0, 0.15))))
            rows.append({
                "persona_id": f"P{persona}",
                "article_id": 1,
                "offer_price": float(price),
                "p_buy": float(p),
                "n_customers": 100,
            })
    frame = pd.DataFrame(rows)
    lo, median, hi = bootstrap_optimal(frame, 34000, 92000, seed=11, n_boot=50)
    assert lo <= median <= hi
    assert 34000 <= lo and hi <= 92000
