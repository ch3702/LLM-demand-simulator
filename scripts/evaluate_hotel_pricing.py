"""호텔 5-SKU 시뮬레이션 → 정식 가격 분석(탄력성·매출최적가·부트스트랩 CI·실제가 대조).

evaluate_pricing.py의 표본효율 파이프라인은 다수 SKU 전제(3분할 학습≥수십 상품)라
5-SKU 호텔 세트에서는 로짓 보정이 성립하지 않는다. 이 스크립트는 5-SKU에 맞는
최소 정식 분석을 산출한다: logit(p) = a + b·price 가중최소제곱, 매출최적가, 페르소나
부트스트랩 95% CI, 실측 2026 가격 대조.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: PANDAS_OK — 레포 표준 데이터 파이프라인

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_PRODUCTS_DIR, DEFAULT_QUERY_PLAN_DIR, DEFAULT_RESPONSES_DIR
from demand_sim.data import load_probability_rows

KIMI_PRICE_PANEL = Path("/Users/robotjang/walkerhill_sim-KIMI/analysis/price_panel_prior.json")
ACTUAL_2026 = {
    1001: ("Winter on Ice Stay Package", "—", "패널 미수록(객실 패키지)"),
    2001: ("Ice Rink Skate Pass", "55,000~65,000원", "GHS-25W-02 2026 주중/주말"),
    3001: ("Pistachio Tree Cake", "100,000원", "GHS-25W-03 2026"),
    4001: ("Strawberry Tart & Yule Log", "100,000원", "GHS-25W-03 2026"),
    5001: ("Chocolate & Bread Set", "100,000원", "GHS-25W-03 2026(대리 대조)"),
}
BOOTSTRAP_B = 500
SEED = 2025


@dataclass(frozen=True, slots=True)
class PriceFit:
    intercept: float
    slope: float

    def p_buy(self, price: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.intercept + self.slope * np.asarray(price, dtype=float))))


def fit_logit(price: np.ndarray, p_buy: np.ndarray, weights: np.ndarray) -> PriceFit:
    """logit(p) = a + b·price 가중최소제곱 (p는 경계 클리핑)."""
    p = np.clip(np.asarray(p_buy, dtype=float), 0.01, 0.99)
    y = np.log(p / (1.0 - p))
    x = np.asarray(price, dtype=float)
    w = np.asarray(weights, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    root_w = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
    return PriceFit(intercept=float(coef[0]), slope=float(coef[1]))


def optimal_price(fit: PriceFit, price_lo: float, price_hi: float, personas: int = 1) -> tuple[float, float]:
    """기대매출 = p_buy(price) · price 최대화 (그리드, 0.5% 간격)."""
    grid = np.linspace(price_lo, price_hi, 400)
    revenue = fit.p_buy(grid) * grid * personas
    best = int(np.argmax(revenue))
    return float(grid[best]), float(revenue[best])


def elasticity_at(fit: PriceFit, price: float) -> float:
    p = float(fit.p_buy(np.array([price]))[0])
    return fit.slope * price * (1.0 - p)


def bootstrap_optimal(
    frame: pd.DataFrame, price_lo: float, price_hi: float, seed: int = SEED, n_boot: int = BOOTSTRAP_B,
) -> tuple[float, float, float]:
    """페르소나 단위 재표집 → 매출최적가 분포의 2.5/97.5 백분위."""
    personas = frame["persona_id"].unique()
    rng = np.random.default_rng(seed)
    optima: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(personas, size=len(personas), replace=True)
        chunks = []
        for draw, persona in enumerate(sample):
            part = frame[frame["persona_id"] == persona].copy()
            part["persona_id"] = f"{persona}#{draw}"
            chunks.append(part)
        resampled = pd.concat(chunks, ignore_index=True)
        fit = fit_logit(resampled["offer_price"], resampled["p_buy"], resampled["n_customers"])
        optima.append(optimal_price(fit, price_lo, price_hi)[0])
    lo, hi = np.percentile(optima, [2.5, 97.5])
    return float(lo), float(np.median(optima)), float(hi)


def curve_points(fit: PriceFit, price_lo: float, price_hi: float, n: int = 25) -> list[list[float]]:
    """대시보드 렌더용 수요곡선 [[price, p_buy], ...]."""
    grid = np.linspace(price_lo, price_hi, n)
    return [[round(float(price), 0), round(float(p), 4)] for price, p in zip(grid, fit.p_buy(grid))]


def analyze(frame: pd.DataFrame, bootstrap: bool = True) -> list[dict]:
    results: list[dict] = []
    for article_id, group in frame.groupby("article_id"):
        price_lo = float(group["offer_price"].min()) * 0.85
        price_hi = float(group["offer_price"].max()) * 1.15
        fit = fit_logit(group["offer_price"], group["p_buy"], group["n_customers"])
        best_price, best_revenue = optimal_price(fit, price_lo, price_hi)
        ci = bootstrap_optimal(group, price_lo, price_hi) if bootstrap else (best_price, best_price, best_price)
        label, actual, source = ACTUAL_2026.get(int(article_id), (f"상품 {article_id}", "—", "—"))
        results.append({
            "article_id": int(article_id),
            "product": label,
            "logit_intercept": round(fit.intercept, 4),
            "logit_slope_per_won": round(fit.slope, 9),
            "elasticity_at_median": round(elasticity_at(fit, float(group["offer_price"].median())), 3),
            "revenue_max_price": round(best_price, 0),
            "revenue_max_price_ci95": [round(ci[0], 0), round(ci[2], 0)],
            "expected_revenue_krw": round(best_revenue, 0),
            "price_range": [round(price_lo, 0), round(price_hi, 0)],
            "curve": curve_points(fit, price_lo, price_hi),
            "actual_2026_price": actual,
            "actual_source": source,
        })
    return results


def l3_inputs(results: list[dict]) -> list[dict]:
    """L3 방향 판정 입력: 시뮬 가격반응 방향 vs 실측 가격 변화. 실수요 부재 시 match=None(보류)."""
    rows: list[dict] = []
    for item in results:
        sim_direction = -1 if item["logit_slope_per_won"] < 0 else 1
        rows.append({
            "article_id": item["article_id"],
            "product": item["product"],
            "sim_price_response_direction": sim_direction,
            "sim_revenue_max_price": item["revenue_max_price"],
            "actual_2026_price": item["actual_2026_price"],
            "direction_match": None,
            "pending_reason": "WISE 실수요(Gate D) 부재 — 외부 방향 판정은 주말 실판매 도착 후 산출",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="5-SKU 호텔 시뮬 가격 분석")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_DIR / "llm_responses_hotel_top5.csv")
    parser.add_argument("--plan", type=Path, default=DEFAULT_QUERY_PLAN_DIR / "plan_top5_hotel.pkl")
    parser.add_argument("--product-info", type=Path, default=DEFAULT_PRODUCTS_DIR / "product_info_top5_hotel.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/pricing_hotel_top5/hotel_pricing_report.json"))
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    responses = load_probability_rows(args.responses)
    plan = pd.read_pickle(args.plan)
    weights = plan[["persona_id", "article_id", "n_customers"]].drop_duplicates()
    frame = responses.merge(weights, on=["persona_id", "article_id"], how="left")
    report = {
        "source": str(args.responses),
        "products": analyze(frame, bootstrap=not args.no_bootstrap),
    }
    report["l3_inputs"] = l3_inputs(report["products"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in report["products"]:
        ci = item["revenue_max_price_ci95"]
        print(
            f"{item['article_id']} {item['product'][:28]:28s} "
            f"매출최적 {item['revenue_max_price']:>9,.0f}원 (CI {ci[0]:,.0f}~{ci[1]:,.0f}) "
            f"탄력성 {item['elasticity_at_median']:>6.2f} | 실제 {item['actual_2026_price']}"
        )
    print(f"→ {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
