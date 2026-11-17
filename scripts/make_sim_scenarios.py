"""3×3 정책 격자(가격 ±10% × 프로모션) LLM 시뮬 시나리오 플랜 생성기."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.data import load_plan

PRICE_DELTAS = (-0.10, 0.0, 0.10)
PROMOS = ("none", "package", "breakfast")
PROMO_TEXT = {
    "none": "",
    "package": "\n정책 조건: 객실+부대시설 패키지로 제안됩니다.",
    "breakfast": "\n정책 조건: 조식 2인 포함 혜택이 추가됩니다.",
}
PRICE_ROUND = 1000
PRICE_LINE = re.compile(r"Offered prices \(KRW\): \[[^\]]*\]")


class ScenarioError(ValueError):
    """시나리오 생성 입력이 기대와 어긋날 때."""


def scenario_name(delta: float, promo: str) -> str:
    return f"price_{int(round(delta * 100)):+03d}_promo_{promo}"


def scale_prices(raw: str, delta: float) -> list[float]:
    try:
        prices = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ScenarioError(f"prices_json 이 올바르지 않습니다: {raw!r}") from error
    return [math.floor(float(price) * (1.0 + delta) / PRICE_ROUND + 0.5) * PRICE_ROUND for price in prices]


def rewrite_prompt(prompt: str, scaled: list[float], promo: str) -> str:
    """프롬프트의 가격 목록을 스케일된 가격으로 교체하고 프로모션 문구를 덧붙인다."""
    replaced, count = PRICE_LINE.subn(f"Offered prices (KRW): {json.dumps(scaled)}", prompt)
    if count == 0:
        raise ScenarioError("프롬프트에서 'Offered prices (KRW)' 목록을 찾지 못했습니다")
    return f"{replaced}{PROMO_TEXT[promo]}"


def build_scenario(plan, delta: float, promo: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        scaled = scale_prices(str(row["prices_json"]), delta)
        rows.append({
            **{key: row[key] for key in plan.columns if key not in {"prices", "prices_json", "price_sig", "prompt"}},
            "prices": scaled,
            "prices_json": json.dumps(scaled),
            "price_sig": f"{row['price_sig']}|p{delta:+.2f}|{promo}",
            "prompt": rewrite_prompt(str(row["prompt"]), scaled, promo),
        })
    return rows


def write_scenarios(plan_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    plan = load_plan(plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for delta in PRICE_DELTAS:
        for promo in PROMOS:
            name = scenario_name(delta, promo)
            rows = build_scenario(plan, delta, promo)
            target = output_dir / f"{name}.json"
            target.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            index.append({
                "scenario": name,
                "price_delta": delta,
                "promo": promo,
                "rows": len(rows),
                "file": str(target),
            })
    (output_dir / "scenarios_index.json").write_text(
        json.dumps({"source": str(plan_path), "scenarios": index}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="3×3 정책 시나리오 플랜 생성")
    parser.add_argument("--plan", type=Path, default=Path("outputs/query_plan/plan_top5_hotel.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/query_plan/scenarios"))
    args = parser.parse_args()
    index = write_scenarios(args.plan, args.output_dir)
    for item in index:
        print(f"{item['scenario']:36s} {item['rows']:>3d}행  가격{item['price_delta']:+.0%} × {item['promo']}")
    print(f"→ {args.output_dir}/scenarios_index.json ({len(index)}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
