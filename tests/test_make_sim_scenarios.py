"""3×3 정책 시나리오 플랜 생성 테스트."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK — 레포 표준 데이터 파이프라인
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_sim_scenarios import (  # noqa: E402
    PROMOS,
    PRICE_DELTAS,
    ScenarioError,
    build_scenario,
    scale_prices,
    scenario_name,
    write_scenarios,
)

PLAN_ROWS = [{
    "article_id": 1001,
    "persona_id": "P001",
    "n_customers": 450,
    "persona_prompt": "데이트족",
    "prices": [400000.0, 445000.0],
    "prices_json": "[400000.0, 445000.0]",
    "price_sig": "sig-1",
    "prompt": "구매확률을 JSON으로 답하세요.\n\nOffered prices (KRW): [400000.0, 445000.0]\n",
}]


def _plan() -> pd.DataFrame:
    return pd.DataFrame(PLAN_ROWS)


class TestScalePrices:
    def test_plus_ten_percent_rounds_to_thousand(self):
        assert scale_prices("[400000.0, 445000.0]", 0.10) == [440000, 490000]

    def test_minus_ten_percent(self):
        assert scale_prices("[400000.0, 445000.0]", -0.10) == [360000, 401000]

    def test_invalid_json_raises(self):
        with pytest.raises(ScenarioError):
            scale_prices("not-json", 0.0)


class TestBuildScenario:
    def test_price_sig_is_unique_per_policy(self):
        signatures = {
            build_scenario(_plan(), delta, promo)[0]["price_sig"]
            for delta in PRICE_DELTAS for promo in PROMOS
        }
        assert len(signatures) == 9

    def test_promo_text_only_when_not_none(self):
        baseline = build_scenario(_plan(), 0.0, "none")[0]["prompt"]
        package = build_scenario(_plan(), 0.0, "package")[0]["prompt"]
        assert "정책 조건" not in baseline
        assert "패키지" in package and len(package) > len(baseline)

    def test_prompt_price_list_is_scaled(self):
        row = build_scenario(_plan(), 0.10, "none")[0]
        assert "[440000, 490000]" in row["prompt"]
        assert "[400000.0, 445000.0]" not in row["prompt"]

    def test_missing_price_line_raises(self):
        plan = _plan()
        plan.loc[0, "prompt"] = "가격 목록이 없는 프롬프트"
        with pytest.raises(ScenarioError):
            build_scenario(plan, 0.0, "none")

    def test_baseline_prices_unchanged(self):
        row = build_scenario(_plan(), 0.0, "none")[0]
        assert json.loads(row["prices_json"]) == [400000.0, 445000.0]


class TestWriteScenarios:
    def test_writes_nine_files_and_index(self, tmp_path):
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(PLAN_ROWS), encoding="utf-8")
        output = tmp_path / "scenarios"
        index = write_scenarios(plan_path, output)
        assert len(index) == 9
        files = sorted(path.name for path in output.glob("price_*.json"))
        assert len(files) == 9
        assert f"{scenario_name(0.1, 'breakfast')}.json" in files
        payload = json.loads((output / "scenarios_index.json").read_text(encoding="utf-8"))
        assert len(payload["scenarios"]) == 9
