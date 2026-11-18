"""LLM 시뮬 입력 플랜(.pkl/.json) 로더 테스트."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK — 레포 표준 데이터 파이프라인
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demand_sim.data import PlanFormatError, load_plan  # noqa: E402

PLAN_ROWS = [
    {
        "article_id": 1001,
        "persona_id": "P001",
        "n_customers": 450,
        "persona_prompt": "28세 데이트족",
        "prices": [400000.0, 445000.0],
        "prices_json": "[400000.0, 445000.0]",
        "price_sig": "sig-1",
        "prompt": "구매확률을 JSON으로 답하세요.",
    },
    {
        "article_id": 2001,
        "persona_id": "P002",
        "n_customers": 300,
        "persona_prompt": "가족 여행객",
        "prices": [45000.0, 49000.0],
        "prices_json": "[45000.0, 49000.0]",
        "price_sig": "sig-2",
        "prompt": "구매확률을 JSON으로 답하세요.",
    },
]


def _write_pickle(path: Path) -> None:
    pd.DataFrame(PLAN_ROWS).to_pickle(path)


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


class TestLoadPlan:
    def test_pickle_and_json_yield_equivalent_frames(self, tmp_path):
        pkl = tmp_path / "plan.pkl"
        js = tmp_path / "plan.json"
        _write_pickle(pkl)
        _write_json(js, PLAN_ROWS)
        from_pkl = load_plan(pkl)
        from_json = load_plan(js)
        assert list(from_pkl.columns) == list(from_json.columns)
        assert from_pkl["price_sig"].tolist() == from_json["price_sig"].tolist()
        assert from_pkl["prices"].tolist() == from_json["prices"].tolist()
        assert from_pkl["n_customers"].tolist() == from_json["n_customers"].tolist()

    def test_json_missing_required_column_raises(self, tmp_path):
        js = tmp_path / "plan.json"
        broken = [{k: v for k, v in PLAN_ROWS[0].items() if k != "prompt"}]
        _write_json(js, broken)
        with pytest.raises(PlanFormatError):
            load_plan(js)

    def test_json_not_a_list_raises(self, tmp_path):
        js = tmp_path / "plan.json"
        js.write_text('{"article_id": 1}', encoding="utf-8")
        with pytest.raises(PlanFormatError):
            load_plan(js)

    def test_json_invalid_raises(self, tmp_path):
        js = tmp_path / "plan.json"
        js.write_text("not json", encoding="utf-8")
        with pytest.raises(PlanFormatError):
            load_plan(js)

    def test_unknown_suffix_raises(self, tmp_path):
        path = tmp_path / "plan.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(PlanFormatError):
            load_plan(path)
