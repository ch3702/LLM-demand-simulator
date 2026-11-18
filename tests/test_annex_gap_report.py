"""부록 A 공백 리포트 로직 테스트."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from report_annex_gaps import GapRow, build_gap_rows, render_markdown, status_of  # noqa: E402

REGISTRY = [
    {"product_id": "GHS-25W-02", "hotel": "그랜드 하얏트 서울"},
    {"product_id": "WH-26W-01", "hotel": "워커힐"},
]
SHEETS = {
    "C_블로그": [{"상품ID": "GHS-25W-02", "상품 언급량": 59, "판정 편수": 30}],
    "C_협찬판정": [
        {"상품ID": "GHS-25W-02", "분류": "협찬"},
        {"상품ID": "GHS-25W-02", "분류": "자비 표시"},
    ],
    "D_게시물": [{"상품ID": "GHS-25W-02"}],
    "D_댓글": [{"상품ID": "GHS-25W-02"}],
}


class TestBuildGapRows:
    def test_complete_row(self):
        rows = build_gap_rows(REGISTRY, SHEETS)
        first = rows[0]
        assert first.product_id == "GHS-25W-02"
        assert first.mentions == 59
        assert first.judged == 30
        assert (first.subsidy, first.self_funded, first.corporate) == (1, 1, 0)
        assert first.has_post and first.has_comments
        assert status_of(first) == "완료"

    def test_missing_row(self):
        rows = build_gap_rows(REGISTRY, SHEETS)
        second = rows[1]
        assert second.mentions is None
        assert status_of(second) == "누락"

    def test_partial_row_when_only_mentions(self):
        sheets = {"C_블로그": [{"상품ID": "GHS-25W-02", "상품 언급량": 10, "판정 편수": 5}],
                  "C_협찬판정": [], "D_게시물": [], "D_댓글": []}
        row = build_gap_rows(REGISTRY[:1], sheets)[0]
        assert status_of(row) == "부분"

    def test_markdown_contains_sections(self):
        markdown = render_markdown(build_gap_rows(REGISTRY, SHEETS))
        assert "# 부록 A 공백 리포트" in markdown
        assert "그랜드 하얏트 서울" in markdown
        assert "30/30" in markdown
        assert "누락" in markdown
