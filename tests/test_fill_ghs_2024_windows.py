"""GHS 2024 판매기간 채움 테스트 (창 산정·신뢰도 마킹·행 갱신)."""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fill_ghs_2024_windows import (  # noqa: E402
    SALE_WINDOWS,
    WindowError,
    collection_window,
    fill_workbook,
)

PRODUCT_HEADERS = ["상품ID", "호텔", "연도", "상품명", "판매 시작일", "판매 종료일", "확인"]
BLOG_HEADERS = ["상품ID", "호텔", "기간 창 시작(판매 시작일)", "기간 창 끝(판매 종료 +30일)"]


def _build(path: Path) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, headers in (("B_상품", PRODUCT_HEADERS), ("C_블로그", BLOG_HEADERS)):
        sheet = workbook.create_sheet(name)
        sheet.append(headers)
        sheet.append(headers)
        for pid in SALE_WINDOWS:
            sheet.append([pid, "그랜드 하얏트 서울"] + [""] * (len(headers) - 2))
    workbook.save(path)


class TestWindow:
    def test_collection_window_adds_thirty_days(self):
        assert collection_window(date(2024, 12, 31)) == date(2025, 1, 30)
        assert collection_window(date(2025, 3, 3)) == date(2025, 4, 2)

    def test_bingsu_marked_estimated_others_confirmed(self):
        confidences = {pid: spec["confidence"] for pid, spec in SALE_WINDOWS.items()}
        assert confidences["GHS-24S-02"] == "추정"
        assert confidences["GHS-24W-01"] == "확정"
        assert confidences["GHS-24S-01"] == "확정"

    def test_every_window_has_sources(self):
        for pid, spec in SALE_WINDOWS.items():
            assert spec["sources"], pid

    def test_fill_workbook_updates_rows_and_windows(self, tmp_path):
        path = tmp_path / "copy.xlsx"
        _build(path)
        updated = fill_workbook(path)
        assert len(updated) == 4
        workbook = load_workbook(path)
        sheet = workbook["B_상품"]
        headers = {sheet.cell(1, col).value: col for col in range(1, sheet.max_column + 1)}
        def as_date(value):
            return value.date() if isinstance(value, datetime) else value

        for row in range(3, sheet.max_row + 1):
            pid = sheet.cell(row, 1).value
            if pid == "GHS-24S-01":
                assert as_date(sheet.cell(row, headers["판매 시작일"]).value) == date(2024, 5, 24)
                assert as_date(sheet.cell(row, headers["판매 종료일"]).value) == date(2024, 9, 30)
                assert "확정" in sheet.cell(row, headers["확인"]).value
        blog = workbook["C_블로그"]
        blog_headers = {blog.cell(1, col).value: col for col in range(1, blog.max_column + 1)}
        values = {}
        for row in range(3, blog.max_row + 1):
            values[blog.cell(row, 1).value] = (
                as_date(blog.cell(row, blog_headers["기간 창 시작(판매 시작일)"]).value),
                as_date(blog.cell(row, blog_headers["기간 창 끝(판매 종료 +30일)"]).value),
            )
        assert values["GHS-24W-01"] == (date(2024, 12, 1), date(2025, 4, 2))
        assert all(isinstance(v, date) for pair in values.values() for v in pair)
        workbook.close()

    def test_missing_product_row_raises(self, tmp_path):
        path = tmp_path / "copy.xlsx"
        workbook = Workbook()
        workbook.remove(workbook.active)
        sheet = workbook.create_sheet("B_상품")
        sheet.append(PRODUCT_HEADERS)
        sheet.append(PRODUCT_HEADERS)
        blog_sheet = workbook.create_sheet("C_블로그")
        blog_sheet.append(BLOG_HEADERS)
        blog_sheet.append(BLOG_HEADERS)
        workbook.save(path)
        try:
            fill_workbook(path)
            raise AssertionError("WindowError가 발생해야 합니다")
        except WindowError:
            pass
