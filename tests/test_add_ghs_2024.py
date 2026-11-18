"""GHS 2024 행 추가 스크립트 테스트 (사본 구조 검증)."""
from __future__ import annotations

import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from add_ghs_2024 import PRODUCTS_2024, add_2024_rows  # noqa: E402

SHEET_HEADERS = {
    "B_상품": ["상품ID", "호텔", "연도", "상품명", "상품 유형", "대표 상품", "대표 가격", "가격 범위", "출처", "확인", "비고"],
    "C_블로그": ["상품ID", "호텔", "검색어(호텔명+이벤트명)", "상품 언급량", "판정 편수"],
    "D_게시물": ["상품ID", "호텔", "공식 게시일", "좋아요(숨김이면 빈칸)"],
    "D_댓글": ["상품ID", "호텔", "게시물 날짜", "읽은 댓글 수"],
}


def _build(path: Path) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, headers in SHEET_HEADERS.items():
        sheet = workbook.create_sheet(name)
        sheet.append(headers)
        sheet.append(headers)  # 워크북 규약: 2행 중복 헤더
        sheet.append(["GHS-25W-02", "그랜드 하얏트 서울"] + [""] * (len(headers) - 2))
    workbook.save(path)


class TestAdd2024Rows:
    def test_appends_four_rows_per_sheet(self, tmp_path):
        target = tmp_path / "copy.xlsx"
        _build(target)
        counts = add_2024_rows(target)
        assert counts == {"B_상품": 4, "C_블로그": 4, "D_게시물": 4, "D_댓글": 4}
        workbook = load_workbook(target)
        for name in SHEET_HEADERS:
            ids = [workbook[name].cell(row, 1).value for row in range(3, workbook[name].max_row + 1)]
            added = [value for value in ids if str(value).startswith("GHS-24")]
            assert len(added) == 4, name
        workbook.close()

    def test_prices_come_from_panel_values_only(self):
        prices = {item.product_id: item.representative_price for item in PRODUCTS_2024}
        assert prices["GHS-24W-01"] == 55000
        assert prices["GHS-24S-01"] == 140000

    def test_metrics_left_blank_for_collection(self, tmp_path):
        target = tmp_path / "copy.xlsx"
        _build(target)
        add_2024_rows(target)
        workbook = load_workbook(target)
        sheet = workbook["C_블로그"]
        last = sheet.max_row
        assert sheet.cell(last, 1).value == "GHS-24S-02"
        assert sheet.cell(last, 4).value in (None, "")
        workbook.close()
