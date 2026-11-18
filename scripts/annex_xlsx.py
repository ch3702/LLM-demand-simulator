"""부록 워크북(호텔조사_정리.xlsx) 직접기입 — C_블로그(시트3)·C_협찬판정(시트4).

표 구조: 1~2행 헤더(2행은 중복 헤더), 데이터는 3행부터. 기입은 수집 대상 상품 행만
치환하고 미수집 상품 행은 그대로 보존한다. 저장 전 원본 타임스탬프 백업을 만든다.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from naver_survey_io import Sheet4Row, canonical_label

SHEET3_NAME = "C_블로그"
SHEET4_NAME = "C_협찬판정"
SHEET3_WIDTH = 18
SHEET4_WIDTH = 7
_DATA_START_ROW = 3  # 1~2행 헤더


def _row_values(sheet: Worksheet, row: int, width: int) -> list[str]:
    values: list[str] = []
    for column in range(1, width + 1):
        raw = sheet.cell(row, column).value
        values.append("" if raw is None else str(raw).strip())
    return values


def _existing_rows(sheet: Worksheet, width: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in range(_DATA_START_ROW, sheet.max_row + 1):
        if sheet.cell(row, 1).value in (None, ""):
            continue
        rows.append(_row_values(sheet, row, width))
    return rows


def _replace_data_rows(sheet: Worksheet, rows: list[list[str]]) -> None:
    """3행 이후 전체 삭제 후 kept+rows 재기입 (헤더 1~2행 보존)."""
    last = max(sheet.max_row, _DATA_START_ROW - 1)
    if last >= _DATA_START_ROW:
        sheet.delete_rows(_DATA_START_ROW, last - _DATA_START_ROW + 1)
    for index, row in enumerate(rows):
        for column, value in enumerate(row, start=1):
            sheet.cell(_DATA_START_ROW + index, column).value = value


def _upsert_table(sheet: Worksheet, matrix: list[list[str]], width: int) -> None:
    """수집 대상 상품 행은 matrix 값으로 치환, 미수집 상품 행은 원문 보존."""
    matrix_ids = {row[0] for row in matrix}
    kept = [row for row in _existing_rows(sheet, width) if row[0] not in matrix_ids]
    _replace_data_rows(sheet, kept + matrix)


def read_sheet4_rows(xlsx_path: Path) -> list[Sheet4Row]:
    """워크북 C_협찬판정의 수기 행을 Sheet4Row로 읽는다 (라벨 별칭 정규화)."""
    workbook = load_workbook(xlsx_path, data_only=True)
    try:
        sheet = workbook[SHEET4_NAME]
        return [
            Sheet4Row(
                product_id=values[0], hotel=values[1], segment=values[2],
                postdate=values[3], link=values[4],
                label=canonical_label(values[5]), evidence=values[6],
            )
            for values in _existing_rows(sheet, SHEET4_WIDTH)
        ]
    finally:
        workbook.close()


def write_annex(
    xlsx_path: Path,
    sheet3_rows: list[list[str]],
    sheet4_rows: list[list[str]],
    backup: bool = True,
) -> Path | None:
    """시트3·4 행렬을 워크북에 기입. 백업 경로 반환(backup=False면 None)."""
    workbook = load_workbook(xlsx_path)
    _upsert_table(workbook[SHEET3_NAME], sheet3_rows, SHEET3_WIDTH)
    _upsert_table(workbook[SHEET4_NAME], sheet4_rows, SHEET4_WIDTH)
    backup_path: Path | None = None
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = xlsx_path.with_name(f"{xlsx_path.stem}_backup_{stamp}{xlsx_path.suffix}")
        shutil.copy2(xlsx_path, backup_path)
    workbook.save(xlsx_path)
    return backup_path
