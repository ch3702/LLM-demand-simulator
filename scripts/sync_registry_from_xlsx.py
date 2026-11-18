"""부록 워크북(호텔조사_정리.xlsx) B_상품 → products_naver.csv 단일 원천 동기화.

- B_상품(검증 완료 SKU)을 레지스트리 행으로 변환: window_end = 판매 종료 +30일.
- C_블로그 스코프 지정 시 해당 시트의 상품ID만 수집 대상으로 하고 검색어·키워드·비고를
  C_블로그 정화본에서 채택한다(부록 양식 충실도 우선).
- 스코프 미지정 시 prefixes로 걸러 파생 검색어(호텔명+괄호 제거 상품명)를 만든다.
- 기존 레지스트리에서 워크북에 없는 상품ID(예: 워커힐 스냅샷 행)는 원문 보존 병합.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

from naver_survey_io import read_registry_rows

WINDOW_TAIL_DAYS = 30
PRODUCTS_SHEET = "B_상품"
_DATA_START_ROW = 3  # 1~2행 헤더(2행 중복), 데이터는 3행부터
_DEFAULT_SCOPE_SHEET = "C_블로그"
_REGISTRY_HEADER = (
    "product_id", "hotel", "query", "hotel_query", "window_start", "window_end",
    "seg2_start", "seg3_start", "keywords", "note", "active",
)
_HOTEL_QUERY = {
    "그랜드 하얏트 서울": "그랜드 하얏트 서울",
    "서울신라": "신라호텔",
    "포시즌스 호텔 서울": "포시즌스 호텔 서울",
}
_KEYWORD_DEFAULTS = {
    "객실 패키지": "호캉스, 패키지, 프로모션",
    "케이크·디저트": "케이크, 디저트, 예약",
    "뷔페": "뷔페, 예약, 가격",
    "크리스마스 디너": "크리스마스디너, 예약, 코스",
    "바·라운지 시즌 메뉴": "시즌메뉴, 칵테일, 분위기",
    "야외·체험": "이용권, 예약, 가격",
}
_PAREN_RE = re.compile(r"\([^)]*\)")
_SPACE_RE = re.compile(r"\s+")


class SyncError(ValueError):
    """워크북 구조·값이 기대와 어긋날 때."""


type CellValue = str | int | float | datetime | date | None


@dataclass(frozen=True, slots=True)
class SyncResult:
    registry_path: Path
    workbook_rows: int
    preserved_ids: tuple[str, ...]


def _cell_text(value: CellValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    raise SyncError(f"unsupported cell value {value!r}")


def _cell_date(value: CellValue) -> date:
    if isinstance(value, bool):
        raise SyncError(f"unsupported date cell {value!r}")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError as error:
            raise SyncError(f"invalid date text {value!r}") from error
    raise SyncError(f"unsupported date cell {value!r}")


def _cell_price(value: CellValue) -> int:
    if isinstance(value, bool):
        raise SyncError(f"unsupported price cell {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.replace(",", "").strip()))
        except ValueError as error:
            raise SyncError(f"invalid price text {value!r}") from error
    raise SyncError(f"unsupported price cell {value!r}")


def _read_table(sheet: Worksheet) -> list[dict[str, CellValue]]:
    header: dict[str, int] = {}
    for column in range(1, sheet.max_column + 1):
        name = sheet.cell(1, column).value
        if isinstance(name, str) and name.strip():
            header[name.strip()] = column
    rows: list[dict[str, CellValue]] = []
    for row in range(_DATA_START_ROW, sheet.max_row + 1):
        if sheet.cell(row, 1).value in (None, ""):
            continue
        rows.append({name: sheet.cell(row, index).value for name, index in header.items()})
    return rows


def _clean_name(name: str) -> str:
    return _SPACE_RE.sub(" ", _PAREN_RE.sub("", name)).strip()


def _scope_index(sheet: Worksheet | None) -> dict[str, dict[str, str]]:
    """C_블로그 상품ID → 검색어·키워드·비고 정화본."""
    if sheet is None:
        return {}
    scoped: dict[str, dict[str, str]] = {}
    for row in range(_DATA_START_ROW, sheet.max_row + 1):
        pid = sheet.cell(row, 1).value
        if pid in (None, ""):
            continue
        scoped[str(pid).strip()] = {
            "query": _cell_text(sheet.cell(row, 3).value),
            "keywords": _cell_text(sheet.cell(row, 16).value),
            "note": _cell_text(sheet.cell(row, 18).value),
        }
    return scoped


def _build_rows(sheet: Worksheet, scope: dict[str, dict[str, str]], prefixes: tuple[str, ...]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in _read_table(sheet):
        pid = _cell_text(record.get("상품ID"))
        if not pid.startswith(prefixes):
            continue
        if scope and pid not in scope:
            continue
        hotel = _cell_text(record.get("호텔"))
        hotel_query = _HOTEL_QUERY.get(hotel, hotel)
        window_start = _cell_date(record.get("판매 시작일"))
        window_end = _cell_date(record.get("판매 종료일")) + timedelta(days=WINDOW_TAIL_DAYS)
        curated = scope.get(pid, {})
        if curated:
            query = curated["query"]
            keywords = curated["keywords"]
            note = curated["note"]
        else:
            query = f"{hotel_query} {_clean_name(_cell_text(record.get('상품명')))}".strip()
            ptype = _cell_text(record.get("상품 유형"))
            keywords = _KEYWORD_DEFAULTS.get(ptype, "시즌상품")
            price = _cell_price(record.get("대표 가격"))
            note = f"대표 {price:,}원({_cell_text(record.get('출처'))}) · [xlsx 동기화 {date.today().isoformat()} — 쿼리 자동 생성, 확인 권장]"
        rows.append([
            pid, hotel, query, hotel_query,
            window_start.isoformat(), window_end.isoformat(), "", "",
            keywords, note, "1",
        ])
    return rows


def sync_registry(
    xlsx_path: Path,
    out_path: Path,
    prefixes: tuple[str, ...],
    scope_sheet: str | None = None,
) -> SyncResult:
    workbook = load_workbook(xlsx_path, data_only=True)
    try:
        scope = _scope_index(workbook[scope_sheet] if scope_sheet else None)
        rows = _build_rows(workbook[PRODUCTS_SHEET], scope, prefixes)
    finally:
        workbook.close()
    if not rows:
        raise SyncError(f"no workbook rows matched prefixes {prefixes} (scope={scope_sheet})")
    workbook_ids = {row[0] for row in rows}
    preserved: list[dict[str, str]] = []
    if out_path.exists():
        preserved = [row for row in read_registry_rows(out_path) if row.get("product_id") not in workbook_ids]
    scope_desc = scope_sheet if scope_sheet else "prefixes=" + ",".join(prefixes)
    lines = [
        f"# {out_path.name} — 부록 워크북 B_상품 동기화 단일 원천 ({date.today().isoformat()})",
        f"# 원천: {xlsx_path.name} · 스코프: {scope_desc} · 도구: scripts/sync_registry_from_xlsx.py",
    ]
    body = [[row.get(name, "") for name in _REGISTRY_HEADER] for row in preserved]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_REGISTRY_HEADER)
    writer.writerows(rows + body)
    out_path.write_text("\n".join(lines) + "\n" + buffer.getvalue(), encoding="utf-8")
    return SyncResult(registry_path=out_path, workbook_rows=len(rows), preserved_ids=tuple(row["product_id"] for row in preserved))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync products_naver registry from the annex workbook B_상품 sheet.")
    parser.add_argument("--xlsx", type=Path, default=Path("/Users/robotjang/Downloads/호텔조사_정리 (2).xlsx"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefixes", default="GHS,SHL,FSS", help="쉼표 구분 상품ID 접두어 필터")
    parser.add_argument("--scope-sheet", default=_DEFAULT_SCOPE_SHEET, help="수집 스코프 시트(C_블로그). 비활성화하려면 none")
    args = parser.parse_args()
    prefixes = tuple(prefix.strip() for prefix in args.prefixes.split(",") if prefix.strip())
    scope = None if args.scope_sheet.lower() == "none" else args.scope_sheet
    result = sync_registry(args.xlsx, args.out, prefixes, scope)
    print(f"동기화 완료: {result.registry_path}  워크북 행 {result.workbook_rows}건 · 보존 {len(result.preserved_ids)}건 {result.preserved_ids or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
