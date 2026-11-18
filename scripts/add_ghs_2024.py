"""GHS(그랜드 하얏트 서울) 2024년 시즌 행 추가 — 사본 워크북에만 기록.

원칙:
- 원본 워크북은 수정하지 않는다(사본 출력 + 해시 검증).
- 창작 금지: 가격은 B_반복 2024 열에서 역추출한 값만 사용하고,
  판매 기간·언급량·협찬 판정·게시물 지표는 빈칸(수집 대기)으로 둔다.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

DEFAULT_SOURCE = Path("/Users/robotjang/Downloads/호텔조사_정리 (2).xlsx")
DEFAULT_OUTPUT = Path("/Users/robotjang/Downloads/호텔조사_정리_GHS2024_v1.xlsx")
HOTEL = "그랜드 하얏트 서울"


@dataclass(frozen=True)
class Product2024:
    product_id: str
    name: str
    product_type: str
    season_label: str
    representative_price: int
    price_range: str
    panel_note: str


PRODUCTS_2024 = (
    Product2024("GHS-24W-01", "야외 아이스링크 이용권 (단독/통합)", "야외·체험", "2024 연말",
                55000, "주중 5.5만 / 주말 5.8만 · 단독입장 1.2~1.5만", "B_반복 2024 열 역추출"),
    Product2024("GHS-24W-02", "더 델리 연말 홀케이크 라인업", "케이크·디저트", "2024 연말",
                90000, "9만~12만 원대", "B_반복 2024 열 역추출(하한)"),
    Product2024("GHS-24S-01", "여름 풀사이드 바비큐", "뷔페", "2024 여름",
                140000, "성인 14만~15만 원대", "B_반복 2024 열 역추출(하한)"),
    Product2024("GHS-24S-02", "여름 시그니처 빙수 (갤러리)", "케이크·디저트", "2024 여름",
                100000, "망고 빙수 10만 원대", "B_반복 2024 열 역추출"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _headers(sheet) -> dict[str, int]:
    return {
        str(sheet.cell(1, col).value).strip(): col
        for col in range(1, sheet.max_column + 1)
        if sheet.cell(1, col).value not in (None, "")
    }


def _append_row(sheet, values: dict[str, Any]) -> int:
    header = _headers(sheet)
    row = sheet.max_row + 1
    for name, value in values.items():
        if name in header:
            sheet.cell(row, header[name]).value = value
    return row


def add_2024_rows(workbook_path: Path) -> dict[str, int]:
    workbook = load_workbook(workbook_path)
    counts = {"B_상품": 0, "C_블로그": 0, "D_게시물": 0, "D_댓글": 0}
    names = {item.product_id: item.name for item in PRODUCTS_2024}
    for item in PRODUCTS_2024:
        _append_row(workbook["B_상품"], {
            "상품ID": item.product_id,
            "호텔": HOTEL,
            "연도": "2024",
            "상품명": item.name,
            "상품 유형": item.product_type,
            "대표 상품": item.season_label,
            "대표 가격": item.representative_price,
            "가격 범위": item.price_range,
            "출처": item.panel_note,
            "확인": "부분 — 판매 기간·언급량·협찬 표본 수집 필요",
            "비고": "2024 소급 추가(자동 구조화). 숫자 미입력 칸은 수집 후 기입",
        })
        counts["B_상품"] += 1
        _append_row(workbook["C_블로그"], {
            "상품ID": item.product_id,
            "호텔": HOTEL,
            "검색어(호텔명+이벤트명)": f"{HOTEL} {names[item.product_id]}",
        })
        counts["C_블로그"] += 1
        _append_row(workbook["D_게시물"], {"상품ID": item.product_id, "호텔": HOTEL})
        counts["D_게시물"] += 1
        _append_row(workbook["D_댓글"], {"상품ID": item.product_id, "호텔": HOTEL})
        counts["D_댓글"] += 1
    workbook.save(workbook_path)
    workbook.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="GHS 2024 시즌 행 추가(사본)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.exists():
        raise SystemExit(f"원본 워크북이 없습니다: {args.source}")
    original_hash = sha256(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.output)
    counts = add_2024_rows(args.output)
    if sha256(args.source) != original_hash:
        raise SystemExit("원본 워크북이 변경되었습니다 — 중단")
    print(f"사본 생성: {args.output}")
    print(f"추가 행: {counts} · 원본 무변경 확인(sha256 동일)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
