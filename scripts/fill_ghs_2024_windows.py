"""GHS 2024 SKU 판매 기간·수집 창 채움 — 웹 검증 출처와 함께 (사본 워크북만 수정).

출처 (2026-11-18 검증):
- 아이스링크 2024-12-01~2025-03-03: 디지털조선 2024-12-03, 호텔앤레스토랑 2024-11-15,
  코스모폴리탄 2024-11-29, NOL — "12월 1일 개장, 2025년 3월 3일까지" 다중 교차 확인.
- 홀케이크 픽업 2024-11-16~2024-12-31: 더 델리 페스티브 케이크 (localnaeil 2024-12-05
  "12월 31일까지", 네이버 예약 출처 블로그 "픽업 11/16~12/31", 얼리버드 ~12/8 15%).
- 풀사이드 BBQ 2024-05-24~2024-09-30: 인사이트 2024-05-20, 호텔앤레스토랑 2024-05-10,
  TableCheck 2024 플랜, 이용 후기 2024-08-20 — 다중 교차 확인.
- 빙수 2024-06-01~2024-08-31 [추정]: "망고 철 6월부터" 개시 패턴(AsiaN 2023)·과거 시즌
  8/31 종료(여행신문 2021)·2024 망고 애프터눈티 8/31 종료(아주경제 2024-06-07) 근거.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from openpyxl import load_workbook

WORKBOOK = Path("/Users/robotjang/Downloads/호텔조사_정리_GHS2024_v1.xlsx")
WINDOW_PLUS_DAYS = 30

SALE_WINDOWS = {
    "GHS-24W-01": {
        "start": date(2024, 12, 1), "end": date(2025, 3, 3), "confidence": "확정",
        "sources": "digitalchosun.dizzo.com(2024-12-03) · hotelrestaurant.co.kr idxno=14819 · NOL world",
        "keywords": "마리떼, 콜라보, 운영기간, 네이버예약, 주중55000, 주말58000, 12월1일개장",
        "note": "2024-12-01 개장·2025-03-03 종료(뉴스 교차확인) · 주중 5.5만/주말 5.8만 재확인",
    },
    "GHS-24W-02": {
        "start": date(2024, 11, 16), "end": date(2024, 12, 31), "confidence": "확정",
        "sources": "localnaeil 652188(2024-12-05) · 더 델리 네이버예약 출처(픽업 11/16~12/31) · 얼리버드 ~12/8 15%",
        "keywords": "페스티브케이크, 얼리버드, 15할인, 픽업, 시크릿트리, 네이버예약, 텀블러",
        "note": "픽업 11/16~12/31 · 5종(트리12만·루돌프11만 등) · 얼리버드 15% ~12/8",
    },
    "GHS-24S-01": {
        "start": date(2024, 5, 24), "end": date(2024, 9, 30), "confidence": "확정",
        "sources": "insight.co.kr 469121(2024-05-20) · hotelrestaurant.co.kr idxno=13775 · TableCheck 2024 플랜",
        "keywords": "멕시칸스타일, 무제한드링크, 타코, 5월24일개장, 178000, 수영장뷰",
        "note": "5/24~9/30 수~일 19~21:30(월화 휴무) · 성인 178.000원(무제한 음료 포함 뉴스 확인)",
    },
    "GHS-24S-02": {
        "start": date(2024, 6, 1), "end": date(2024, 8, 31), "confidence": "추정",
        "sources": "망고철 6월 개시 패턴(AsiaN 2023) · 과거 시즌 8/31 종료(traveltimes 2021) · 망고 애프터누티티 8/31(ajunews 2024-06-07)",
    },
}


class WindowError(ValueError):
    """판매 기간 채우기 대상·형식이 어긋날 때."""


def collection_window(end: date, plus: int = WINDOW_PLUS_DAYS) -> date:
    """레지스트리 규약: 수집 창 끝 = 판매 종료 + 30일."""
    return end + timedelta(days=plus)


def fill_workbook(path: Path) -> list[tuple[str, str, str]]:
    workbook = load_workbook(path)
    updated: list[tuple[str, str, str]] = []
    try:
        product = workbook["B_상품"]
        blog = workbook["C_블로그"]
        prod_headers = {
            str(product.cell(1, col).value).strip(): col
            for col in range(1, product.max_column + 1) if product.cell(1, col).value
        }
        blog_headers = {
            str(blog.cell(1, col).value).strip(): col
            for col in range(1, blog.max_column + 1) if blog.cell(1, col).value
        }
        done = set()
        for row in range(3, product.max_row + 1):
            pid = product.cell(row, prod_headers["상품ID"]).value
            if pid not in SALE_WINDOWS:
                continue
            spec = SALE_WINDOWS[pid]
            product.cell(row, prod_headers["판매 시작일"]).value = spec["start"]
            product.cell(row, prod_headers["판매 종료일"]).value = spec["end"]
            product.cell(row, prod_headers["확인"]).value = f"{spec['confidence']} — 출처: {spec['sources']}"
            done.add(pid)
        missing = set(SALE_WINDOWS) - done
        if missing:
            raise WindowError(f"B_상품에서 대상 행을 찾지 못했습니다: {sorted(missing)}")
        for row in range(3, blog.max_row + 1):
            pid = blog.cell(row, blog_headers["상품ID"]).value
            if pid not in SALE_WINDOWS:
                continue
            spec = SALE_WINDOWS[pid]
            blog.cell(row, blog_headers["기간 창 시작(판매 시작일)"]).value = spec["start"]
            blog.cell(row, blog_headers["기간 창 끝(판매 종료 +30일)"]).value = collection_window(spec["end"])
            if spec.get("keywords") and "자주 함께 나온 키워드" in blog_headers:
                blog.cell(row, blog_headers["자주 함께 나온 키워드"]).value = spec["keywords"]
            if spec.get("note") and "비고" in blog_headers:
                blog.cell(row, blog_headers["비고"]).value = spec["note"]
            updated.append((str(pid), str(spec["start"]), str(collection_window(spec["end"]))))
        if len(updated) != len(SALE_WINDOWS):
            raise WindowError(f"C_블로그 창 채우기 불완료: {len(updated)}/{len(SALE_WINDOWS)}")
    finally:
        workbook.save(path)
        workbook.close()
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="GHS 2024 판매기간·수집창 채움(사본)")
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    args = parser.parse_args()
    if not args.workbook.exists():
        raise SystemExit(f"사본 워크북이 없습니다: {args.workbook}")
    for pid, start, end in fill_workbook(args.workbook):
        print(f"{pid}: 판매 시작 {start} → 수집 창 끝 {end}")
    print(f"완료 → {args.workbook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
