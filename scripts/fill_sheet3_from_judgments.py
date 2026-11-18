"""시트4 판정 결과 → 시트3 집계 칸 반영 (판정 편수·분류 카운트·협찬 비율·수집일).

협찬 비율 정의: (협찬 + 기업 게시) / (판정 편수 − 판정 보류) — 기업 게시도 호텔 발 의료화
콘텐츠로 간주. 정의는 비고에 명시한다.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

DEFAULT_WORKBOOK = Path("/Users/robotjang/Downloads/호텔조사_정리_GHS2024_v1.xlsx")
CLASS_COLUMNS = ("협찬", "기업 게시", "자비 표시", "표시 없음", "판정 보류")


def collect_classifications(sheet: Any, headers: dict[str, int]) -> dict[str, dict[str, int]]:
    tally: dict[str, dict[str, int]] = {}
    for row in range(3, sheet.max_row + 1):
        pid = sheet.cell(row, headers["상품ID"]).value
        if pid in (None, ""):
            continue
        label = str(sheet.cell(row, headers["분류"]).value).strip()
        entry = tally.setdefault(str(pid), {
            "협찬": 0, "기업 게시": 0, "자비 표시": 0, "표시 없음": 0, "판정 보류": 0,
        })
        if label in entry:
            entry[label] += 1
    return tally


def fill_sheet3(workbook_path: Path, hotel_prefix: str = "GHS") -> list[tuple[str, int, int, str]]:
    workbook = load_workbook(workbook_path)
    sheet3 = workbook["C_블로그"]
    sheet4 = workbook["C_협찬판정"]
    headers = {str(sheet3.cell(1, col).value).strip(): col
               for col in range(1, sheet3.max_column + 1) if sheet3.cell(1, col).value}
    sheet4_headers = {str(sheet4.cell(1, col).value).strip(): col
                      for col in range(1, sheet4.max_column + 1) if sheet4.cell(1, col).value}
    tally = collect_classifications(sheet4, sheet4_headers)
    updated: list[tuple[str, int, int, str]] = []
    stamp = f"[카카오 집계 {datetime.now():%Y-%m-%d}] 협찬비율=(협찬+기업게시)/유지분"
    for row in range(3, sheet3.max_row + 1):
        pid = str(sheet3.cell(row, headers["상품ID"]).value)
        if not pid.startswith(hotel_prefix) or pid not in tally:
            continue
        counts = tally[pid]
        judged = sum(counts[k] for k in CLASS_COLUMNS)
        denominator = judged - counts["판정 보류"]
        ratio = (counts["협찬"] + counts["기업 게시"]) / denominator if denominator > 0 else None
        sheet3.cell(row, headers["판정 편수"]).value = judged
        for name in CLASS_COLUMNS:
            if name in headers:
                sheet3.cell(row, headers[name]).value = counts[name]
        if ratio is not None:
            sheet3.cell(row, headers["협찬 비율"]).value = round(ratio, 3)
        sheet3.cell(row, headers["수집일"]).value = f"{datetime.now():%Y-%m-%d}"
        note = sheet3.cell(row, headers["비고"])
        note.value = f"{(note.value + ' · ') if note.value else ''}{stamp}"
        updated.append((pid, judged, int(round((ratio or 0) * 1000)), f"{counts['협찬']}/{counts['기업 게시']}/{counts['자비 표시']}/{counts['표시 없음']}/{counts['판정 보류']}"))
    workbook.save(workbook_path)
    workbook.close()
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="시트4 → 시트3 집계 반영")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--prefix", default="GHS")
    args = parser.parse_args()
    if not args.workbook.exists():
        raise SystemExit(f"워크북이 없습니다: {args.workbook}")
    for pid, judged, ratio_milli, classes in fill_sheet3(args.workbook, args.prefix):
        print(f"{pid}: 판정 {judged}편 · 협찬비율 {ratio_milli / 1000:.3f} · 협찬/기업/자비/표시/보류 = {classes}")
    print(f"완료 → {args.workbook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
