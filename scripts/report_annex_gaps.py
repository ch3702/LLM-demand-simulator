"""부록 A 공백 리포트 — 호텔·상품ID별 시트3~6 정량 칸 충족 현황.

레지스트리(수집 대상)와 워크북(C_블로그·C_협찬판정·D_게시물·D_댓글)을 대조해
언급량·협찬비율(30편 표본)·게시물·댓글 칸의 누락 목록을 팀 분담용으로 출력한다.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = Path("/Users/robotjang/Downloads/호텔조사_정리 (2).xlsx")
REGISTRY_PATHS = (
    REPO_ROOT / "data" / "grand_hyatt_survey" / "products_naver.csv",
    REPO_ROOT / "data" / "walkerhill_survey" / "products_naver.csv",
)
JUDGMENT_TARGET = 30


@dataclass(frozen=True, slots=True)
class GapRow:
    product_id: str
    hotel: str
    mentions: int | None
    judged: int
    subsidy: int
    corporate: int
    self_funded: int
    judged_rows: int
    has_post: bool
    has_comments: bool


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def load_registry(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        lines = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            continue
        rows.extend(row for row in csv.DictReader(lines) if row.get("product_id"))
    return rows


def _sheet_records(sheet, key_header: str = "상품ID") -> list[dict[str, Any]]:
    headers = {}
    for column in range(1, sheet.max_column + 1):
        name = sheet.cell(1, column).value
        if isinstance(name, str) and name.strip():
            headers[name.strip()] = column
    if key_header not in headers:
        return []
    key_col = headers[key_header]
    records = []
    for row in range(3, sheet.max_row + 1):
        value = sheet.cell(row, key_col).value
        if value in (None, ""):
            continue
        records.append({
            name: sheet.cell(row, index).value
            for name, index in headers.items()
        })
    return records


def build_gap_rows(
    registry: list[Mapping[str, str]],
    sheets: Mapping[str, list[Mapping[str, Any]]],
) -> list[GapRow]:
    blog = {str(item.get("상품ID", "")).strip(): item for item in sheets.get("C_블로그", [])}
    judged: dict[str, int] = {}
    classes: dict[str, dict[str, int]] = {}
    for item in sheets.get("C_협찬판정", []):
        pid = str(item.get("상품ID", "")).strip()
        if not pid:
            continue
        judged[pid] = judged.get(pid, 0) + 1
        bucket = classes.setdefault(pid, {"협찬": 0, "기업 게시": 0, "자비 표시": 0})
        label = str(item.get("분류", "")).strip()
        if label in bucket:
            bucket[label] += 1
    posts = {str(item.get("상품ID", "")).strip() for item in sheets.get("D_게시물", [])}
    comments = {str(item.get("상품ID", "")).strip() for item in sheets.get("D_댓글", [])}

    rows: list[GapRow] = []
    for entry in registry:
        pid = str(entry["product_id"]).strip()
        item = blog.get(pid, {})
        bucket = classes.get(pid, {"협찬": 0, "기업 게시": 0, "자비 표시": 0})
        rows.append(GapRow(
            product_id=pid,
            hotel=str(entry.get("hotel", "")).strip(),
            mentions=_int(item.get("상품 언급량")),
            judged=_int(item.get("판정 편수")) or 0,
            subsidy=bucket["협찬"],
            corporate=bucket["기업 게시"],
            self_funded=bucket["자비 표시"],
            judged_rows=judged.get(pid, 0),
            has_post=pid in posts,
            has_comments=pid in comments,
        ))
    return rows


def status_of(row: GapRow) -> str:
    complete = (
        row.mentions is not None
        and row.judged >= JUDGMENT_TARGET
        and (row.subsidy + row.corporate + row.self_funded) > 0
        and row.has_post
        and row.has_comments
    )
    if complete:
        return "완료"
    if row.mentions is None and row.judged == 0 and not row.has_post:
        return "누락"
    return "부분"


def render_markdown(rows: list[GapRow]) -> str:
    lines = [
        "# 부록 A 공백 리포트 (시트3~6 정량 칸)",
        "",
        f"- 대상 상품: {len(rows)}개",
        f"- 완료: {sum(1 for row in rows if status_of(row) == '완료')} · "
        f"부분: {sum(1 for row in rows if status_of(row) == '부분')} · "
        f"누락: {sum(1 for row in rows if status_of(row) == '누락')}",
        f"- 협찬 판정 목표: 상품당 {JUDGMENT_TARGET}편 표본",
        "",
    ]
    for hotel in sorted({row.hotel for row in rows}):
        subset = [row for row in rows if row.hotel == hotel]
        lines += [
            f"## {hotel} ({len(subset)}개)",
            "",
            "| 상품ID | 언급량 | 판정 편수 | 협찬/기업/자비 | D_게시물 | D_댓글 | 상태 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in subset:
            mentions = "—" if row.mentions is None else f"{row.mentions:,}"
            verdict = f"{row.subsidy}/{row.corporate}/{row.self_funded}"
            lines.append(
                f"| {row.product_id} | {mentions} | {row.judged}/{JUDGMENT_TARGET} | {verdict} "
                f"| {'✓' if row.has_post else '—'} | {'✓' if row.has_comments else '—'} | {status_of(row)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="부록 A 공백 리포트 생성")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "docs" / "부록A_공백리포트.md")
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "outputs" / "annex_gap_report.csv")
    args = parser.parse_args()
    if not args.workbook.exists():
        raise SystemExit(f"워크북이 없습니다: {args.workbook}")
    workbook = load_workbook(args.workbook, data_only=True)
    try:
        sheets = {name: _sheet_records(workbook[name]) for name in ("C_블로그", "C_협찬판정", "D_게시물", "D_댓글")}
    finally:
        workbook.close()
    rows = build_gap_rows(load_registry(REGISTRY_PATHS), sheets)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(rows), encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product_id", "hotel", "mentions", "judged", "subsidy", "corporate", "self_funded", "has_post", "has_comments", "status"])
        for row in rows:
            writer.writerow([row.product_id, row.hotel, row.mentions, row.judged, row.subsidy,
                             row.corporate, row.self_funded, row.has_post, row.has_comments, status_of(row)])
    complete = sum(1 for row in rows if status_of(row) == "완료")
    print(f"공백 리포트 → {args.out_md} | 대상 {len(rows)} · 완료 {complete}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
