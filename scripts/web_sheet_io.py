"""웹 수집 결과의 시트 기입 로직 (시트3 언급량 · 시트4 판정 초안행)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

def fill_sheet4(workbook_path: Path, results: list[dict[str, Any]], max_samples: int) -> int:
    workbook = load_workbook(workbook_path)
    sheet = workbook["C_협찬판정"]
    headers = {str(sheet.cell(1, col).value).strip(): col for col in range(1, sheet.max_column + 1) if sheet.cell(1, col).value}
    existing = {str(sheet.cell(row, headers["글 링크"]).value) for row in range(3, sheet.max_row + 1)}
    added = 0
    for result in results:
        window = result["window"]
        section_span = (window[1] - window[0]).days / 3.0
        for sample in result["samples"][:max_samples]:
            if sample.url in existing or not sample.posted:
                continue
            posted = datetime.fromisoformat(sample.posted).date() if sample.posted else window[0]
            section = min(2, int((posted - window[0]).days / max(section_span, 1)))
            row = sheet.max_row + 1
            sheet.cell(row, headers["상품ID"]).value = result["product_id"]
            sheet.cell(row, headers["호텔"]).value = result["hotel"]
            sheet.cell(row, headers["구간"]).value = section + 1
            sheet.cell(row, headers["작성일"]).value = sample.posted
            sheet.cell(row, headers["글 링크"]).value = sample.url
            sheet.cell(row, headers["분류"]).value = sample.classification
            sheet.cell(row, headers["근거(제목·첫 화면 문구)"]).value = f"[HTML초안] 제목: {sample.title[:80]}"
            existing.add(sample.url)
            added += 1
    workbook.save(workbook_path)
    workbook.close()
    return added


def fill_mentions(workbook_path: Path, results: list[dict[str, Any]]) -> int:
    """시트3 언급량이 비어 있거나 [API 집계 필요]인 칸만 순회 수집 건수로 채운다."""
    workbook = load_workbook(workbook_path)
    sheet = workbook["C_블로그"]
    headers = {str(sheet.cell(1, col).value).strip(): col for col in range(1, sheet.max_column + 1) if sheet.cell(1, col).value}
    filled = 0
    for row in range(3, sheet.max_row + 1):
        pid = str(sheet.cell(row, headers["상품ID"]).value)
        result = next((item for item in results if item["product_id"] == pid), None)
        if result is None:
            continue
        cell = sheet.cell(row, headers["상품 언급량"])
        if cell.value in (None, "") or "API" in str(cell.value):
            cell.value = result["total"]
            stamp = f"[HTML 순회수집 {datetime.now():%Y-%m-%d}] {result['total']}건 (20건/페이지 순회정의, API와 정의 상이)"
            note = sheet.cell(row, headers["비고"])
            note.value = f"{note.value + ' · ' if note.value else ''}{stamp}"
            filled += 1
    workbook.save(workbook_path)
    workbook.close()
    return filled


def kakao_fill_sheet4(workbook_path: Path, rows: list[dict[str, str]], target_per_product: int) -> list[tuple[str, int, int]]:
    """카카오 블로그 표본으로 시트4를 30편 목표까지 채운다. (네이버 요청 0회)"""
    workbook = load_workbook(workbook_path)
    sheet = workbook["C_협찬판정"]
    blog_sheet = workbook["C_블로그"]
    headers = {str(sheet.cell(1, col).value).strip(): col for col in range(1, sheet.max_column + 1) if sheet.cell(1, col).value}
    blog_headers = {str(blog_sheet.cell(1, col).value).strip(): col for col in range(1, blog_sheet.max_column + 1) if blog_sheet.cell(1, col).value}
    by_pid: dict[str, dict] = {}
    for row in range(3, sheet.max_row + 1):
        pid = str(sheet.cell(row, headers["상품ID"]).value)
        link = str(sheet.cell(row, headers["글 링크"]).value)
        by_pid.setdefault(pid, {"rows": 0, "links": set()})
        by_pid[pid]["rows"] += 1
        by_pid[pid]["links"].add(link)
    summary: list[tuple[str, int, int]] = []
    for row in rows:
        pid = row["product_id"]
        state = by_pid.setdefault(pid, {"rows": 0, "links": set()})
        start = date.fromisoformat(row["window_start"])
        end = date.fromisoformat(row["window_end"])
        docs_total, docs, _pages = kakao_blog_samples(
            row["query"], start, end, target=max(target_per_product - state["rows"], 0),
        )
        added = 0
        section_span = max((end - start).days / 3.0, 1)
        for doc in docs:
            posted = date.fromisoformat(doc["posted"])
            label, evidence = classify(Sample(url=doc["url"], title=doc["title"], posted=doc["posted"],
                                              body=doc["contents"], tags="", classification="", evidence=""), row["hotel"])
            if state["rows"] >= target_per_product or doc["url"] in state["links"]:
                continue
            new_row = sheet.max_row + 1
            sheet.cell(new_row, headers["상품ID"]).value = pid
            sheet.cell(new_row, headers["호텔"]).value = row["hotel"]
            sheet.cell(new_row, headers["구간"]).value = min(2, int((posted - start).days / section_span)) + 1
            sheet.cell(new_row, headers["작성일"]).value = doc["posted"]
            sheet.cell(new_row, headers["글 링크"]).value = doc["url"]
            sheet.cell(new_row, headers["분류"]).value = label
            sheet.cell(new_row, headers["근거(제목·첫 화면 문구)"]).value =                 f"[KAKAO초안] 제목: {doc['title'][:60]} / 첫 화면: {doc['contents'][:70]}"
            state["links"].add(doc["url"])
            state["rows"] += 1
            added += 1
        summary.append((pid, added, docs_total))
    workbook.save(workbook_path)
    workbook.close()
    return summary


