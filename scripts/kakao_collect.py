"""카카오(Daum) 블로그 검색 보조 수집 — 시트4 판정 초안행 생성.

언급량(총 건수)은 전체 기간 기준이라 시트3 언급량란에는 쓰지 않는다.
표본 문구(제목·첫 화면)를 그대로 근거로 기입해 사람 판정 비용을 줄인다.
"""
from __future__ import annotations

import html as html_mod
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

KAKAO_SEARCH_URL = "https://dapi.kakao.com/v2/search/blog.json"


class KkError(RuntimeError):
    """카카오 API 실행 실패."""


def _http_json(query: str, page: int, size: int) -> dict[str, Any]:
    key = os.environ.get("KAKAO_REST_API_KEY", "")
    if not key:
        raise KkError("KAKAO_REST_API_KEY 미설정 — developers.kakao.com에서 즉시 발급 가능")
    request = urllib.request.Request(
        f"{KAKAO_SEARCH_URL}?query={urllib.parse.quote(query)}&page={page}&size={size}",
        headers={"Authorization": f"KakaoAK {key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _strip_bold(text: str) -> str:
    return html_mod.unescape(re.sub(r"<[^>]+>", "", text))


def kakao_blog_samples(
    query: str, start: date, end: date, target: int = 30, max_pages: int = 8, size: int = 50,
) -> tuple[int, list[dict[str, str]], int]:
    """카카오 블로그 검색으로 수집 창 안의 포스트를 모은다.

    반환 (검색 문서수, 창 안 문서 목록, 스캔 문서수). 언급량(총 건수)은
    전체 기간 기준이라 시트3 언급량란에는 쓰지 않는다(정의 혼용 방지).
    """
    docs_total = 0
    in_window: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for page in range(1, max_pages + 1):
        data = _http_json(query, page, size)
        docs_total = int(data["meta"]["total_count"])
        for doc in data["documents"]:
            url = doc["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            posted = date.fromisoformat(str(doc["datetime"])[:10])
            if not start <= posted <= end:
                continue
            in_window.append({
                "url": url,
                "title": _strip_bold(doc["title"]),
                "contents": _strip_bold(doc["contents"]),
                "posted": posted.isoformat(),
                "blogname": doc.get("blogname", ""),
            })
        if len(in_window) >= target or data["meta"].get("is_end", True):
            break
        time.sleep(0.3)
    return docs_total, in_window[:target], len(seen_urls)


def kakao_fill_sheet4(
    workbook_path: Path, rows: list[dict[str, str]], sample_classify: Any, target_per_product: int = 30,
    max_pages: int = 8,
) -> list[tuple[str, int, int]]:
    """카카오 블로그 표본으로 시트4를 목표 편수까지 채운다 (네이버 요청 0회).

    sample_classify(text, hotel) — fetch_web_mentions.classify를 주입받아 분류한다.
    """
    workbook = load_workbook(workbook_path)
    sheet = workbook["C_협찬판정"]
    headers = {str(sheet.cell(1, col).value).strip(): col
               for col in range(1, sheet.max_column + 1) if sheet.cell(1, col).value}
    state: dict[str, dict[str, Any]] = {}
    for row in range(3, sheet.max_row + 1):
        pid = str(sheet.cell(row, headers["상품ID"]).value)
        link = str(sheet.cell(row, headers["글 링크"]).value)
        entry = state.setdefault(pid, {"rows": 0, "links": set()})
        entry["rows"] += 1
        entry["links"].add(link)
    summary: list[tuple[str, int, int]] = []
    for row in rows:
        pid = str(row["product_id"])
        state.setdefault(pid, {"rows": 0, "links": set()})
        start = date.fromisoformat(row["window_start"])
        end = date.fromisoformat(row["window_end"])
        docs_total, docs, _scanned = kakao_blog_samples(
            row["query"], start, end, target=max(target_per_product - state[pid]["rows"], 0),
            max_pages=max_pages,
        )
        added = 0
        section_span = max((end - start).days / 3.0, 1)
        for doc in docs:
            if state[pid]["rows"] >= target_per_product or doc["url"] in state[pid]["links"]:
                continue
            posted = date.fromisoformat(doc["posted"])
            label, _ev = sample_classify(
                f"{doc['title']} {doc['contents']}", row["hotel"])
            new_row = sheet.max_row + 1
            sheet.cell(new_row, headers["상품ID"]).value = pid
            sheet.cell(new_row, headers["호텔"]).value = row["hotel"]
            sheet.cell(new_row, headers["구간"]).value = min(2, int((posted - start).days / section_span)) + 1
            sheet.cell(new_row, headers["작성일"]).value = doc["posted"]
            sheet.cell(new_row, headers["글 링크"]).value = doc["url"]
            sheet.cell(new_row, headers["분류"]).value = label
            sheet.cell(new_row, headers["근거(제목·첫 화면 문구)"]).value =                 f"[KAKAO초안] 제목: {doc['title'][:60]} / 첫 화면: {doc['contents'][:70]}"
            state[pid]["links"].add(doc["url"])
            state[pid]["rows"] += 1
            added += 1
        summary.append((pid, added, docs_total))
    workbook.save(workbook_path)
    workbook.close()
    return summary
