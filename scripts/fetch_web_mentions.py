"""웹 기반 멘션 수집기 — 네이버/카페/카카오 소스 구분 (선택적 임시 수집).

정책:
- 네이버: 공식 검색 API 키가 없는 임시 경로 — where=blog 페이지 순회로 고유 포스트를
  전수 수집(20건/페이지)하고, 개별 글은 모바일 페이지(공개)에서 제목·작성일·본문·태그를 읽어
  협찬/기업 게시/자비 표시 분류 초안을 남긴다. 언급량은 "순회 수집 건수" 정의로 기입하고
  원본 레지스트리 규약과의 차이를 비고에 명시한다.
- 카카오(다음 카페·블로그): KAKAO_REST_API_KEY 환경변수가 있을 때만 보조 수집(v2/search/cafe).
- 카페 직접 접근: 로그인 게이트 대부분 — 수집하지 않는다.

주의: HTML 크롤링은 이용약관 회색지대이며 마크업 변경에 취약하다. 논문 원천 수치는
NAVER 공식 API로 최종 검증하고, 이 수집기의 값은 임시 보간용으로만 병행 기입한다.
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from urllib.error import URLError
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
POST_PER_PAGE = 20
MAX_PAGES = 30
KAKAO_CAFE_URL = "https://dapi.kakao.com/v2/search/cafe.json"

POST_URL_RE = re.compile(r"blog\.naver\.com/([A-Za-z0-9_]+)/(\d{9,12})")
NEXT_PAGE_RE = re.compile(r"page=(\d+)")
DATE_RE = re.compile(r"(20\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?")
TAG_RE = re.compile(r"#([가-힣A-Za-z0-9_]{2,20})")

SUBSIDY_HINTS = ("협찬", "제공받", "제공을 받", "광고", "체험단", "리뷰어", "지원받", "제품을 받")
SELF_HINTS = ("자비", "내돈", "직접 결제", "본인 부담", "제 금액", "자비로")
CORP_HINTS = ("보도자료", "제공한", "매체 기사", "공식 블로그", "사진=그랜드")


class WebFetchError(RuntimeError):
    """HTML 수집 실패."""


@dataclass(frozen=True, slots=True)
class Sample:
    url: str
    title: str
    posted: str
    body: str
    tags: str
    classification: str
    evidence: str


def _http_get(url: str, timeout: int = 30, retries: int = 2) -> str:
    last = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="ignore")
        except (URLError, TimeoutError, OSError) as error:  # 429·네트워크·타임아웃 백오프 재시도
            last = error
            time.sleep(1.5 * (attempt + 1))
    raise WebFetchError(f"fetch 실패({url[:60]}…): {last}")


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", fragment)
    text = html_mod.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()


def walk_blog_posts(query: str, start: date, end: date, max_posts: int = 30) -> tuple[int, list[str]]:
    """블로그 전용 검색을 페이지 순회해 고유 포스트 URL을 수집한다. 반환 (순회 건수, URL 목록)."""
    nso = f"p%3Afrom{start:%Y%m%d}to{end:%Y%m%d}%2Ca%3Aall"
    seen: set[tuple[str, str]] = set()
    urls: list[str] = []
    for page in range(1, MAX_PAGES + 1):
        page_url = (
            "https://search.naver.com/search.naver?"
            f"nso={nso}&page={page}&qdt=0&query={urllib.parse.quote_plus(query)}"
            f"&sm=tab_pge&ssc=tab.ur.all&start={(page - 1) * POST_PER_PAGE + 1}"
        )
        page_html = _http_get(page_url)
        found = []
        for user, post in POST_URL_RE.findall(page_html):
            if (user, post) not in seen:
                seen.add((user, post))
                found.append(f"https://blog.naver.com/{user}/{post}")
        urls.extend(found)
        fresh = len(found) > 0
        if page * POST_PER_PAGE >= max_posts or len(found) == 0 or not fresh:
            break
        time.sleep(0.7)
    return len(urls), urls[:max_posts]


def fetch_post(url: str) -> Sample:
    mobile = url.replace("blog.naver.com", "m.blog.naver.com")
    page = _http_get(mobile)
    text = html_mod.unescape(re.sub(r"<[^>]+>", " ", page))
    date_match = DATE_RE.search(text)
    title_match = re.search(r"<title>([^<]+)</title>", page)
    tags = " ".join(f"#{tag}" for tag in dict.fromkeys(TAG_RE.findall(text)))
    return Sample(
        url=url,
        title=strip_tags(title_match.group(1)) if title_match else "",
        posted=f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}" if date_match else "",
        body=(text[:1200]).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"),
        tags=tags,
        classification="판정 보류",
        evidence="",
    )


def classify(sample: Sample, hotel: str) -> tuple[str, str]:
    haystack = f"{sample.title} {sample.body[:600]}"
    if any(hint in haystack for hint in SUBSIDY_HINTS):
        return "협찬", "제목·본문 표시 키워드 발견"
    if any(hint in haystack for hint in SELF_HINTS):
        return "자비 표시", "본문 자비·내돈 표현"
    if "보도자료" in haystack or "사진=" in sample.body:
        return "기업 게시", "보도자료·제공 이미지 기재"
    return "표시 없음", f"'{hotel}' 관련 표시 문구 미발견"


def kakao_cafe_search(query: str, start: date, end: date) -> dict[str, Any]:
    key = os.environ.get("KAKAO_REST_API_KEY", "")
    if not key:
        return {"skipped": True, "reason": "KAKAO_REST_API_KEY 미설정"}
    request = urllib.request.Request(
        f"{KAKAO_CAFE_URL}?query={urllib.parse.quote(query)}&page=1&size=10",
        headers={"Authorization": f"KakaoAK {key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def read_registry(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    import csv
    return list(csv.DictReader(lines))


def collect_product(row: dict[str, str], sample_per_product: int) -> dict[str, Any]:
    query = row["query"]
    start = date.fromisoformat(row["window_start"])
    end = date.fromisoformat(row["window_end"])
    total, urls = walk_blog_posts(query, start, end, max_posts=sample_per_product)
    hotel = row["hotel"]
    samples: list[Sample] = []
    for url in urls:
        try:
            sample = fetch_post(url)
            label, evidence = classify(sample, hotel)
            samples.append(Sample(
                url=url, title=sample.title, posted=sample.posted, body=sample.body[:300],
                tags=sample.tags, classification=label, evidence=evidence,
            ))
        except (WebFetchError, URLError, TimeoutError, OSError) as error:  # 개별 글 실패는 건너뛰고 진행
            samples.append(Sample(url=url, title="", posted="", body=f"fetch 실패: {error}",
                                  tags="", classification="판정 보류", evidence=str(error)))
        time.sleep(0.6)
    return {
        "product_id": row["product_id"], "hotel": hotel, "query": query,
        "window": (start, end), "total": total, "samples": samples,
        "kakao": kakao_cafe_search(query, start, end),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description="웹 기반 멘션 수집(임시) — 네이버/카카오")
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / "data" / "grand_hyatt_survey" / "products_naver.csv")
    parser.add_argument("--workbook", type=Path, default=Path("/Users/robotjang/Downloads/호텔조사_정리_GHS2024_v1.xlsx"))
    parser.add_argument("--only", default="GHS-24", help="쉼표 구분 상품ID 접두어 필터")
    parser.add_argument("--sample-per-product", type=int, default=30)
    parser.add_argument("--write-sheets", action="store_true", help="사본 워크북 시트3·4에 실제 기입")
    args = parser.parse_args()
    rows = [row for row in read_registry(args.registry)
            if any(row["product_id"].startswith(prefix) for prefix in args.only.split(","))]
    if not rows:
        raise SystemExit(f"대상 행이 없습니다: {args.only}")
    results = [collect_product(row, args.sample_per_product) for row in rows]
    for result in results:
        labels = {}
        for sample in result["samples"]:
            labels[sample.classification] = labels.get(sample.classification, 0) + 1
        kakao = result["kakao"]
        kakao_note = "미설정→스킵" if kakao.get("skipped") else f"다음카페 total {kakao.get('meta', {}).get('total_count')}"
        print(f"{result['product_id']} '{result['query']}' 순회 {result['total']}건 · 표본 {len(result['samples'])}편 "
            f"· {labels} · 카카오:{kakao_note}")
    if args.write_sheets:
        judge = fill_sheet4(args.workbook, results, args.sample_per_product)
        mentions = fill_mentions(args.workbook, results)
        print(f"기입 완료 → 시트4 {judge}행 · 시트3 언급량 {mentions}칸")
    return 0


if __name__ == "__main__":
    sys.exit(main())
