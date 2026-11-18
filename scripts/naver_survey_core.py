"""Core types and Naver blog collection logic (부록 A 시트3·4 수집기).

Pure domain layer: collection, parsing, segmentation, windowed counting.
I/O (registry, sheets, counts cache) lives in naver_survey_io.py; CLI in fetch_naver_mentions.py.
"""
from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import NewType, Protocol, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ProductId = NewType("ProductId", str)

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class Segment(StrEnum):
    """판매 구간 라벨 — 시트4 '구간' 컬럼 값(1/2/3)과 동일한 문자열."""

    OPEN = "1"
    PEAK = "2"
    TAIL = "3"


@dataclass(frozen=True, slots=True)
class Product:
    product_id: ProductId
    hotel: str
    query: str
    hotel_query: str
    window_start: date
    window_end: date
    seg2_start: date
    seg3_start: date
    keywords: str
    note: str


@dataclass(frozen=True, slots=True)
class ParseError(Exception):
    field: str
    detail: str


@dataclass(frozen=True, slots=True)
class NaverApiError(Exception):
    status: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class MissingCredentialsError(Exception):
    env_keys: tuple[str, ...]


class BlogItem(TypedDict):
    title: str
    link: str
    postdate: date
    description: str
    blogger: str


class BlogResponse(TypedDict):
    total: int
    items: list[BlogItem]


class BlogSearchClient(Protocol):
    def search(self, query: str, start: int, sort: str) -> BlogResponse: ...


@dataclass(frozen=True, slots=True)
class WindowCount:
    in_window: int
    total_estimate: int
    capped: bool


@dataclass(frozen=True, slots=True)
class MentionStats:
    product: WindowCount
    hotel: WindowCount


@dataclass(frozen=True, slots=True)
class SamplePost:
    postdate: date
    link: str
    title: str
    summary: str
    blogger: str


@dataclass(frozen=True, slots=True)
class FillPlan:
    """자동기입 게이트 — 표본이 min_sample 이상인 상품만 시트에 기입한다."""

    ready: tuple[ProductId, ...]
    pending: tuple[tuple[ProductId, int], ...]


_TAG_RE = re.compile(r"<[^>]+>")
_POSTDATE_RE = re.compile(r"^\d{8}$")
_API_URL = "https://openapi.naver.com/v1/search/blog.json"
_DISPLAY = 100
_API_START_CAP = 1000
_SAMPLE_MAX_PAGES = 10


def segment_of(product: Product, postdate: date) -> Segment:
    if postdate >= product.seg3_start:
        return Segment.TAIL
    if postdate >= product.seg2_start:
        return Segment.PEAK
    return Segment.OPEN


def fill_plan(sample_counts: dict[ProductId, int], min_sample: int) -> FillPlan:
    ready = tuple(pid for pid, count in sample_counts.items() if count >= min_sample)
    pending = tuple((pid, count) for pid, count in sample_counts.items() if count < min_sample)
    return FillPlan(ready=ready, pending=pending)


def _strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text))


def _to_date(stamp: str) -> date:
    return date(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]))


def _optional_text(raw: dict[str, JsonValue], key: str) -> str:
    value = raw.get(key)
    return _strip_html(value) if isinstance(value, str) else ""


def _parse_item(raw: JsonValue) -> BlogItem:
    match raw:  # noqa: MATCH_OK — JsonValue는 부분 dict 패턴으로 소진 불가, 기본 분기는 typed ParseError(침묵 아님)
        case {"title": str() as title, "link": str() as link, "postdate": str() as postdate}:
            if _POSTDATE_RE.match(postdate) is None:
                raise ParseError(field="postdate", detail=f"expected YYYYMMDD, got {postdate!r}")
            return BlogItem(
                title=_strip_html(title),
                link=link,
                postdate=_to_date(postdate),
                description=_optional_text(raw, "description"),
                blogger=_optional_text(raw, "bloggername"),
            )
        case _:
            raise ParseError(field="item", detail="missing title/link/postdate keys")


def parse_response(payload: JsonValue) -> BlogResponse:
    match payload:  # noqa: MATCH_OK — JsonValue는 부분 dict 패턴으로 소진 불가, 기본 분기는 typed ParseError(침묵 아님)
        case {"total": int() as total, "items": list() as raw_items}:
            return BlogResponse(total=total, items=[_parse_item(raw) for raw in raw_items])
        case _:
            raise ParseError(field="response", detail=f"expected object with total/items, got {type(payload).__name__}")


class NaverBlogSearch:
    """openapi.naver.com blog.json 클라이언트 (display=100, start는 API 상한 1000)."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret

    @classmethod
    def from_env(cls) -> NaverBlogSearch:
        client_id = os.environ.get("NAVER_CLIENT_ID")
        client_secret = os.environ.get("NAVER_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise MissingCredentialsError(env_keys=("NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"))
        return cls(client_id, client_secret)

    def search(self, query: str, start: int, sort: str) -> BlogResponse:
        params = urlencode({"query": query, "display": _DISPLAY, "start": start, "sort": sort})
        request = Request(
            f"{_API_URL}?{params}",
            headers={
                "X-Naver-Client-Id": self._client_id,
                "X-Naver-Client-Secret": self._client_secret,
                "User-Agent": "seasons-greeting-ghs/1.0",
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                payload: JsonValue = json.load(response)
        except HTTPError as error:
            raise NaverApiError(status=error.code, detail=f"Naver API HTTP {error.code}: {error.reason}") from None
        except URLError as error:
            raise NaverApiError(status=None, detail=f"Naver API connection failed: {error.reason}") from None
        return parse_response(payload)


def windowed_count(client: BlogSearchClient, query: str, window_start: date, window_end: date) -> WindowCount:
    """기간 창 내 상품 언급량. 링크 중복 제거, API 상한 1000건 초과 시 capped."""
    total_estimate = 0
    seen: set[str] = set()
    for start in range(1, _API_START_CAP + 1, _DISPLAY):
        response = client.search(query, start, "sim")
        if total_estimate == 0:
            total_estimate = response["total"]
        seen.update(item["link"] for item in response["items"] if window_start <= item["postdate"] <= window_end)
        if start + _DISPLAY - 1 >= response["total"]:
            break
    return WindowCount(in_window=len(seen), total_estimate=total_estimate, capped=total_estimate > _API_START_CAP)


def newest_samples(
    client: BlogSearchClient, query: str, window_start: date, window_end: date, limit: int
) -> list[SamplePost]:
    """최신순 순회로 기간 창 내 최신 포스트 limit편. 10페이지 한도 내에서 부족하면 그대로 반환."""
    seen: set[str] = set()
    posts: list[SamplePost] = []
    for page in range(_SAMPLE_MAX_PAGES):
        response = client.search(query, page * _DISPLAY + 1, "date")
        if not response["items"]:
            break
        for item in response["items"]:
            if item["link"] in seen or not window_start <= item["postdate"] <= window_end:
                continue
            seen.add(item["link"])
            posts.append(
                SamplePost(
                    postdate=item["postdate"],
                    link=item["link"],
                    title=item["title"],
                    summary=item["description"],
                    blogger=item["blogger"],
                )
            )
            if len(posts) >= limit:
                return posts
    return posts
