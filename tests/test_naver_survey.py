"""Naver survey collector regression tests (부록 A 시트3-4 수집기).

Run: uv run --no-project --with pytest pytest tests/test_naver_survey.py -q
"""
from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from naver_survey_core import (
    BlogItem,
    BlogResponse,
    JsonValue,
    MentionStats,
    ParseError,
    Product,
    ProductId,
    SamplePost,
    Segment,
    WindowCount,
    fill_plan,
    newest_samples,
    parse_response,
    segment_of,
    windowed_count,
)
from naver_survey_io import (
    LabelTally,
    load_counts,
    load_registry,
    read_sheet4,
    save_counts,
    tally_by_product,
    write_sheet3,
    write_sheet4,
)

WINDOW = (date(2025, 12, 5), date(2026, 4, 1))


def _product(
    pid: str = "GHS-25W-02",
    seg2: date = date(2026, 1, 1),
    seg3: date = date(2026, 2, 1),
    window: tuple[date, date] = WINDOW,
) -> Product:
    return Product(
        product_id=ProductId(pid),
        hotel="그랜드 하얏트 서울",
        query=f"그랜드 하얏트 서울 이벤트-{pid}",
        hotel_query="그랜드 하얏트 서울",
        window_start=window[0],
        window_end=window[1],
        seg2_start=seg2,
        seg3_start=seg3,
        keywords="키워드",
        note="비고",
    )


def _item(index: int, postdate: date, link: str | None = None) -> BlogItem:
    return BlogItem(
        title=f"<b>후기</b>{index}",
        link=link or f"https://blog/{index}",
        postdate=postdate,
        description="내돈내산 스니펫",
        blogger=f"b{index}",
    )


class FakeBlogClient:
    """인메모리 가짜 클라이언트 — 고정 게시물 목록에서 페이지를 슬라이스."""

    def __init__(self, posts: list[BlogItem], total: int | None = None) -> None:
        self._posts = posts
        self._total = len(posts) if total is None else total

    def search(self, query: str, start: int, sort: str) -> BlogResponse:
        ordered = self._posts
        if sort == "date":
            ordered = sorted(self._posts, key=lambda item: item["postdate"], reverse=True)
        offset = start - 1
        return BlogResponse(total=self._total, items=ordered[offset : offset + 100])


def test_segment_matches_appendix_hand_rows_when_dates_given():
    products = {
        "GHS-25W-02": (_product(), date(2025, 12, 20), Segment.OPEN),
        "GHS-25W-02-tail": (_product(), date(2026, 2, 17), Segment.TAIL),
        "GHS-25W-03": (_product("GHS-25W-03", date(2025, 12, 1), date(2026, 1, 1), (date(2025, 11, 14), date(2026, 1, 30))), date(2025, 11, 19), Segment.OPEN),
        "GHS-25W-04": (_product("GHS-25W-04", date(2025, 12, 1), date(2026, 1, 1), (date(2025, 11, 20), date(2026, 1, 10))), date(2025, 12, 29), Segment.PEAK),
        "GHS-26S-02": (_product("GHS-26S-02", date(2026, 7, 1), date(2026, 9, 1), (date(2026, 5, 2), date(2026, 10, 27))), date(2026, 5, 17), Segment.OPEN),
        "GHS-26S-02-tail": (_product("GHS-26S-02", date(2026, 7, 1), date(2026, 9, 1), (date(2026, 5, 2), date(2026, 10, 27))), date(2026, 9, 23), Segment.TAIL),
    }
    for name, (product, post_date, expected) in products.items():
        assert segment_of(product, post_date) == expected, name


def test_windowed_count_when_posts_inside_and_outside_window():
    posts = [
        _item(1, date(2025, 12, 10)),
        _item(2, date(2026, 1, 5)),
        _item(3, date(2024, 1, 1)),
        _item(4, date(2027, 1, 1)),
        _item(1, date(2025, 12, 11), link="https://blog/1"),
    ]
    stats = windowed_count(FakeBlogClient(posts), "q", *WINDOW)
    assert stats == WindowCount(in_window=2, total_estimate=5, capped=False)


def test_windowed_count_when_pagination_cap_reached():
    posts = [_item(i, date(2026, 1, 10)) for i in range(1000)]
    stats = windowed_count(FakeBlogClient(posts, total=2500), "q", *WINDOW)
    assert stats.capped is True
    assert stats.in_window == 1000


def test_newest_samples_when_posts_predate_window():
    posts = [_item(i, date(2026, 2, 1 + i)) for i in range(3)]
    posts += [_item(10, date(2025, 1, 1)), _item(11, date(2024, 1, 1))]
    sampled = newest_samples(FakeBlogClient(posts), "q", WINDOW[0], WINDOW[1], limit=30)
    assert [post.link for post in sampled] == ["https://blog/2", "https://blog/1", "https://blog/0"]


def test_newest_samples_when_limit_reached_before_window_exhausted():
    posts = [_item(i, date(2026, 1, 1) + timedelta(days=i)) for i in range(40)]
    sampled = newest_samples(FakeBlogClient(posts), "q", WINDOW[0], WINDOW[1], limit=30)
    assert len(sampled) == 30
    assert sampled[0].link == "https://blog/39"


def test_fill_plan_when_mixed_sample_counts():
    plan = fill_plan({ProductId("A"): 30, ProductId("B"): 12, ProductId("C"): 30}, 30)
    assert plan.ready == (ProductId("A"), ProductId("C"))
    assert plan.pending == ((ProductId("B"), 12),)


def test_parse_response_when_naver_html_tags_present():
    payload: JsonValue = {
        "total": 2,
        "items": [
            {"title": "<b>하얏트</b> 아이스링크", "link": "https://blog/1", "postdate": "20251220", "description": "내돈내산 &quot;후기&quot;", "bloggername": "블로거"},
            {"title": "케이크", "link": "https://blog/2", "postdate": "20251221", "description": "", "bloggername": ""},
        ],
    }
    response = parse_response(payload)
    assert response["total"] == 2
    assert response["items"][0]["title"] == "하얏트 아이스링크"
    assert response["items"][0]["postdate"] == date(2025, 12, 20)


def test_parse_response_when_postdate_malformed():
    payload: JsonValue = {"total": 1, "items": [{"title": "t", "link": "l", "postdate": "2025-12-20", "description": "", "bloggername": "b"}]}
    with pytest.raises(ParseError):
        parse_response(payload)


def test_tally_by_product_when_alias_used():
    judgments = [(ProductId("P1"), "협찬"), (ProductId("P1"), "자비 구매 명시"), (ProductId("P1"), "판정 보류")]
    tallies = tally_by_product(judgments)
    tally = tallies[ProductId("P1")]
    assert tally == LabelTally(sponsored=1, corporate=0, self_paid=1, unlabeled=0, pending=1)
    assert tally.judged == 3
    assert tally.sponsored_ratio == pytest.approx(1 / 3)


def test_read_sheet4_when_label_unknown(tmp_path: Path):
    sheet = tmp_path / "sheet4.csv"
    sheet.write_text("상품ID,호텔,구간,작성일,글 링크,분류,근거\nP1,호텔,1,2026-01-01,https://blog/1,광고,근거\n", encoding="utf-8-sig")
    with pytest.raises(ParseError):
        read_sheet4(sheet)


def test_write_sheet4_when_prior_labels_exist(tmp_path: Path):
    product = _product()
    posts = [SamplePost(date(2026, 1, 5), "https://blog/1", "제목1", "요약1", "b1"), SamplePost(date(2026, 1, 6), "https://blog/2", "제목2", "요약2", "b2")]
    sheet = tmp_path / "sheet4.csv"
    write_sheet4(sheet, [(product, posts)], {}, "2026-11-17")
    first = {row.link: row.label for row in read_sheet4(sheet)}
    assert first == {"https://blog/1": "판정 보류", "https://blog/2": "판정 보류"}
    relabeled = {link: ("협찬" if link == "https://blog/1" else label) for link, label in first.items()}
    write_sheet4(sheet, [(product, posts)], relabeled, "2026-11-17")
    second = {row.link: row.label for row in read_sheet4(sheet)}
    assert second == {"https://blog/1": "협찬", "https://blog/2": "판정 보류"}


def test_load_registry_when_active_and_selected_rows(tmp_path: Path):
    registry = tmp_path / "products.csv"
    registry.write_text(
        "product_id,hotel,query,hotel_query,window_start,window_end,seg2_start,seg3_start,keywords,note,active\n"
        "P1,H,Q,HQ,2025-12-05,2026-04-01,2026-01-01,2026-02-01,kw,note,1\n"
        "P2,H,Q,HQ,2025-11-14,2026-01-30,,,kw,note,0\n",
        encoding="utf-8-sig",
    )
    assert [str(p.product_id) for p in load_registry(registry)] == ["P1"]
    both = load_registry(registry, selected=frozenset({"P2"}))
    assert [str(p.product_id) for p in both] == ["P1", "P2"]
    # seg 미기록 행은 창 3등분 폴백
    p2 = both[1]
    assert p2.seg2_start == date(2025, 11, 14) + (date(2026, 1, 30) - date(2025, 11, 14)) // 3


def test_save_and_load_counts_roundtrip(tmp_path: Path):
    path = tmp_path / "counts.json"
    stats = {ProductId("P1"): MentionStats(product=WindowCount(137, 500, False), hotel=WindowCount(1500, 5000, True))}
    save_counts(path, stats, generated_with_mock=True)
    loaded, is_mock = load_counts(path)
    assert is_mock is True
    assert loaded[ProductId("P1")].hotel.capped is True
    assert loaded[ProductId("P1")].product.in_window == 137


def test_write_sheet3_when_cap_and_mock_notes(tmp_path: Path):
    product = _product()
    stats = {product.product_id: MentionStats(product=WindowCount(137, 500, False), hotel=WindowCount(1500, 5000, True))}
    sheet = tmp_path / "sheet3.csv"
    write_sheet3(sheet, [product], stats, {}, "2026-11-17", is_mock=True)
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][5] == "137" and rows[1][6] == "1500"
    assert rows[1][7] == "0.0913"
    assert "상한" in rows[1][17] and "[MOCK 데이터]" in rows[1][17]
