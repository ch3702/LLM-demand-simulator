"""Annex workbook integration tests (B_상품→레지스트리 동기화 + C_블로그·C_협찬판정 직접기입 + Gate D).

Run: uv run --no-project --with pytest --with openpyxl pytest tests/test_annex_sync.py -q
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from annex_xlsx import read_sheet4_rows, write_annex
from fetch_naver_mentions import gate_d_reminder
from naver_survey_io import (
    _SHEET3_HEADER,
    _SHEET4_HEADER,
    Sheet4Row,
    merge_sheet4_rows,
    read_registry_rows,
)
from naver_survey_core import Product, ProductId, SamplePost
from sync_registry_from_xlsx import sync_registry

PRODUCTS_HEADER = [
    "상품ID", "호텔", "연도", "상품명", "상품 유형", "제안 유형(판정 보류일 때)", "대표 상품",
    "판매 시작일", "판매 종료일", "대표 가격", "가격 범위", "구성", "타깃 고객", "판매 장소",
    "판매 상태", "출처", "확인", "워커힐 대응 상품ID(취합)", "비고",
]


def _placeholder_sheet3_row(pid: str, hotel: str) -> list[str]:
    row = [pid, hotel, "", "", ""] + ["[API 집계 필요]"] * 2 + ["[계산]"] + ["30"] + ["[표본 판정]"] * 5 + ["[계산]", "", "2026-11-17", ""]
    assert len(row) == 18
    return row


def _build_workbook(path: Path) -> None:
    wb = Workbook()
    b = wb.active
    b.title = "B_상품"
    b.append(PRODUCTS_HEADER)
    b.append(PRODUCTS_HEADER)
    b.append([
        "GHS-25W-02", "그랜드 하얏트 서울", 2025, "야외 아이스링크 이용권 (통합/단독)", "야외·체험", "",
        "대표", datetime(2025, 12, 5), datetime(2026, 3, 2), 58000, "16,000~58,000", "통합권(입장+대여)",
        "커플·기념일", "GL층", "종료", "공식 정책 PDF", "확인", "(취합 후 매핑)", "주중·주말 5.8만 단일화 운영",
    ])
    b.append([
        "SHL-25W-02", "서울신라", 2025, "패스트리 부티크 크리스마스 케이크", "케이크·디저트", "",
        "대표", datetime(2025, 11, 24), datetime(2025, 12, 31), 500000, "80,000~500,000", "트러플 케이크",
        "커플·기념일", "패스트리 부티크", "종료", "언론 보도", "확인", "(취합 후 매핑)", "하루 3개 한정",
    ])
    b.append([
        "GHS-25W-04", "그랜드 하얏트 서울", 2025, "더 테라스 연말 디너 뷔페 (창가석 패키지)", "뷔페", "",
        "대표", datetime(2025, 12, 1), datetime(2025, 12, 31), 199000, "180,000~495,000", "피크일 2부제",
        "커플·기념일", "더 테라스", "종료", "네이버 블로그", "확인", "(취합 후 매핑)", "C_블로그 스코프 외",
    ])
    c = wb.create_sheet("C_블로그")
    c.append(list(_SHEET3_HEADER))
    c.append(list(_SHEET3_HEADER))
    c.append([
        "GHS-25W-02", "그랜드 하얏트 서울", "그랜드 하얏트 서울 아이스링크", "2025-12-05", "2026-04-01",
        "[API 집계 필요]", "[API 집계 필요]", "[계산]", 30, "[표본 판정]", "[표본 판정]", "[표본 판정]",
        "[표본 판정]", "[표본 판정]", "[계산]", "정빙시간, 핫초코", "2026-11-17", "일 8부제 운영, 정빙 대기 다수",
    ])
    c.append(_placeholder_sheet3_row("SHL-25W-02", "서울신라"))
    j = wb.create_sheet("C_협찬판정")
    j.append(list(_SHEET4_HEADER))
    j.append(list(_SHEET4_HEADER))
    j.append(["GHS-25W-02", "그랜드 하얏트 서울", "1", "2025-12-20", "https://blog.example/x", "자비 표시", "제목: 아이스링크 후기"])
    j.append(["SHL-25W-02", "서울신라", "2", "2025-11-25", "https://blog.example/y", "표시 없음", "제목: 케이크 예약"])
    wb.save(path)


@pytest.fixture()
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "annex.xlsx"
    _build_workbook(path)
    return path


def test_sync_scopes_by_blog_sheet(workbook: Path, tmp_path: Path) -> None:
    out = tmp_path / "products_naver.csv"
    result = sync_registry(workbook, out, prefixes=("GHS", "SHL"), scope_sheet="C_블로그")
    rows = {row["product_id"]: row for row in read_registry_rows(out)}
    assert result.workbook_rows == 2
    assert set(rows) == {"GHS-25W-02", "SHL-25W-02"}  # GHS-25W-04는 C_블로그 스코프 외 → 배제
    assert rows["GHS-25W-02"]["query"] == "그랜드 하얏트 서울 아이스링크"
    assert rows["GHS-25W-02"]["keywords"] == "정빙시간, 핫초코"
    assert rows["GHS-25W-02"]["window_start"] == "2025-12-05"
    assert rows["GHS-25W-02"]["window_end"] == "2026-04-01"  # 종료 2026-03-02 +30일
    assert rows["GHS-25W-02"]["note"] == "일 8부제 운영, 정빙 대기 다수"
    assert rows["GHS-25W-02"]["active"] == "1"


def test_sync_prefix_filter_and_preserves_unknown_ids(workbook: Path, tmp_path: Path) -> None:
    out = tmp_path / "products_naver.csv"
    out.write_text(
        "# 기존 레지스트리\n"
        "product_id,hotel,query,hotel_query,window_start,window_end,seg2_start,seg3_start,keywords,note,active\n"
        "WH-26W-01,그랜드 워커힐 서울,그랜드 워커힐 휘바 욜루아,그랜드 워커힐 서울,2026-12-18,2027-01-24,,,\"산타, 패키지\",\"pkg 8859\",1\n",
        encoding="utf-8",
    )
    result = sync_registry(workbook, out, prefixes=("SHL",))
    rows = {row["product_id"]: row for row in read_registry_rows(out)}
    assert result.workbook_rows == 1
    assert set(rows) == {"WH-26W-01", "SHL-25W-02"}
    assert rows["WH-26W-01"]["note"] == "pkg 8859"  # 기존 행 원문 보존
    assert rows["SHL-25W-02"]["query"] == "신라호텔 패스트리 부티크 크리스마스 케이크"
    assert rows["SHL-25W-02"]["hotel_query"] == "신라호텔"
    assert rows["SHL-25W-02"]["window_end"] == "2026-01-30"  # 종료 2025-12-31 +30일
    assert "500,000원" in rows["SHL-25W-02"]["note"]


def test_sync_drops_parenthetical_in_query(workbook: Path, tmp_path: Path) -> None:
    out = tmp_path / "products_naver.csv"
    sync_registry(workbook, out, prefixes=("GHS",), scope_sheet=None)
    rows = {row["product_id"]: row for row in read_registry_rows(out)}
    assert rows["GHS-25W-02"]["query"] == "그랜드 하얏트 서울 야외 아이스링크 이용권"


def test_merge_sheet4_preserves_workbook_labels_and_dedupes(workbook: Path) -> None:
    prior = read_sheet4_rows(workbook)
    assert [row.label for row in prior] == ["자비 구매 명시", "경제적 이해관계 표시 없음"]  # 별칭 정규화
    product = Product(
        product_id=ProductId("GHS-25W-02"), hotel="그랜드 하얏트 서울",
        query="그랜드 하얏트 서울 아이스링크", hotel_query="그랜드 하얏트 서울",
        window_start=date(2025, 12, 5), window_end=date(2026, 4, 1),
        seg2_start=date(2026, 1, 13), seg3_start=date(2026, 2, 21),
        keywords="", note="",
    )
    entries = [(product, [
        SamplePost(
            postdate=date(2025, 12, 20), link="https://blog.example/x",
            title="아이스링크 후기", summary="내돈내산", blogger="b",
        ),
        SamplePost(
            postdate=date(2025, 12, 22), link="https://blog.example/z",
            title="신규 표본", summary="정보성", blogger="c",
        ),
    ])]
    merged = merge_sheet4_rows(entries, prior)
    assert merged[0][4] == "https://blog.example/y"  # 표본 밖 수기 행 캐리오버
    x_row = next(row for row in merged if row[4] == "https://blog.example/x")
    assert x_row[2] == "1" and x_row[5] == "자비 구매 명시"  # 구간 + 수기 라벨 승계
    z_row = next(row for row in merged if row[4] == "https://blog.example/z")
    assert z_row[5] == "판정 보류"
    dup_prior = list(prior) + [prior[0]]
    merged_dup = merge_sheet4_rows([], dup_prior)
    identities = [(row[0], row[2], row[3], row[4]) for row in merged_dup]
    assert len(identities) == len(set(identities))  # 재실행 누적 중복 제거
    placeholder_dups = list(prior) + [  # 자리표시 링크 반복(같은 링크, 다른 작성일)은 유실되면 안 된다
        Sheet4Row(product_id="GHS-25W-02", hotel="그랜드 하얏트 서울", segment="3",
                  postdate="2026-02-17", link="naver.com", label="경제적 이해관계 표시 없음", evidence="제목: 겨울 방문"),
        Sheet4Row(product_id="GHS-25W-03", hotel="그랜드 하얏트 서울", segment="1",
                  postdate="2025-11-19", link="naver.com", label="판정 보류", evidence="제목: 케이크 예약"),
    ]
    merged_placeholders = merge_sheet4_rows([], placeholder_dups)
    naver_rows = [row for row in merged_placeholders if row[4] == "naver.com"]
    assert {(row[0], row[3]) for row in naver_rows} == {("GHS-25W-02", "2026-02-17"), ("GHS-25W-03", "2025-11-19")}


def test_write_annex_fills_placeholders_and_backs_up(workbook: Path, tmp_path: Path) -> None:
    sheet3_rows = [[
        "GHS-25W-02", "그랜드 하얏트 서울", "그랜드 하얏트 서울 아이스링크", "2025-12-05", "2026-04-01",
        "59", "59", "1.0000", "30", "0", "0", "0", "0", "30", "0.0000",
        "정빙시간, 핫초코", "2026-11-17", "일 8부제 운영, 정빙 대기 다수 · [MOCK 데이터]",
    ]]
    sheet4_rows = [
        ["SHL-25W-02", "서울신라", "2", "2025-11-25", "https://blog.example/y", "경제적 이해관계 표시 없음", "제목: 케이크 예약"],
        ["GHS-25W-02", "그랜드 하얏트 서울", "1", "2025-12-20", "https://blog.example/x", "자비 구매 명시", "제목: 아이스링크 후기"],
        ["GHS-25W-02", "그랜드 하얏트 서울", "1", "2025-12-22", "https://blog.example/z", "판정 보류", "제목: 신규 표본"],
    ]
    backup = write_annex(workbook, sheet3_rows, sheet4_rows, backup=True)
    assert backup is not None and backup.exists()
    wb = load_workbook(workbook)
    ws3 = wb["C_블로그"]
    values = {str(ws3.cell(r, 1).value): [str(ws3.cell(r, c).value) for c in range(1, 19)]
              for r in range(3, ws3.max_row + 1) if ws3.cell(r, 1).value}
    assert values["GHS-25W-02"][5] == "59"  # [API 집계 필요] 충전
    assert values["GHS-25W-02"][13] == "30"
    assert values["SHL-25W-02"][5] == "[API 집계 필요]"  # 미수집 상품 행 무훼손
    ws4 = wb["C_협찬판정"]
    rows4 = [[str(ws4.cell(r, c).value) for c in range(1, 8)] for r in range(3, ws4.max_row + 1) if ws4.cell(r, 1).value]
    assert len(rows4) == 3
    assert {row[4] for row in rows4} == {"https://blog.example/x", "https://blog.example/y", "https://blog.example/z"}


def test_write_annex_without_backup(workbook: Path) -> None:
    backup = write_annex(workbook, [], [], backup=False)
    assert backup is None


def test_gate_d_reminder_flips_on_flag(tmp_path: Path) -> None:
    message = gate_d_reminder(tmp_path)
    assert "Gate D" in message
    (tmp_path / "gate_d_done.flag").write_text("done", encoding="utf-8")
    assert gate_d_reminder(tmp_path) == ""
