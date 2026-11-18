"""seasons_greeting-GHS — 네이버 블로그 언급량 수집 + 시트3·4 자동기입 (부록 A).

How to run (프로젝트 기간 매일 오전/저녁 수동 실행):
    NAVER_CLIENT_ID=... NAVER_CLIENT_SECRET=... bin/seasons_greeting-GHS
    bin/seasons_greeting-GHS --mock        # 크레덴셜 없이 파이프라인 점검
    bin/seasons_greeting-GHS --select GHS-25W-02

동작: 레지스트리 상품별 기간 창 언급량 집계 + 최신 표본 수집 → counts.json 저장
      → 표본이 --min-sample(기본 30) 충족한 상품만 sheet3(집계표)·sheet4(판정기록표) 자동 기입.
      sheet4 기존 '분류' 라벨은 링크 기준으로 보존되므로 수동 재판정이 매 실행마다 유지된다.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from annex_xlsx import read_sheet4_rows as read_annex_sheet4_rows
from annex_xlsx import write_annex

from naver_survey_core import (
    BlogItem,
    BlogResponse,
    BlogSearchClient,
    MissingCredentialsError,
    MentionStats,
    NaverBlogSearch,
    Product,
    ProductId,
    SamplePost,
    fill_plan,
    newest_samples,
    windowed_count,
)
from naver_survey_io import (
    Sheet4Row,
    load_registry,
    merge_sheet4_rows,
    read_sheet4,
    save_counts,
    sheet3_rows,
    tally_by_product,
    write_sheet3_rows,
    write_sheet4_rows,
)

_MOCK_PHRASES = ("내돈내산 방문 후기", "[광고] 체험단 방문", "정보성 요약 정리", "가족과 다녀온 후기", "데이트 코스 정리")


class MockBlogSearch:
    """결정론적 오프라인 응답 — 2025-11~2026-10 격일 포스트로 모든 창의 표본이 충족되게 한다."""

    def __init__(self) -> None:
        self._cache: dict[str, BlogResponse] = {}

    def search(self, query: str, start: int, sort: str) -> BlogResponse:
        if query not in self._cache:
            self._cache[query] = self._build(query)
        response = self._cache[query]
        items = response["items"]
        ordered = items if sort == "sim" else sorted(items, key=lambda item: item["postdate"], reverse=True)
        return BlogResponse(total=response["total"], items=ordered[start - 1 : start - 1 + 100])

    def _build(self, query: str) -> BlogResponse:
        digest = hashlib.md5(query.encode()).hexdigest()[:6]
        base = date(2025, 11, 1)
        items = [
            BlogItem(
                title=f"{query} {index}번째 후기",
                link=f"https://mock.blog/{digest}/{index}",
                postdate=base + timedelta(days=2 * index),
                description=_MOCK_PHRASES[index % len(_MOCK_PHRASES)],
                blogger=f"블로거{index % 7}",
            )
            for index in range(365)
        ]
        total = 300 + (len(query) * 137) % 1100
        return BlogResponse(total=max(total, len(items)), items=items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Naver blog mentions; auto-fill sheets when the sample is satisfied.")
    parser.add_argument("--registry", type=Path, default=Path("data/grand_hyatt_survey/products_naver.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/grand_hyatt_survey"))
    parser.add_argument("--mock", action="store_true", help="Naver API 대신 결정론적 오프라인 응답 사용")
    parser.add_argument("--min-sample", type=int, default=30, help="시트 자동기입에 필요한 표본 수 (기본 30)")
    parser.add_argument("--force", action="store_true", help="표본 미충족 상품도 강제 기입")
    parser.add_argument("--select", default="", help="쉼표 구분 product_id (지정 시 inactive 포함 선택)")
    parser.add_argument("--annex-xlsx", type=Path, default=None, help="부록 워크북 경로 — 시트3·4를 C_블로그·C_협찬판정에 직접 기입")
    parser.add_argument("--no-backup", action="store_true", help="부록 워크북 백업 생성 생략")
    return parser.parse_args()


def _client(use_mock: bool) -> BlogSearchClient:
    if use_mock:
        return MockBlogSearch()
    try:
        return NaverBlogSearch.from_env()
    except MissingCredentialsError as error:
        raise SystemExit(
            f"네이버 API 키 없음: {', '.join(error.env_keys)} 환경변수를 설정하거나 --mock 을 사용하세요"
        ) from None


def _collect(client: BlogSearchClient, products: list[Product], min_sample: int) -> tuple[
    list[tuple[Product, list[SamplePost]]], dict[ProductId, MentionStats], dict[ProductId, int],
]:
    entries: list[tuple[Product, list[SamplePost]]] = []
    stats: dict[ProductId, MentionStats] = {}
    sample_counts: dict[ProductId, int] = {}
    for product in products:
        product_counts = windowed_count(client, product.query, product.window_start, product.window_end)
        hotel_counts = windowed_count(client, product.hotel_query, product.window_start, product.window_end)
        posts = newest_samples(client, product.query, product.window_start, product.window_end, min_sample)
        stats[product.product_id] = MentionStats(product=product_counts, hotel=hotel_counts)
        sample_counts[product.product_id] = len(posts)
        entries.append((product, posts))
    return entries, stats, sample_counts


def _write_sheets(
    out_dir: Path,
    entries: list[tuple[Product, list[SamplePost]]],
    products: list[Product],
    stats: dict[ProductId, MentionStats],
    selected_ids: set[ProductId],
    today: str,
    is_mock: bool,
    prior_extra: list[Sheet4Row],
) -> tuple[list[list[str]], list[list[str]]]:
    sheet4 = out_dir / "sheet4_sample_judgments.csv"
    sheet3 = out_dir / "sheet3_blog_mentions.csv"
    entries_for_write = [entry for entry in entries if entry[0].product_id in selected_ids]
    csv_prior = read_sheet4(sheet4) if sheet4.exists() else []
    # 부록 워크북 수기 행을 먼저 두면 같은 링크에서 워크북 라벨이 우선한다(사람 판정 = 마스터).
    sheet4_matrix = merge_sheet4_rows(entries_for_write, list(prior_extra) + csv_prior)
    write_sheet4_rows(sheet4, sheet4_matrix)
    judgments = [(ProductId(row.product_id), row.label) for row in read_sheet4(sheet4)]
    sheet3_products = [product for product in products if product.product_id in selected_ids]
    sheet3_matrix = sheet3_rows(sheet3_products, stats, tally_by_product(judgments), today, is_mock)
    write_sheet3_rows(sheet3, sheet3_matrix)
    return sheet3_matrix, sheet4_matrix


def gate_d_reminder(out_dir: Path) -> str:
    """Gate D(주말 수요 실적 = 백테스트 모수) 미해결 알림. out_dir/gate_d_done.flag 생성 시 해제."""
    flag = out_dir / "gate_d_done.flag"
    if flag.exists():
        return ""
    return (
        "  ⚠ Gate D 미해결 — 주말 수요 실적(백테스트 모수) 미확보. "
        f"확보 시 '{flag}' 파일을 만들면 이 알림이 사라집니다."
    )


def _report(
    products: list[Product],
    stats: dict[ProductId, MentionStats],
    sample_counts: dict[ProductId, int],
    plan_ready: set[ProductId],
    min_sample: int,
    is_mock: bool,
    wrote: bool,
) -> None:
    mode = "MOCK" if is_mock else "LIVE"
    print(f"[seasons_greeting] {date.today().isoformat()} {mode} 수집 결과")
    for product in products:
        pid = product.product_id
        counts = stats[pid]
        state = "시트3·4 갱신 완료" if pid in plan_ready else "표본 미달 — 대기"
        capped = " (상한)" if counts.product.capped or counts.hotel.capped else ""
        print(
            f"  {pid}  표본 {sample_counts[pid]}/{min_sample}"
            f"  언급량 {counts.product.in_window} / 호텔 {counts.hotel.in_window}{capped}  → {state}"
        )
    print("  counts → naver_counts.json" + ("  | sheets → sheet3_blog_mentions.csv, sheet4_sample_judgments.csv" if wrote else ""))


def main() -> int:
    args = parse_args()
    client = _client(args.mock)
    selected = frozenset(args.select.split(",")) if args.select else None
    products = load_registry(args.registry, selected)
    entries, stats, sample_counts = _collect(client, products, args.min_sample)
    save_counts(args.out_dir / "naver_counts.json", stats, generated_with_mock=args.mock)
    plan = fill_plan(sample_counts, args.min_sample)
    write_ids = {product.product_id for product in products} if args.force else set(plan.ready)
    wrote = bool(write_ids)
    annex_backup: Path | None = None
    if wrote:
        prior_extra: list[Sheet4Row] = []
        if args.annex_xlsx is not None and args.annex_xlsx.exists():
            prior_extra = read_annex_sheet4_rows(args.annex_xlsx)
        sheet3_matrix, sheet4_matrix = _write_sheets(
            args.out_dir, entries, products, stats, write_ids,
            date.today().isoformat(), args.mock, prior_extra,
        )
        if args.annex_xlsx is not None:
            annex_backup = write_annex(
                args.annex_xlsx, sheet3_matrix, sheet4_matrix, backup=not args.no_backup,
            )
    _report(products, stats, sample_counts, set(plan.ready), args.min_sample, args.mock, wrote)
    if annex_backup is not None:
        print(f"  annex → {args.annex_xlsx}  (backup: {annex_backup.name})")
    gate = gate_d_reminder(args.out_dir)
    if gate:
        print(gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
