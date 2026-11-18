"""Registry + sheet/counts I/O for the Naver survey collector (시트3·4 자동기입)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from naver_survey_core import MentionStats, ParseError, Product, ProductId, SamplePost, WindowCount, segment_of

CANONICAL_LABELS = ("협찬", "기업 게시", "자비 구매 명시", "경제적 이해관계 표시 없음", "판정 보류")
_LABEL_ALIASES = {"자비 표시": "자비 구매 명시", "표시 없음": "경제적 이해관계 표시 없음"}
_LABEL_FIELDS = {"협찬": "sponsored", "기업 게시": "corporate", "자비 구매 명시": "self_paid", "경제적 이해관계 표시 없음": "unlabeled", "판정 보류": "pending"}
_DEFAULT_LABEL = "판정 보류"
_SHEET3_HEADER = (
    "상품ID", "호텔", "검색어(호텔명+이벤트명)", "기간 창 시작(판매 시작일)", "기간 창 끝(판매 종료 +30일)",
    "상품 언급량", "같은 기간 호텔명 언급량", "상품/호텔 비율", "판정 편수", "협찬", "기업 게시",
    "자비 표시", "표시 없음", "판정 보류", "협찬 비율", "자주 함께 나온 키워드", "수집일", "비고",
)
_SHEET4_HEADER = ("상품ID", "호텔", "구간", "작성일", "글 링크", "분류", "근거")


@dataclass(frozen=True, slots=True)
class LabelTally:
    sponsored: int
    corporate: int
    self_paid: int
    unlabeled: int
    pending: int

    @property
    def judged(self) -> int:
        return self.sponsored + self.corporate + self.self_paid + self.unlabeled + self.pending

    @property
    def sponsored_ratio(self) -> float:
        return self.sponsored / self.judged if self.judged else 0.0


@dataclass(frozen=True, slots=True)
class Sheet4Row:
    product_id: str
    hotel: str
    segment: str
    postdate: str
    link: str
    label: str
    evidence: str


def _read_commented_csv(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return [dict(row) for row in csv.DictReader(lines)]


def read_registry_rows(path: Path) -> list[dict[str, str]]:
    """레지스트리 원문 행(주석 제외) — 동기화 병합용."""
    return _read_commented_csv(path)


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _segment_dates(row: dict[str, str], window_start: date, window_end: date) -> tuple[date, date]:
    raw2, raw3 = (row.get("seg2_start") or "").strip(), (row.get("seg3_start") or "").strip()
    if raw2 and raw3:
        return date.fromisoformat(raw2), date.fromisoformat(raw3)
    span = (window_end - window_start) // 3
    return window_start + span, window_start + 2 * span


def _parse_product_row(pid: str, row: dict[str, str]) -> Product:
    try:
        window_start = date.fromisoformat(row["window_start"].strip())
        window_end = date.fromisoformat(row["window_end"].strip())
        seg2_start, seg3_start = _segment_dates(row, window_start, window_end)
    except ValueError as error:
        raise ParseError(field="window", detail=f"{pid}: invalid date ({error})") from None
    return Product(
        product_id=ProductId(pid),
        hotel=row["hotel"].strip(),
        query=row["query"].strip(),
        hotel_query=(row.get("hotel_query") or "").strip() or row["hotel"].strip(),
        window_start=window_start,
        window_end=window_end,
        seg2_start=seg2_start,
        seg3_start=seg3_start,
        keywords=(row.get("keywords") or "").strip(),
        note=(row.get("note") or "").strip(),
    )


def load_registry(path: Path, selected: frozenset[str] | None = None) -> list[Product]:
    """active=1 행을 읽는다. selected 지정 시 해당 product_id는 active 여부와 무관하게 포함."""
    products: list[Product] = []
    for row in _read_commented_csv(path):
        pid = row["product_id"].strip()
        if not pid:
            continue
        is_active = (row.get("active") or "1").strip() == "1"
        if not is_active and (selected is None or pid not in selected):
            continue
        products.append(_parse_product_row(pid, row))
    return products


def canonical_label(raw: str) -> str:
    label = _LABEL_ALIASES.get(raw.strip(), raw.strip())
    if label not in CANONICAL_LABELS:
        raise ParseError(field="분류", detail=f"unknown label {raw!r}; expected one of {CANONICAL_LABELS}")
    return label


def read_sheet4(path: Path) -> list[Sheet4Row]:
    rows: list[Sheet4Row] = []
    for raw in _read_commented_csv(path):
        rows.append(
            Sheet4Row(
                product_id=(raw.get("상품ID") or "").strip(),
                hotel=(raw.get("호텔") or "").strip(),
                segment=(raw.get("구간") or "").strip(),
                postdate=(raw.get("작성일") or "").strip(),
                link=(raw.get("글 링크") or "").strip(),
                label=canonical_label(raw.get("분류") or ""),
                evidence=(raw.get("근거") or "").strip(),
            )
        )
    return rows


def tally_by_product(judgments: list[tuple[ProductId, str]]) -> dict[ProductId, LabelTally]:
    buckets: dict[ProductId, dict[str, int]] = {}
    for pid, label in judgments:
        field = _LABEL_FIELDS.get(label)
        if field is None:
            raise ParseError(field="분류", detail=f"unknown label {label!r}")
        stats = buckets.setdefault(pid, dict.fromkeys(_LABEL_FIELDS.values(), 0))
        stats[field] += 1
    return {pid: LabelTally(**fields) for pid, fields in buckets.items()}


def _sampled_rows(
    entries: list[tuple[Product, list[SamplePost]]],
    prior_labels: dict[str, str],
) -> list[list[str]]:
    return [
        [
            str(product.product_id),
            product.hotel,
            segment_of(product, post.postdate).value,
            post.postdate.isoformat(),
            post.link,
            prior_labels.get(post.link, _DEFAULT_LABEL),
            f"제목: {post.title} / 요약: {post.summary}",
        ]
        for product, posts in entries
        for post in posts
    ]


def merge_sheet4_rows(
    entries: list[tuple[Product, list[SamplePost]]],
    prior_rows: list[Sheet4Row],
) -> list[list[str]]:
    """표본 행 + 수기 행 병합(순수 함수). 수기 라벨은 링크 기준 승계, 표본 밖 수기 행은 캐리오버.

    중복 제거 키는 (상품ID, 구간, 작성일, 링크) — 부록 수기행의 자리표시 링크 반복
    (예: 여러 행이 'naver.com')을 유실하지 않으면서 재실행 누적 중복만 걸러낸다.
    """
    new_links = {post.link for _, posts in entries for post in posts}
    seen: set[tuple[str, str, str, str]] = set()
    carried: list[list[str]] = []
    for row in prior_rows:
        if row.link in new_links:
            continue
        identity = (row.product_id, row.segment, row.postdate, row.link)
        if identity in seen:
            continue
        seen.add(identity)
        carried.append([row.product_id, row.hotel, row.segment, row.postdate, row.link, row.label, row.evidence])
    prior_labels = {row.link: row.label for row in prior_rows}
    return carried + _sampled_rows(entries, prior_labels)


def write_sheet4_rows(path: Path, rows: list[list[str]]) -> None:
    _write_csv(path, _SHEET4_HEADER, rows)


def write_sheet4(
    path: Path,
    entries: list[tuple[Product, list[SamplePost]]],
    prior_labels: dict[str, str],
    collected_on: str,
) -> None:
    """표본 행 기입. prior_labels로 사람이 재판정한 라벨을 보존하고, 표본 밖 기존 행은 그대로 유지."""
    prior_rows = read_sheet4(path) if path.exists() else []
    new_links = {post.link for _, posts in entries for post in posts}
    carried = [
        [row.product_id, row.hotel, row.segment, row.postdate, row.link, row.label, row.evidence]
        for row in prior_rows
        if row.link not in new_links
    ]
    _write_csv(path, _SHEET4_HEADER, carried + _sampled_rows(entries, prior_labels))


def sheet3_rows(
    products: list[Product],
    stats: dict[ProductId, MentionStats],
    tallies: dict[ProductId, LabelTally],
    collected_on: str,
    is_mock: bool,
) -> list[list[str]]:
    """시트3 행 행렬(순수 함수) — CSV 기입과 부록 xlsx 직접기입이 공유."""
    rows: list[list[str]] = []
    for product in products:
        counts = stats[product.product_id]
        tally = tallies.get(product.product_id, LabelTally(0, 0, 0, 0, 0))
        ratio = counts.product.in_window / counts.hotel.in_window if counts.hotel.in_window else 0.0
        note = product.note
        if counts.product.capped or counts.hotel.capped:
            note = (note + " · " if note else "") + "언급량 상한(1000건) 도달 — total 추정치 참조"
        if is_mock:
            note = (note + " · " if note else "") + "[MOCK 데이터] 실 API 키 설정 후 재수집 필요"
        rows.append(
            [
                str(product.product_id), product.hotel, product.query,
                product.window_start.isoformat(), product.window_end.isoformat(),
                str(counts.product.in_window), str(counts.hotel.in_window), f"{ratio:.4f}",
                str(tally.judged), str(tally.sponsored), str(tally.corporate), str(tally.self_paid),
                str(tally.unlabeled), str(tally.pending), f"{tally.sponsored_ratio:.4f}",
                product.keywords, collected_on, note,
            ]
        )
    return rows


def write_sheet3_rows(path: Path, rows: list[list[str]]) -> None:
    _write_csv(path, _SHEET3_HEADER, rows)


def write_sheet3(
    path: Path,
    products: list[Product],
    stats: dict[ProductId, MentionStats],
    tallies: dict[ProductId, LabelTally],
    collected_on: str,
    is_mock: bool,
) -> None:
    write_sheet3_rows(path, sheet3_rows(products, stats, tallies, collected_on, is_mock))


def _wc_dict(counts: WindowCount) -> dict[str, int | bool]:
    return {"in_window": counts.in_window, "total_estimate": counts.total_estimate, "capped": counts.capped}


def save_counts(path: Path, stats: dict[ProductId, MentionStats], generated_with_mock: bool) -> None:
    payload = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "generated_with_mock": generated_with_mock,
        "products": {
            str(pid): {"product": _wc_dict(stat.product), "hotel": _wc_dict(stat.hotel)}
            for pid, stat in stats.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_counts(path: Path) -> tuple[dict[ProductId, MentionStats], bool]:
    """counts.json 경계 파싱 — 구조가 어긋나면 ParseError."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ParseError(field="counts", detail="expected JSON object")
    products = payload.get("products")
    if not isinstance(products, dict):
        raise ParseError(field="counts", detail="missing products object")
    stats: dict[ProductId, MentionStats] = {}
    for pid, entry in products.items():
        if not isinstance(entry, dict):
            raise ParseError(field="counts", detail=f"product {pid}: expected object")
        product, hotel = entry.get("product"), entry.get("hotel")
        if not isinstance(product, dict) or not isinstance(hotel, dict):
            raise ParseError(field="counts", detail=f"product {pid}: expected product/hotel objects")
        stats[ProductId(str(pid))] = MentionStats(product=WindowCount(**product), hotel=WindowCount(**hotel))
    return stats, bool(payload.get("generated_with_mock", False))
