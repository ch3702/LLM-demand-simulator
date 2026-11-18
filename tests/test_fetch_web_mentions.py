"""웹 멘션 수집기 분류·안전장치 테스트 (네트워크 없음)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fetch_web_mentions import (  # noqa: E402
    Sample,
    classify,
    strip_tags,
)


def _sample(body: str, title: str = "테스트"):
    return Sample(url="https://blog.naver.com/x/1", title=title, posted="2024-12-05",
                  body=body, tags="#테스트", classification="판정 보류", evidence="")


class TestClassify:
    def test_subsidy_keywords_first(self):
        label, evidence = classify(_sample("아이스링크 초대 받았어요. 제공받고 다녀왔습니다."), "그랜드 하앏트 서울")
        assert label == "협찬"

    def test_self_paid_detected(self):
        label, _ = classify(_sample("내돈내산으로 직접 결제해 다녀온 후기"), "그랜드 하앏트 서울")
        assert label == "자비 표시"

    def test_corporate_press_detected(self):
        label, _ = classify(_sample("사진=그랜드 하앏트 서울 제공. 보도자료입니다"), "그랜드 하앏트 서울")
        assert label == "기업 게시"

    def test_plain_review_is_no_disclosure(self):
        label, evidence = classify(_sample("LA갈비가 제일 맛있었다"), "그랜드 하앏트 서울")
        assert label == "표시 없음"
        assert "미발견" in evidence

    def test_subsidy_beats_self_when_both(self):
        label, _ = classify(_sample("제공받았고 내돈내산 아님"), "그랜드 하앏트 서울")
        assert label == "협찬"


class TestStripTags:
    def test_removes_tags_and_collapses(self):
        assert strip_tags("<p>본문</p>\n  <b>굵게</b>") == "본문 굵게"
