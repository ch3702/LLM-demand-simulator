from __future__ import annotations

import asyncio  # noqa: ANYIO_OK — OpenAI AsyncOpenAI 클라이언트 경로에 필요
import base64
import csv
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # noqa: PANDAS_OK — 레포 표준 데이터 파이프라인
from openai import AsyncOpenAI, OpenAI
from tqdm.auto import tqdm

from .config import PACKAGE_DIR


class LLMConfigError(ValueError):
    """LLM 설정(모드·엔드포인트)이 올바르지 않을 때."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    model: str = "gpt-5-mini"
    requests_per_minute: int = 60
    max_concurrency: int = 10
    n_draws: int = 1
    base_url: str | None = None
    api_mode: str = "auto"  # "auto" | "responses" | "chat"

    def resolved_mode(self) -> str:
        """auto: OpenAI 공식 엔드포인트면 responses, 그 외 호환 서버는 chat."""
        if self.api_mode not in ("auto", "responses", "chat"):
            raise LLMConfigError(f"api_mode 는 auto/responses/chat 중 하나여야 합니다: {self.api_mode!r}")
        if self.api_mode != "auto":
            return self.api_mode
        url = (self.base_url or "").lower()
        if not url or "api.openai.com" in url:
            return "responses"
        return "chat"


def _client_kwargs(api_key: str, cfg: LLMConfig) -> dict[str, str]:
    kwargs = {"api_key": api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return kwargs


def content_for_chat(prompt: str, image_path: Path | None) -> list[dict]:
    """OpenAI 호환 chat/completions 용 콘텐츠(이미지 포함 가능)."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_path is None:
        return content
    mime, _ = mimetypes.guess_type(str(image_path))
    with image_path.open("rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("ascii")
    content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return content


async def _complete_async(client, cfg: LLMConfig, prompt: str, image_path: Path | None) -> str:
    if cfg.resolved_mode() == "chat":
        reply = await client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": content_for_chat(prompt, image_path)}],
        )
        return (reply.choices[0].message.content or "").strip()
    reply = await client.responses.create(
        model=cfg.model,
        input=[{"role": "user", "content": content_with_optional_image(prompt, image_path)}],
    )
    return reply.output_text


def _complete_sync(client, cfg: LLMConfig, prompt: str, image_path: Path | None) -> str:
    if cfg.resolved_mode() == "chat":
        reply = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": content_for_chat(prompt, image_path)}],
        )
        return (reply.choices[0].message.content or "").strip()
    reply = client.responses.create(
        model=cfg.model,
        input=[{"role": "user", "content": content_with_optional_image(prompt, image_path)}],
    )
    return reply.output_text


def content_with_optional_image(prompt: str, image_path: Path | None) -> list[dict]:
    content = [{"type": "input_text", "text": prompt}]
    if image_path is None:
        return content
    mime, _ = mimetypes.guess_type(str(image_path))
    with image_path.open("rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("ascii")
    content.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
    return content


def image_path_for_product(images_dir: Path, article_id: int) -> Path | None:
    path = images_dir / f"{int(article_id)}.jpg"
    return path if path.exists() else None


async def run_simulation_async(
    plan: pd.DataFrame,
    api_key: str,
    output_csv: Path,
    images_dir: Path,
    cfg: LLMConfig,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(output_csv)
    write_header = not output_csv.exists()
    handle = output_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    if write_header:
        writer.writerow(["persona_id", "article_id", "price_sig", "prices_json", "draw_id", "response", "ts", "image_path"])

    client = AsyncOpenAI(**_client_kwargs(api_key, cfg))
    sem = asyncio.Semaphore(cfg.max_concurrency)
    gate_state = {"lock": asyncio.Lock(), "next_t": 0.0}
    missing_images_seen: set[int] = set()
    pbar = tqdm(total=len(plan) * cfg.n_draws, desc="LLM simulation")

    async def one_row(row: pd.Series) -> None:
        persona_id = str(row["persona_id"])
        article_id = int(row["article_id"])
        price_sig = str(row["price_sig"])
        prices_json = str(row["prices_json"])
        prompt = str(row["prompt"])
        image_path = image_path_for_product(images_dir, article_id)
        if image_path is None and article_id not in missing_images_seen:
            print(f"No image found for product {article_id}.")
            missing_images_seen.add(article_id)
        missing_draws = [
            draw_id
            for draw_id in range(cfg.n_draws)
            if (persona_id, article_id, price_sig, draw_id) not in done
        ]
        if not missing_draws:
            pbar.update(cfg.n_draws)
            return
        async with sem:
            for draw_id in missing_draws:
                await _rate_limit(gate_state, cfg.requests_per_minute)
                text = await _complete_async(client, cfg, prompt, image_path)
                writer.writerow(
                    [
                        persona_id,
                        article_id,
                        price_sig,
                        prices_json,
                        draw_id,
                        text,
                        int(time.time()),
                        display_image_path(image_path),
                    ]
                )
                done.add((persona_id, article_id, price_sig, draw_id))
                handle.flush()
                pbar.update(1)

    try:
        await asyncio.gather(*(one_row(plan.iloc[idx]) for idx in range(len(plan))))
    finally:
        pbar.close()
        handle.close()


def run_simulation_sync(
    plan: pd.DataFrame,
    api_key: str,
    output_csv: Path,
    images_dir: Path,
    cfg: LLMConfig,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_keys(output_csv)
    write_header = not output_csv.exists()
    client = OpenAI(**_client_kwargs(api_key, cfg))
    missing_images_seen: set[int] = set()
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["persona_id", "article_id", "price_sig", "prices_json", "draw_id", "response", "ts", "image_path"])
        for _, row in tqdm(plan.iterrows(), total=len(plan), desc="LLM simulation"):
            persona_id = str(row["persona_id"])
            article_id = int(row["article_id"])
            price_sig = str(row["price_sig"])
            image_path = image_path_for_product(images_dir, article_id)
            if image_path is None and article_id not in missing_images_seen:
                print(f"No image found for product {article_id}.")
                missing_images_seen.add(article_id)
            for draw_id in range(cfg.n_draws):
                key = (persona_id, article_id, price_sig, draw_id)
                if key in done:
                    continue
                response = _complete_sync(client, cfg, str(row["prompt"]), image_path)
                writer.writerow(
                    [
                        persona_id,
                        article_id,
                        price_sig,
                        str(row["prices_json"]),
                        draw_id,
                        response,
                        int(time.time()),
                        display_image_path(image_path),
                    ]
                )
                done.add(key)
                handle.flush()


async def _rate_limit(gate_state: dict, requests_per_minute: int) -> None:
    min_interval = 60.0 / max(1, requests_per_minute)
    lock: asyncio.Lock = gate_state["lock"]
    async with lock:
        now = asyncio.get_running_loop().time()
        if now < gate_state["next_t"]:
            await asyncio.sleep(gate_state["next_t"] - now)
            now = asyncio.get_running_loop().time()
        gate_state["next_t"] = now + min_interval


def _completed_keys(output_csv: Path) -> set[tuple[str, int, str, int]]:
    if not output_csv.exists():
        return set()
    previous = pd.read_csv(output_csv)
    return set(
        zip(
            previous["persona_id"].astype(str),
            previous["article_id"].astype(int),
            previous["price_sig"].astype(str),
            previous["draw_id"].astype(int),
        )
    )


def display_image_path(image_path: Path | None) -> str:
    if image_path is None:
        return ""
    try:
        return image_path.resolve().relative_to(PACKAGE_DIR.resolve()).as_posix()
    except ValueError:
        return f"images/{image_path.name}"
