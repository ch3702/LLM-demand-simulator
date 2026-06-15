from __future__ import annotations

import asyncio
import base64
import csv
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI, OpenAI
from tqdm.auto import tqdm

from .config import PACKAGE_DIR


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-5-mini"
    requests_per_minute: int = 60
    max_concurrency: int = 10
    n_draws: int = 1


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

    client = AsyncOpenAI(api_key=api_key)
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
                response = await client.responses.create(
                    model=cfg.model,
                    input=[{"role": "user", "content": content_with_optional_image(prompt, image_path)}],
                )
                writer.writerow(
                    [
                        persona_id,
                        article_id,
                        price_sig,
                        prices_json,
                        draw_id,
                        response.output_text,
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
    client = OpenAI(api_key=api_key)
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
                response = client.responses.create(
                    model=cfg.model,
                    input=[{"role": "user", "content": content_with_optional_image(str(row["prompt"]), image_path)}],
                )
                writer.writerow(
                    [
                        persona_id,
                        article_id,
                        price_sig,
                        str(row["prices_json"]),
                        draw_id,
                        response.output_text,
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
