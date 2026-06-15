from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_llm_response(text: str) -> tuple[list[float], list[float], str] | None:
    """Parse a JSON LLM response with prices and purchase probabilities."""
    if not isinstance(text, str):
        return None
    payload = text.strip()
    if not payload or payload.startswith("ERROR"):
        return None
    payload = re.sub(r"^\s*```(?:json)?\s*", "", payload)
    payload = re.sub(r"\s*```\s*$", "", payload)
    if not payload.startswith("{"):
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if match is None:
            return None
        payload = match.group(0)
    payload = payload.replace(r"\$", "$")
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    prices = obj.get("prices")
    probs = obj.get("p_buy")
    if not isinstance(prices, list) or not isinstance(probs, list) or len(prices) != len(probs):
        return None
    try:
        clean_prices = [float(price) for price in prices]
        clean_probs = [float(prob) for prob in probs]
    except (TypeError, ValueError):
        return None
    return clean_prices, clean_probs, str(obj.get("reason", ""))


def load_probability_rows(response_csv: Path) -> pd.DataFrame:
    """Convert raw LLM response rows to one row per persona-product-price."""
    raw = pd.read_csv(response_csv)
    rows: list[dict] = []
    bad_rows = 0
    for record in raw.itertuples(index=False):
        parsed = parse_llm_response(getattr(record, "response"))
        if parsed is None:
            bad_rows += 1
            continue
        prices, probs, reason = parsed
        persona_id = str(getattr(record, "persona_id"))
        article_id = int(getattr(record, "article_id"))
        draw_id = int(getattr(record, "draw_id", 0))
        for price, prob in zip(prices, probs):
            rows.append(
                {
                    "persona_id": persona_id,
                    "article_id": article_id,
                    "offer_price": round(float(price), 2),
                    "draw_id": draw_id,
                    "p_buy": float(np.clip(prob, 0.0, 1.0)),
                    "reason": reason,
                }
            )
    if bad_rows:
        print(f"Skipped {bad_rows} unparsable LLM responses.")
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No valid LLM probabilities parsed from {response_csv}")
    return out


def load_sales(path: Path) -> pd.DataFrame:
    sales = pd.read_csv(path)
    if "price" in sales.columns and "offer_price" not in sales.columns:
        sales = sales.rename(columns={"price": "offer_price"})
    required = {"date", "article_id", "offer_price", "demand"}
    missing = required.difference(sales.columns)
    if missing:
        raise ValueError(f"Sales file {path} is missing columns: {sorted(missing)}")
    sales["date"] = pd.to_datetime(sales["date"])
    sales["article_id"] = sales["article_id"].astype(int)
    sales["offer_price"] = sales["offer_price"].astype(float).round(2)
    sales["demand"] = sales["demand"].astype(int)
    return sales


def build_prompting_design_rows(sales: pd.DataFrame, probabilities: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join real demand rows with wide LLM persona probabilities."""
    q_wide = (
        probabilities.groupby(["article_id", "offer_price", "persona_id"], as_index=False)["p_buy"]
        .mean()
        .pivot(index=["article_id", "offer_price"], columns="persona_id", values="p_buy")
    )
    rows = sales.merge(q_wide.reset_index(), on=["article_id", "offer_price"], how="inner")
    persona_ids = sorted(
        col for col in rows.columns if col not in {"date", "article_id", "offer_price", "demand"}
    )
    rows = rows.dropna(subset=persona_ids).copy()
    return rows, persona_ids


def load_product_embeddings(path: Path) -> tuple[pd.DataFrame, list[str]]:
    embeddings = pd.read_csv(path)
    embeddings["article_id"] = embeddings["article_id"].astype(int)
    cols = [col for col in embeddings.columns if col.startswith(("text_emb_", "image_emb_", "product_emb_"))]
    if not cols:
        raise ValueError(f"No product embedding columns found in {path}")
    return embeddings[["article_id"] + cols].copy(), cols


def load_persona_embeddings(path: Path, max_personas: int | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    embeddings = pd.read_csv(path)
    embeddings["persona_id"] = embeddings["persona_id"].astype(str)
    cols = [col for col in embeddings.columns if col.startswith("persona_emb_")]
    if not cols:
        raise ValueError(f"No persona embedding columns found in {path}")
    embeddings = embeddings.sort_values("persona_id").reset_index(drop=True)
    if max_personas is not None:
        embeddings = embeddings.head(max_personas).copy()
    return embeddings[["persona_id"] + cols].copy(), embeddings["persona_id"].tolist(), cols

