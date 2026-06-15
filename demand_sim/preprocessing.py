from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import PRICE_SCALE


@dataclass(frozen=True)
class ProductPreprocessConfig:
    top_n: int = 100
    online_only: bool = True
    product_type: str = "Trousers"
    cutoff_date: str = "2019-09-20"
    min_distinct_prices: int = 5


@dataclass(frozen=True)
class PersonaPreprocessConfig:
    train_start: str = "2018-09-01"
    train_end: str = "2019-09-19"
    n_personas: int = 100
    top_taste_k: int = 100
    chunksize: int = 2_000_000


def build_products(
    transactions_path: Path,
    articles_path: Path,
    output_dir: Path,
    cfg: ProductPreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build top-product daily demand and product metadata tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    txns = pd.read_csv(transactions_path)
    articles = pd.read_csv(articles_path)
    if cfg.online_only:
        txns = txns[txns["sales_channel_id"] == 1].copy()

    product_ids = articles.loc[articles["product_type_name"] == cfg.product_type, "article_id"]
    txns = txns[txns["article_id"].isin(product_ids)].copy()
    txns.to_csv(output_dir / f"txns_{cfg.product_type.lower()}_{'online' if cfg.online_only else 'all'}.csv", index=False)

    txns["t_dat"] = pd.to_datetime(txns["t_dat"])
    demand = (
        txns.groupby(["t_dat", "article_id", "price"], as_index=False)
        .size()
        .rename(columns={"size": "demand", "t_dat": "date"})
    )
    demand["price"] = (demand["price"] * PRICE_SCALE).round(2)

    one_price = demand.groupby(["date", "article_id"])["price"].transform("nunique").eq(1)
    demand = demand[one_price].copy()
    demand = demand[demand["date"] < pd.Timestamp(cfg.cutoff_date)].copy()

    n_prices = demand.groupby("article_id")["price"].nunique()
    eligible = n_prices[n_prices >= cfg.min_distinct_prices].index
    demand = demand[demand["article_id"].isin(eligible)].copy()

    top = (
        demand.groupby("article_id")["demand"]
        .sum()
        .sort_values(ascending=False)
        .head(cfg.top_n)
        .index
    )
    sales = demand[demand["article_id"].isin(top)].copy()
    sales = sales.rename(columns={"price": "offer_price"}).sort_values(["article_id", "date", "offer_price"])

    suffix = f"top{cfg.top_n}_{'online' if cfg.online_only else 'all'}"
    sales.to_csv(output_dir / f"sales_{suffix}.csv", index=False)
    product_info = articles[articles["article_id"].isin(top)].drop_duplicates("article_id").copy()
    product_info.to_csv(output_dir / f"product_info_{suffix}.csv", index=False)
    return sales, product_info


def build_personas(
    transactions_path: Path,
    customers_path: Path,
    articles_path: Path,
    output_dir: Path,
    cfg: PersonaPreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build customer feature table and persona prompt table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0, t1 = pd.to_datetime(cfg.train_start), pd.to_datetime(cfg.train_end)
    customers = pd.read_csv(customers_path, usecols=["customer_id", "age"])
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customer_index = pd.Index(customers["customer_id"].astype(str).to_numpy())

    articles = pd.read_csv(articles_path, usecols=["article_id", "product_type_name"])
    articles = articles.dropna(subset=["article_id", "product_type_name"])
    type_names = sorted(articles["product_type_name"].unique().tolist())
    type_to_id = {name: idx for idx, name in enumerate(type_names)}
    articles["type_id"] = articles["product_type_name"].map(type_to_id)
    article_to_type = pd.Series(articles["type_id"].to_numpy(), index=articles["article_id"].to_numpy())

    n_customers = len(customers)
    n_types = len(type_names)
    txn_count = np.zeros(n_customers)
    sum_price = np.zeros(n_customers)
    taste_counts = np.zeros((n_customers, n_types))

    reader = pd.read_csv(
        transactions_path,
        usecols=["t_dat", "customer_id", "article_id", "price"],
        chunksize=cfg.chunksize,
    )
    for chunk in tqdm(reader, desc="customer features", unit="chunk"):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        chunk = chunk[(chunk["t_dat"] >= t0) & (chunk["t_dat"] <= t1)].copy()
        if chunk.empty:
            continue
        chunk["cidx"] = customer_index.get_indexer(chunk["customer_id"].astype(str).to_numpy())
        chunk = chunk[chunk["cidx"] >= 0].copy()
        chunk["type_id"] = chunk["article_id"].map(article_to_type)
        chunk = chunk.dropna(subset=["type_id", "price"])
        if chunk.empty:
            continue
        chunk["price"] = chunk["price"] * PRICE_SCALE
        by_customer = chunk.groupby("cidx").agg(n=("price", "size"), s=("price", "sum"))
        idx = by_customer.index.to_numpy()
        txn_count[idx] += by_customer["n"].to_numpy()
        sum_price[idx] += by_customer["s"].to_numpy()
        by_taste = chunk.groupby(["cidx", "type_id"]).size().reset_index(name="cnt")
        for row in by_taste.itertuples(index=False):
            taste_counts[int(row.cidx), int(row.type_id)] += int(row.cnt)

    mean_price = np.divide(sum_price, txn_count, out=np.zeros_like(sum_price), where=txn_count > 0)
    customer_features = customers.copy()
    customer_features["txn_count"] = txn_count
    customer_features["mean_price"] = np.round(mean_price, 2)
    top_type_id = taste_counts.argmax(axis=1)
    customer_features["top_product_type"] = [
        type_names[int(type_id)] if txn_count[idx] > 0 else "NO_PURCHASE"
        for idx, type_id in enumerate(top_type_id)
    ]
    customer_features.to_csv(output_dir / "customer_features.csv", index=False)

    personas = _build_persona_cells(customer_features, t0, t1, cfg)
    personas.to_csv(output_dir / "persona_cells.csv", index=False)
    personas[["persona_id", "n_customers", "persona_prompt", "age_bin", "engagement_bin", "price_tier", "taste_bucket"]].to_csv(
        output_dir / "persona_prompts.csv",
        index=False,
    )
    return customer_features, personas


def _build_persona_cells(
    customer_features: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    cfg: PersonaPreprocessConfig,
) -> pd.DataFrame:
    df = customer_features[customer_features["txn_count"] > 0].copy()
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[16, 25, 35, 45, 55, 200],
        right=False,
        labels=["16-24", "25-34", "35-44", "45-54", "55+"],
    )
    span_months = max(((train_end - train_start).days + 1) / 30.4375, 1e-6)
    df["txn_per_month"] = df["txn_count"] / span_months
    df["engagement_bin"] = _qcut_with_rank_fallback(df["txn_per_month"], ["low", "mid", "high"])
    df["price_tier"] = _qcut_with_rank_fallback(df["mean_price"], ["low", "mid", "high"])
    top_types = df["top_product_type"].value_counts().head(cfg.top_taste_k).index
    df["taste_bucket"] = np.where(df["top_product_type"].isin(top_types), df["top_product_type"], "OTHER")

    cell_cols = ["age_bin", "engagement_bin", "price_tier", "taste_bucket"]
    cells = df.groupby(cell_cols, dropna=False).size().reset_index(name="n_customers")
    cells = cells.sort_values("n_customers", ascending=False).head(cfg.n_personas).reset_index(drop=True)
    cells["persona_id"] = [f"P{idx + 1:03d}" for idx in range(len(cells))]
    stats = (
        df.groupby(cell_cols)
        .agg(
            txn_q25=("txn_per_month", lambda s: float(np.quantile(s, 0.25))),
            txn_q75=("txn_per_month", lambda s: float(np.quantile(s, 0.75))),
            price_q25=("mean_price", lambda s: float(np.quantile(s, 0.25))),
            price_q75=("mean_price", lambda s: float(np.quantile(s, 0.75))),
        )
        .reset_index()
    )
    cells = cells.merge(stats, on=cell_cols, how="left")
    cells["persona_prompt"] = cells.apply(_persona_prompt, axis=1)
    return cells


def _qcut_with_rank_fallback(values: pd.Series, labels: list[str]) -> pd.Series:
    out = pd.qcut(values, q=len(labels), labels=labels, duplicates="drop")
    if out.nunique(dropna=True) >= 2:
        return out
    ranked = values.rank(method="average")
    n_bins = min(len(labels), int(ranked.nunique()))
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates="drop")


def _integer_interval(q25: float, q75: float) -> tuple[int, int]:
    low = int(np.floor(q25))
    high = int(np.ceil(q75))
    if high <= low:
        high = low + 1
    return low, high


def _persona_prompt(row: pd.Series) -> str:
    freq_low, freq_high = _integer_interval(float(row["txn_q25"]), float(row["txn_q75"]))
    return (
        "You are a customer of H&M in the years 2018-2019. "
        f"Your age is {row['age_bin']}. "
        f"You make about {freq_low} to {freq_high} purchases per month. "
        f"Your typical paid price is between {row['price_q25']:.2f} and {row['price_q75']:.2f}. "
        f"Most of your purchases are in: {row['taste_bucket']}.\n\n"
        "Task: Given a product and a list of prices, return the probability you would buy at each price. "
        "Output JSON exactly: {\"prices\": [...], \"p_buy\": [...], \"reason\": \"<=30 words\"}."
    )


def build_query_plan(
    sales_path: Path,
    product_info_path: Path,
    persona_prompts_path: Path,
    output_path: Path,
    max_prices: int = 10,
    max_personas: int = 50,
) -> pd.DataFrame:
    """Build the product-persona LLM query plan."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(sales_path)
    if "price" in sales.columns and "offer_price" not in sales.columns:
        sales = sales.rename(columns={"price": "offer_price"})
    pop = sales.groupby(["article_id", "offer_price"]).size().reset_index(name="popularity")
    pairs: list[tuple[int, float]] = []
    for article_id, group in tqdm(pop.groupby("article_id", sort=False), desc="price grid"):
        chosen = (
            group.sort_values(["popularity", "offer_price"], ascending=[False, True])
            .head(max_prices)["offer_price"]
            .to_list()
        )
        for price in sorted(chosen):
            pairs.append((int(article_id), float(price)))
    price_grid = pd.DataFrame(pairs, columns=["article_id", "offer_price"])
    prices_by_product = price_grid.groupby("article_id")["offer_price"].apply(
        lambda s: sorted(float(x) for x in s.dropna().unique().tolist())
    )

    products = pd.read_csv(product_info_path)
    prod_cols = ["article_id", "prod_name", "product_type_name", "colour_group_name", "detail_desc"]
    product_table = products[prod_cols].drop_duplicates("article_id").set_index("article_id")
    personas = pd.read_csv(persona_prompts_path).head(max_personas).copy()

    plan = pd.DataFrame({"article_id": prices_by_product.index.to_numpy(dtype=int)})
    plan = plan.assign(_key=1).merge(personas.assign(_key=1), on="_key").drop(columns="_key")
    plan["prices"] = plan["article_id"].map(lambda article_id: prices_by_product.loc[article_id])
    plan["prices_json"] = plan["prices"].map(json.dumps)
    plan["price_sig"] = plan["prices_json"].map(lambda text: hashlib.md5(text.encode("utf-8")).hexdigest())
    plan["product_block"] = plan["article_id"].map(lambda article_id: _product_prompt_block(product_table.loc[article_id]))
    plan["prompt"] = plan.apply(
        lambda row: (
            str(row["persona_prompt"]).strip()
            + "\n\n"
            + str(row["product_block"]).strip()
            + "\n\n"
            + "Offered prices (USD): "
            + json.dumps(sorted(float(price) for price in row["prices"]))
            + "\n"
        ),
        axis=1,
    )
    plan.to_pickle(output_path)
    plan[["persona_id", "article_id", "prices_json", "price_sig", "prompt"]].to_csv(
        output_path.with_suffix(".csv"),
        index=False,
    )
    return plan


def _product_prompt_block(row: pd.Series) -> str:
    return "\n".join(
        [
            f"Product name: {row['prod_name']}",
            f"Product type: {row['product_type_name']}",
            f"Product color: {row['colour_group_name']}",
            f"Detailed description: {row['detail_desc']}",
        ]
    )

