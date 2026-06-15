from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
PRODUCT_TEXT_COLUMNS = [
    "prod_name",
    "product_type_name",
    "colour_group_name",
    "graphical_appearance_name",
    "detail_desc",
]


def build_product_embeddings(
    product_info_path: Path,
    image_dir: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 16,
    device_name: str | None = None,
    allow_download: bool = False,
) -> dict:
    """Build concatenated text+image SigLIP embeddings for products."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, AutoModel, AutoProcessor = _load_embedding_dependencies()

    product_info = pd.read_csv(product_info_path)
    product_info = product_info.drop_duplicates("article_id").sort_values("article_id").reset_index(drop=True)
    product_info["article_id"] = product_info["article_id"].astype(int)
    product_info["embedding_text"] = product_info.apply(build_product_text, axis=1)
    product_info["image_path"] = product_info["article_id"].map(lambda article_id: image_dir / f"{int(article_id)}.jpg")

    missing_images = [str(path) for path in product_info["image_path"] if not Path(path).exists()]
    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"Missing {len(missing_images)} product images. First missing paths:\n{preview}")

    device = choose_device(torch, device_name)
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=not allow_download)
    model = AutoModel.from_pretrained(model_id, dtype=dtype, local_files_only=not allow_download).to(device).eval()

    text_embeddings = encode_texts(product_info["embedding_text"].tolist(), model, processor, device, batch_size, "Encoding product text")
    image_embeddings = encode_images([Path(path) for path in product_info["image_path"]], model, processor, device, batch_size)
    embeddings = np.hstack([text_embeddings, image_embeddings]).astype(np.float32)
    article_ids = product_info["article_id"].to_numpy(dtype=np.int64)

    csv_path = output_dir / "siglip2_product_embeddings.csv"
    npz_path = output_dir / "siglip2_product_embeddings.npz"
    metadata_path = output_dir / "siglip2_product_embeddings_metadata.json"
    np.savez_compressed(
        npz_path,
        article_id=article_ids,
        text_embedding=text_embeddings.astype(np.float32),
        image_embedding=image_embeddings.astype(np.float32),
        embedding=embeddings,
    )
    columns = {"article_id": article_ids}
    columns.update({f"text_emb_{idx:04d}": text_embeddings[:, idx] for idx in range(text_embeddings.shape[1])})
    columns.update({f"image_emb_{idx:04d}": image_embeddings[:, idx] for idx in range(image_embeddings.shape[1])})
    pd.DataFrame(columns).to_csv(csv_path, index=False)
    metadata = {
        "model_id": model_id,
        "product_info": str(product_info_path),
        "image_dir": str(image_dir),
        "n_products": int(len(product_info)),
        "text_embedding_dim": int(text_embeddings.shape[1]),
        "image_embedding_dim": int(image_embeddings.shape[1]),
        "concatenated_embedding_dim": int(embeddings.shape[1]),
        "text_columns": PRODUCT_TEXT_COLUMNS,
        "outputs": {"csv": str(csv_path), "npz": str(npz_path)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def build_persona_embeddings(
    personas_path: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 32,
    device_name: str | None = None,
    allow_download: bool = False,
) -> dict:
    """Build SigLIP text embeddings for persona descriptions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, AutoModel, AutoProcessor = _load_embedding_dependencies()

    personas = pd.read_csv(personas_path)
    personas = personas.sort_values("persona_id").reset_index(drop=True)
    personas["persona_text"] = personas.apply(build_persona_text, axis=1)

    device = choose_device(torch, device_name)
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=not allow_download)
    model = AutoModel.from_pretrained(model_id, dtype=dtype, local_files_only=not allow_download).to(device).eval()

    embeddings = encode_texts(personas["persona_text"].tolist(), model, processor, device, batch_size, "Encoding persona text")
    csv_path = output_dir / "siglip2_persona_embeddings.csv"
    npz_path = output_dir / "siglip2_persona_embeddings.npz"
    metadata_path = output_dir / "siglip2_persona_embeddings_metadata.json"
    columns = {"persona_id": personas["persona_id"].astype(str).to_numpy()}
    columns.update({f"persona_emb_{idx:04d}": embeddings[:, idx] for idx in range(embeddings.shape[1])})
    pd.DataFrame(columns).to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, persona_id=personas["persona_id"].astype(str).to_numpy(), embedding=embeddings)
    metadata = {
        "model_id": model_id,
        "personas": str(personas_path),
        "n_personas": int(len(personas)),
        "embedding_dim": int(embeddings.shape[1]),
        "outputs": {"csv": str(csv_path), "npz": str(npz_path)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def build_product_text(row: pd.Series) -> str:
    labels = {
        "prod_name": "Product name",
        "product_type_name": "Product type",
        "colour_group_name": "Color",
        "graphical_appearance_name": "Appearance",
        "detail_desc": "Description",
    }
    parts = []
    for col in PRODUCT_TEXT_COLUMNS:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            parts.append(f"{labels[col]}: {str(value).strip()}")
    return "\n".join(parts)


def build_persona_text(row: pd.Series) -> str:
    parts = [
        f"Age group: {row['age_bin']}",
        f"Purchase frequency: {row['engagement_bin']}",
        f"Typical paid price tier: {row['price_tier']}",
        f"Most common product category: {row['taste_bucket']}",
    ]
    if pd.notna(row.get("txn_q25")) and pd.notna(row.get("txn_q75")):
        parts.append(f"Purchases per month: between {float(row['txn_q25']):.2f} and {float(row['txn_q75']):.2f}")
    if pd.notna(row.get("price_q25")) and pd.notna(row.get("price_q75")):
        parts.append(f"Typical paid price: between {float(row['price_q25']):.2f} and {float(row['price_q75']):.2f}")
    return "\n".join(parts)


def choose_device(torch, requested: str | None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_texts(texts: list[str], model, processor, device, batch_size: int, desc: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    arrays = []
    for batch in tqdm(list(batched(texts, batch_size)), desc=desc):
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            features = pooled_tensor(model.get_text_features(**inputs))
            features = F.normalize(features, p=2, dim=-1)
        arrays.append(features.cpu().numpy())
    return np.vstack(arrays).astype(np.float32)


def encode_images(image_paths: list[Path], model, processor, device, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    arrays = []
    for batch_paths in tqdm(list(batched(image_paths, batch_size)), desc="Encoding product images"):
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            features = pooled_tensor(model.get_image_features(**inputs))
            features = F.normalize(features, p=2, dim=-1)
        arrays.append(features.cpu().numpy())
    return np.vstack(arrays).astype(np.float32)


def pooled_tensor(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    return output


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_embedding_dependencies():
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing embedding dependencies. Install them with:\n"
            "  pip install torch transformers pillow tqdm\n"
            f"Original import error: {exc}"
        ) from exc
    return torch, AutoModel, AutoProcessor

