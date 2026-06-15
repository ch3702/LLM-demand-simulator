from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.config import DEFAULT_EMBEDDINGS_DIR, DEFAULT_PERSONAS_DIR
from demand_sim.embeddings import DEFAULT_MODEL_ID, build_persona_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SigLIP text embeddings for persona descriptions.")
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS_DIR / "persona_cells.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face model downloads. By default only cached files are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_persona_embeddings(
        personas_path=args.personas,
        output_dir=args.output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device_name=args.device,
        allow_download=args.allow_download,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
