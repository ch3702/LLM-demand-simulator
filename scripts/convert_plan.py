"""플랜 .pkl → .json 변환 (LLM 시뮬 입력 JSON화)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand_sim.data import load_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 시뮬 플랜 pkl→json 변환")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".json")
    frame = load_plan(args.input)
    frame.to_json(output, orient="records", force_ascii=False, indent=1)
    print(f"플랜 변환 → {output}  ({len(frame)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
