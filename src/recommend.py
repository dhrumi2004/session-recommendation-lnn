"""Recommend the next items for a supplied RetailRocket item-ID sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from src.models import GRURecommender, LTCRecommender, PopularityRecommender
from src.train import _select_device


def recommend_items(
    *,
    item_ids: Sequence[str],
    item_to_index: dict[str, int],
    model: torch.nn.Module | PopularityRecommender,
    model_name: str,
    device: torch.device,
    timestamps_ms: Sequence[int] | None = None,
    top_k: int = 5,
    max_sequence_length: int = 10,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate ranked recommendations and report unknown input IDs.

    Scores are softmax probabilities over the complete training vocabulary for
    neural models, or normalized training occurrence counts for popularity.
    """
    if model_name not in {"popularity", "gru", "ltc"}:
        raise ValueError("model_name must be popularity, gru, or ltc")
    if not item_ids:
        raise ValueError("provide at least one input item ID")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    if timestamps_ms is not None:
        if len(timestamps_ms) != len(item_ids):
            raise ValueError("timestamps must have exactly one value per input item")
        if any(timestamp < 0 for timestamp in timestamps_ms):
            raise ValueError("timestamps must be non-negative Unix milliseconds")
        if any(
            current < previous
            for previous, current in zip(timestamps_ms, timestamps_ms[1:])
        ):
            raise ValueError("timestamps must be in chronological order")

    known_pairs: list[tuple[int, int | None]] = []
    unknown_items: list[str] = []
    for position, item_id in enumerate(item_ids):
        item_index = item_to_index.get(str(item_id))
        if item_index is None:
            unknown_items.append(str(item_id))
            continue
        timestamp = timestamps_ms[position] if timestamps_ms is not None else None
        known_pairs.append((item_index, timestamp))
    if not known_pairs:
        raise ValueError("none of the supplied item IDs exist in the training vocabulary")

    known_pairs = known_pairs[-max_sequence_length:]
    encoded_items = [item_index for item_index, _ in known_pairs]
    if isinstance(model, PopularityRecommender):
        scores = model.score()
        scores[0] = 0
        scores /= scores.sum()
    else:
        model.eval()
        input_tensor = torch.tensor(
            [encoded_items], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            if model_name == "ltc":
                if timestamps_ms is None:
                    elapsed_seconds = [1.0] * len(known_pairs)
                else:
                    known_timestamps = [
                        timestamp for _, timestamp in known_pairs
                    ]
                    elapsed_seconds = [
                        1.0
                        if position == 0
                        else (int(known_timestamps[position]) - int(known_timestamps[position - 1]))
                        / 1000.0
                        for position in range(len(known_timestamps))
                    ]
                time_tensor = torch.tensor(
                    [elapsed_seconds], dtype=torch.float32, device=device
                )
                logits = model(input_tensor, time_tensor)
            else:
                logits = model(input_tensor)
            scores = torch.softmax(logits[0], dim=0).cpu()

    scores = scores.clone().float()
    scores[0] = -torch.inf
    if top_k > len(item_to_index):
        raise ValueError(
            f"top_k cannot exceed the training vocabulary size ({len(item_to_index)})"
        )
    recommended_scores, indices = torch.topk(scores, k=top_k)
    index_to_item = {index: original_id for original_id, index in item_to_index.items()}
    recommendations = [
        {
            "item_id": index_to_item[int(index)],
            "score": float(score),
        }
        for index, score in zip(indices.tolist(), recommended_scores.tolist())
    ]
    return recommendations, unknown_items


def _load_model(
    model_name: str,
    checkpoint_dir: Path,
    device: torch.device,
) -> torch.nn.Module | PopularityRecommender:
    if model_name == "gru":
        return GRURecommender.from_checkpoint(
            checkpoint_dir / "gru.pt", map_location="cpu"
        ).to(device)
    if model_name == "ltc":
        return LTCRecommender.from_checkpoint(
            checkpoint_dir / "ltc.pt", map_location="cpu"
        ).to(device)
    state: Any = torch.load(
        checkpoint_dir / "popularity.pt", map_location="cpu", weights_only=True
    )
    return PopularityRecommender.from_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("popularity", "gru", "ltc"), default="ltc"
    )
    parser.add_argument(
        "--items",
        required=True,
        help="Comma-separated original RetailRocket item IDs, in session order.",
    )
    parser.add_argument(
        "--timestamps-ms",
        help=(
            "Optional comma-separated Unix-millisecond timestamps aligned with "
            "--items; required only when real LTC elapsed times are desired."
        ),
    )
    parser.add_argument("--mapping", type=Path, default=Path("data/processed/item_mapping.json"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-sequence-length", type=int, default=10)
    args = parser.parse_args()

    original_ids = [value.strip() for value in args.items.split(",")]
    if any(not value for value in original_ids):
        parser.error("--items must be a comma-separated list of non-empty IDs")
    timestamps: list[int] | None = None
    if args.timestamps_ms is not None:
        try:
            timestamps = [int(value.strip()) for value in args.timestamps_ms.split(",")]
        except ValueError as error:
            parser.error(f"--timestamps-ms must contain integers: {error}")
        if len(timestamps) != len(original_ids):
            parser.error("--timestamps-ms must contain one value per --items entry")

    mapping_data = json.loads(args.mapping.read_text(encoding="utf-8"))
    item_to_index = {
        str(item_id): int(index)
        for item_id, index in mapping_data["item_to_index"].items()
    }
    device = _select_device(args.device)
    model = _load_model(args.model, args.checkpoint_dir, device)
    recommendations, unknown_items = recommend_items(
        item_ids=original_ids,
        item_to_index=item_to_index,
        model=model,
        model_name=args.model,
        device=device,
        timestamps_ms=timestamps,
        top_k=5,
        max_sequence_length=args.max_sequence_length,
    )
    if unknown_items:
        print(
            "Ignored input IDs outside the training vocabulary: "
            + ", ".join(unknown_items),
            file=sys.stderr,
        )
    print(json.dumps({"model": args.model, "recommendations": recommendations}, indent=2))


if __name__ == "__main__":
    main()
