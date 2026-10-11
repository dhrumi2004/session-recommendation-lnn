"""Evaluate popularity, GRU, and LTC recommenders on the same test examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import GRURecommender, LTCRecommender, PopularityRecommender
from src.train import PrefixDataset, _select_device, collate_prefixes


METRIC_CUTOFFS = (5, 10)


def metrics_from_topk(
    ranked_items: torch.Tensor,
    targets: torch.Tensor,
    cutoffs: tuple[int, ...] = METRIC_CUTOFFS,
) -> dict[str, float]:
    """Calculate Recall@K and MRR@K for one relevant item per example."""
    if ranked_items.ndim != 2:
        raise ValueError("ranked_items must have shape [examples, k]")
    if targets.ndim != 1 or targets.shape[0] != ranked_items.shape[0]:
        raise ValueError("targets must contain one ID per ranked_items row")
    if ranked_items.shape[0] == 0:
        raise ValueError("at least one ranked example is required")
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive")
    if max(cutoffs) > ranked_items.shape[1]:
        raise ValueError("ranked_items does not include the largest cutoff")

    matches = ranked_items.eq(targets.unsqueeze(1))
    ranks = torch.where(
        matches.any(dim=1),
        matches.to(torch.int64).argmax(dim=1) + 1,
        torch.zeros(targets.shape[0], dtype=torch.int64, device=targets.device),
    )
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        hits = ranks.gt(0) & ranks.le(cutoff)
        result[f"Recall@{cutoff}"] = hits.float().mean().item()
        reciprocal_ranks = torch.where(
            hits, ranks.float().reciprocal(), torch.zeros_like(ranks, dtype=torch.float32)
        )
        result[f"MRR@{cutoff}"] = reciprocal_ranks.mean().item()
    return result


def _merge_metric_totals(
    total: dict[str, float], batch_metrics: dict[str, float], count: int
) -> None:
    for name, value in batch_metrics.items():
        total[name] = total.get(name, 0.0) + value * count


def _iter_training_interactions(path: str | Path) -> Iterable[list[int]]:
    """Yield each mapped training-session interaction exactly once.

    Each session's first item appears only in its prefixes; all subsequent
    items appear once as targets. Counting the first input once per session and
    every target therefore recovers training occurrence popularity.
    """
    seen_sessions: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as examples_file:
        for line in examples_file:
            row = json.loads(line)
            session_id = row["session_id"]
            if session_id not in seen_sessions:
                seen_sessions.add(session_id)
                yield [int(row["input_ids"][0])]
            yield [int(row["target_id"])]


def fit_popularity_from_examples(
    path: str | Path, num_items: int
) -> PopularityRecommender:
    """Fit occurrence popularity without counting prefix duplicates."""
    return PopularityRecommender(num_items).fit(_iter_training_interactions(path))


def _evaluate_model(
    model: torch.nn.Module,
    model_name: str,
    dataset: PrefixDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prefixes,
        num_workers=0,
    )
    if not isinstance(model, PopularityRecommender):
        model.eval()
    popularity_top_items: torch.Tensor | None = None
    if isinstance(model, PopularityRecommender):
        popularity_top_items = torch.argsort(
            model.score(), descending=True, stable=True
        )[: max(METRIC_CUTOFFS)].to(device)
    metric_totals: dict[str, float] = {}
    evaluated = 0
    with torch.inference_mode():
        for item_ids, timespans, targets in tqdm(
            loader, desc=f"test {model_name}", leave=False
        ):
            item_ids = item_ids.to(device)
            timespans = timespans.to(device)
            targets = targets.to(device)
            if popularity_top_items is not None:
                ranked_items = popularity_top_items.unsqueeze(0).expand(
                    item_ids.shape[0], -1
                )
            elif model_name == "ltc":
                scores = model(item_ids, timespans)
                ranked_items = scores.topk(max(METRIC_CUTOFFS), dim=1).indices
            else:
                scores = model(item_ids)
                ranked_items = scores.topk(max(METRIC_CUTOFFS), dim=1).indices
            batch_metrics = metrics_from_topk(ranked_items, targets)
            count = targets.shape[0]
            _merge_metric_totals(metric_totals, batch_metrics, count)
            evaluated += count
    if evaluated == 0:
        raise ValueError("test split contains no mapped examples")
    return {
        "metrics": {name: value / evaluated for name, value in metric_totals.items()},
        "evaluated_examples": evaluated,
    }


def plot_training_losses(histories: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Plot actual training and validation histories for trained models."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axis.plot(epochs, history["train_loss"], marker="o", label=f"{model_name} train")
        axis.plot(
            epochs,
            history["validation_loss"],
            marker="s",
            linestyle="--",
            label=f"{model_name} validation",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Training and validation loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_test_metrics(results: dict[str, Any], output_path: Path) -> None:
    """Plot all four held-out ranking metrics for every recommender."""
    model_names = list(results["models"])
    metric_names = [
        "Recall@5",
        "Recall@10",
        "MRR@5",
        "MRR@10",
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = torch.arange(len(metric_names), dtype=torch.float32).numpy()
    bar_width = 0.8 / len(model_names)
    for model_index, model_name in enumerate(model_names):
        values = [
            results["models"][model_name]["metrics"][metric]
            for metric in metric_names
        ]
        offset = (model_index - (len(model_names) - 1) / 2) * bar_width
        axis.bar(positions + offset, values, bar_width, label=model_name)
    axis.set_xticks(positions, metric_names)
    axis.set_ylabel("Score")
    axis.set_title("Held-out next-item ranking metrics")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def evaluate_all(
    *,
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    batch_size: int = 128,
    max_sequence_length: int = 10,
) -> dict[str, Any]:
    """Evaluate all models on the same catalog, cutoff, and held-out examples."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(data_dir)
    checkpoints = Path(checkpoint_dir)
    output = Path(output_dir)
    mapping = json.loads((root / "item_mapping.json").read_text(encoding="utf-8"))
    num_items = len(mapping["item_to_index"]) + 1
    test_data = PrefixDataset(
        root / "test_examples.jsonl", max_sequence_length=max_sequence_length
    )
    device = _select_device(device_name)

    popularity = fit_popularity_from_examples(
        root / "train_examples.jsonl", num_items
    )
    popularity_state = {
        "num_items": popularity.num_items,
        "item_counts": popularity.item_counts,
    }
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(popularity_state, checkpoints / "popularity.pt")

    models: dict[str, torch.nn.Module] = {
        "Popularity": popularity,
        "GRU": GRURecommender.from_checkpoint(
            checkpoints / "gru.pt", map_location="cpu"
        ).to(device),
        "LTC": LTCRecommender.from_checkpoint(
            checkpoints / "ltc.pt", map_location="cpu"
        ).to(device),
    }
    results: dict[str, Any] = {
        "dataset": "RetailRocket events.csv",
        "device": str(device),
        "candidate_universe_size": num_items - 1,
        "ranking_cutoffs": list(METRIC_CUTOFFS),
        "max_sequence_length": max_sequence_length,
        "evaluated_examples": len(test_data),
        "excluded_unavailable_test_targets": json.loads(
            (root / "preprocessing_report.json").read_text(encoding="utf-8")
        )["test_targets_unavailable_in_training_vocabulary"],
        "target_policy": (
            "Only test targets in the training-fitted item vocabulary are "
            "evaluated; unknown targets are counted in preprocessing report."
        ),
        "models": {},
    }
    for model_name, model in models.items():
        results["models"][model_name] = _evaluate_model(
            model, model_name.lower(), test_data, device, batch_size
        )

    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "test_metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    histories: dict[str, dict[str, Any]] = {}
    for model_name in ("gru", "ltc"):
        history_path = checkpoints / f"{model_name}_training.json"
        histories[model_name.upper()] = json.loads(
            history_path.read_text(encoding="utf-8")
        )
    plot_training_losses(histories, output / "training_validation_loss.png")
    plot_test_metrics(results, output / "test_metrics_comparison.png")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=10)
    args = parser.parse_args()
    result = evaluate_all(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
