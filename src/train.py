"""Train the GRU or LTC recommender using validation-loss early stopping."""

from __future__ import annotations

import argparse
import json
import random
from array import array
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models import GRURecommender, LTCRecommender


class PrefixDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Compact in-memory numeric representation of JSONL prefix examples."""

    def __init__(self, path: str | Path, max_sequence_length: int = 50) -> None:
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.item_values = array("I")
        self.timespan_values = array("f")
        self.offsets = array("Q", [0])
        self.targets = array("I")

        with Path(path).open("r", encoding="utf-8") as examples_file:
            for line_number, line in enumerate(examples_file, start=1):
                row: dict[str, Any] = json.loads(line)
                item_ids = row["input_ids"]
                timespans = row.get("timespans")
                if not isinstance(item_ids, list) or not item_ids:
                    raise ValueError(
                        f"{path}:{line_number} must contain a non-empty input_ids list"
                    )
                if not isinstance(timespans, list) or len(timespans) != len(item_ids):
                    raise ValueError(
                        f"{path}:{line_number} must contain one timespan per input item"
                    )
                item_ids = item_ids[-max_sequence_length:]
                timespans = timespans[-max_sequence_length:]
                if any(int(item_id) <= 0 for item_id in item_ids):
                    raise ValueError(
                        f"{path}:{line_number} contains padding or invalid item IDs"
                    )
                if any(not 0 <= float(span) < float("inf") for span in timespans):
                    raise ValueError(
                        f"{path}:{line_number} contains a non-finite or negative timespan"
                    )
                self.item_values.extend(int(item_id) for item_id in item_ids)
                self.timespan_values.extend(float(span) for span in timespans)
                self.offsets.append(len(self.item_values))
                target_id = int(row["target_id"])
                if target_id <= 0:
                    raise ValueError(f"{path}:{line_number} has an invalid target ID")
                self.targets.append(target_id)

        if not self.targets:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        start, end = self.offsets[index], self.offsets[index + 1]
        item_ids = torch.tensor(self.item_values[start:end], dtype=torch.long)
        timespans = torch.tensor(self.timespan_values[start:end], dtype=torch.float32)
        return item_ids, timespans, self.targets[index]


def collate_prefixes(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad variable-length prefixes and elapsed-time vectors."""
    item_sequences, time_sequences, targets = zip(*batch)
    return (
        pad_sequence(item_sequences, batch_first=True, padding_value=0),
        pad_sequence(time_sequences, batch_first=True, padding_value=0.0),
        torch.tensor(targets, dtype=torch.long),
    )


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but MPS is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loss_for_batch(
    model: torch.nn.Module,
    model_name: str,
    item_ids: torch.Tensor,
    timespans: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    scores = (
        model(item_ids, timespans)
        if model_name == "ltc"
        else model(item_ids)
    )
    return F.cross_entropy(scores, targets)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model_name: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    example_count = 0
    progress = tqdm(loader, desc=f"{phase} {epoch}", leave=False)
    with torch.set_grad_enabled(training):
        for item_ids, timespans, targets in progress:
            item_ids = item_ids.to(device)
            timespans = timespans.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(model, model_name, item_ids, timespans, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            current_count = targets.shape[0]
            total_loss += loss.detach().item() * current_count
            example_count += current_count
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
    if example_count == 0:
        raise ValueError(f"{phase} data loader returned no examples")
    return total_loss / example_count


def train_model(
    *,
    model_name: str,
    data_dir: str | Path,
    checkpoint_path: str | Path,
    history_path: str | Path,
    epochs: int = 3,
    batch_size: int = 256,
    embedding_dim: int = 32,
    hidden_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 2,
    max_sequence_length: int = 50,
    seed: int = 42,
    requested_device: str = "auto",
    num_workers: int = 0,
) -> dict[str, Any]:
    """Fit one sequential model and save its best validation-loss checkpoint."""
    if model_name not in {"gru", "ltc"}:
        raise ValueError("model_name must be 'gru' or 'ltc'")
    if epochs < 1 or batch_size < 1 or patience < 1 or num_workers < 0:
        raise ValueError("epochs, batch_size, patience must be positive and workers non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)

    root = Path(data_dir)
    mapping_path = root / "item_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    num_items = len(mapping["item_to_index"]) + 1
    train_data = PrefixDataset(
        root / "train_examples.jsonl", max_sequence_length=max_sequence_length
    )
    validation_data = PrefixDataset(
        root / "validation_examples.jsonl", max_sequence_length=max_sequence_length
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_prefixes,
        generator=generator,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_prefixes,
        pin_memory=False,
    )

    model_class = GRURecommender if model_name == "gru" else LTCRecommender
    model = model_class(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
    )
    device = _select_device(requested_device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    epochs_without_improvement = 0
    history: dict[str, Any] = {
        "model": model_name,
        "device": str(device),
        "seed": seed,
        "config": {
            "num_items": num_items,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "max_sequence_length": max_sequence_length,
            "num_workers": num_workers,
        },
        "train_loss": [],
        "validation_loss": [],
        "best_epoch": None,
    }

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            model_name,
            device,
            optimizer,
            epoch,
            "train",
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            model_name,
            device,
            None,
            epoch,
            "validation",
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            history["best_epoch"] = epoch
            model.save_checkpoint(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping: validation loss did not improve.")
                break

    history["best_validation_loss"] = best_loss
    history["epochs_completed"] = len(history["train_loss"])
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gru", "ltc"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or Path(
        f"artifacts/checkpoints/{args.model}.pt"
    )
    history_path = args.history or Path(
        f"artifacts/checkpoints/{args.model}_training.json"
    )
    result = train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
        requested_device=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
