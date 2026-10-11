"""GRU sequence model for next-item recommendation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class GRURecommender(nn.Module):
    """Predict the next item from the final valid hidden state of a session.

    ``num_items`` includes padding index 0 and all train-vocabulary indices.
    Inputs are right-padded integer tensors shaped ``[batch, sequence_length]``.
    """

    checkpoint_version = 1

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if num_items < 2:
            raise ValueError("num_items must include padding and at least one item")
        if embedding_dim < 1 or hidden_size < 1:
            raise ValueError("embedding_dim and hidden_size must be positive")
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=num_items, embedding_dim=embedding_dim, padding_idx=0
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, num_items)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence_length]")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids cannot have an empty batch or sequence")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer item indices")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.num_items):
            raise ValueError("input_ids contains an index outside the item vocabulary")

        valid = input_ids.ne(0)
        lengths = valid.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("every input sequence must contain at least one item")
        expected_valid = (
            torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if torch.any(valid != expected_valid):
            raise ValueError("input sequences must use right padding without gaps")

        embeddings = self.embedding(input_ids)
        sequence_output, _ = self.gru(embeddings)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        final_indices = lengths - 1
        final_hidden = sequence_output[batch_indices, final_indices]
        scores = self.output(final_hidden)
        return scores.masked_fill(
            torch.arange(self.num_items, device=input_ids.device).eq(0),
            torch.finfo(scores.dtype).min,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights and the constructor configuration."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": self.checkpoint_version,
                "config": {
                    "num_items": self.num_items,
                    "embedding_dim": self.embedding_dim,
                    "hidden_size": self.hidden_size,
                },
                "state_dict": self.state_dict(),
            },
            checkpoint_path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "GRURecommender":
        """Load a model checkpoint created by :meth:`save_checkpoint`."""
        checkpoint: Any = torch.load(
            path, map_location=map_location, weights_only=True
        )
        if checkpoint.get("format_version") != cls.checkpoint_version:
            raise ValueError("Unsupported GRU checkpoint format version")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
