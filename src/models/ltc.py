"""Liquid Time-Constant model for time-aware next-item recommendation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ncps.torch import LTC
from torch import nn


class LTCRecommender(nn.Module):
    """Encode an item prefix using an LTC cell and score its final valid state.

    ``timespans`` are elapsed seconds between observed history events. The
    first event uses a 1-second reference interval, since no prior event is
    available. Index 0 is reserved for right-padding; padded steps use zero.
    """

    checkpoint_version = 1

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 32,
        hidden_size: int = 32,
        ode_unfolds: int = 6,
    ) -> None:
        super().__init__()
        if num_items < 2:
            raise ValueError("num_items must include padding and at least one item")
        if embedding_dim < 1 or hidden_size < 1 or ode_unfolds < 1:
            raise ValueError(
                "embedding_dim, hidden_size, and ode_unfolds must be positive"
            )
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.ltc = LTC(
            input_size=embedding_dim,
            units=hidden_size,
            return_sequences=True,
            batch_first=True,
            ode_unfolds=ode_unfolds,
        )
        self.output = nn.Linear(hidden_size, num_items)

    def forward(
        self,
        input_ids: torch.Tensor,
        timespans: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        if timespans is not None:
            if timespans.shape != input_ids.shape:
                raise ValueError("timespans must have the same shape as input_ids")
            if not timespans.is_floating_point():
                raise TypeError("timespans must be floating-point elapsed seconds")
            if not torch.isfinite(timespans).all() or torch.any(timespans < 0):
                raise ValueError("timespans must contain finite, non-negative seconds")
            if torch.any(timespans.masked_select(~valid) != 0):
                raise ValueError("right-padding positions must have zero timespan")
            timespans = timespans.to(dtype=self.embedding.weight.dtype)

        embedded = self.embedding(input_ids)
        if timespans is None:
            timespans = torch.ones_like(input_ids, dtype=embedded.dtype)
        batch_size, sequence_length = input_ids.shape
        hidden = embedded.new_zeros((batch_size, self.ltc.state_size))
        sequence_outputs = []
        for step in range(sequence_length):
            active = valid[:, step]
            elapsed = timespans[:, step : step + 1]
            # ncps' batched LTC.forward squeezes [batch] elapsed times, which
            # cannot broadcast against the cell's [units] parameters. Call
            # its underlying cell with [batch, 1], and retain old state for
            # padding rows. The small floor handles simultaneous events, for
            # which the LTC ODE solver cannot divide by zero elapsed time.
            elapsed = elapsed.clamp_min(1e-3)
            elapsed = torch.where(active.unsqueeze(1), elapsed, torch.ones_like(elapsed))
            output, candidate_hidden = self.ltc.rnn_cell(
                embedded[:, step, :], hidden, elapsed_time=elapsed
            )
            hidden = torch.where(active.unsqueeze(1), candidate_hidden, hidden)
            sequence_outputs.append(output)
        sequence_output = torch.stack(sequence_outputs, dim=1)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        final_hidden = sequence_output[batch_indices, lengths - 1]
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
                    "ode_unfolds": self.ode_unfolds,
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
    ) -> "LTCRecommender":
        """Load a model checkpoint created by :meth:`save_checkpoint`."""
        checkpoint: Any = torch.load(
            path, map_location=map_location, weights_only=True
        )
        if checkpoint.get("format_version") != cls.checkpoint_version:
            raise ValueError("Unsupported LTC checkpoint format version")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
