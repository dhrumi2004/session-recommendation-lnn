"""Training-session item-frequency baseline."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import torch


class PopularityRecommender:
    """Rank items by their occurrence count in training sessions.

    The item vocabulary reserves index 0 for padding. ``num_items`` includes
    that padding index, while input sessions should contain valid indices only.
    """

    def __init__(self, num_items: int) -> None:
        if num_items < 2:
            raise ValueError("num_items must include padding and at least one item")
        self.num_items = num_items
        self.item_counts = torch.zeros(num_items, dtype=torch.float64)
        self.is_fitted = False

    def fit(self, sessions: Iterable[Sequence[int]]) -> "PopularityRecommender":
        """Count each occurrence in the supplied training sessions."""
        counts: Counter[int] = Counter()
        session_count = 0
        for session in sessions:
            session_count += 1
            for item_id in session:
                item_index = int(item_id)
                if item_index <= 0 or item_index >= self.num_items:
                    raise ValueError(
                        f"training item index {item_index} is outside the vocabulary"
                    )
                counts[item_index] += 1
        if session_count == 0 or not counts:
            raise ValueError("at least one non-empty training session is required")
        self.item_counts.zero_()
        for item_index, count in counts.items():
            self.item_counts[item_index] = count
        self.is_fitted = True
        return self

    def score(self) -> torch.Tensor:
        """Return one global score per item, with padding excluded."""
        if not self.is_fitted:
            raise RuntimeError("fit must be called before scoring")
        scores = self.item_counts.clone()
        scores[0] = torch.finfo(scores.dtype).min
        return scores

    def top_k(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return item indices and occurrence counts in descending order."""
        if k < 1 or k > self.num_items - 1:
            raise ValueError("k must be between 1 and the number of valid item IDs")
        scores = self.score()
        item_indices = torch.argsort(scores, descending=True, stable=True)[:k]
        top_scores = scores[item_indices]
        return item_indices, top_scores

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """Return serializable baseline state."""
        if not self.is_fitted:
            raise RuntimeError("fit must be called before saving state")
        return {"num_items": self.num_items, "item_counts": self.item_counts.clone()}

    @classmethod
    def from_state_dict(
        cls, state: dict[str, torch.Tensor | int]
    ) -> "PopularityRecommender":
        model = cls(num_items=int(state["num_items"]))
        item_counts = state["item_counts"]
        if not isinstance(item_counts, torch.Tensor):
            raise TypeError("item_counts must be a tensor")
        if item_counts.shape != (model.num_items,):
            raise ValueError("item_counts shape does not match num_items")
        if torch.any(item_counts < 0) or item_counts[0] != 0:
            raise ValueError("item_counts contains invalid values")
        model.item_counts = item_counts.to(dtype=torch.float64).clone()
        model.is_fitted = bool(torch.any(model.item_counts[1:] > 0))
        if not model.is_fitted:
            raise ValueError("baseline state contains no fitted item counts")
        return model
