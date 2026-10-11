import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from src.models import LTCRecommender
from src.train import PrefixDataset, collate_prefixes


class LTCRecommenderTests(unittest.TestCase):
    def test_forward_uses_elapsed_times_and_excludes_padding(self) -> None:
        torch.manual_seed(4)
        model = LTCRecommender(
            num_items=8, embedding_dim=5, hidden_size=8, ode_unfolds=3
        )
        item_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        timespans = torch.tensor(
            [[2.0, 5.0, 1.0], [3.0, 4.0, 0.0]], dtype=torch.float32
        )

        scores = model(item_ids, timespans)

        self.assertEqual(tuple(scores.shape), (2, 8))
        self.assertTrue(torch.all(scores[:, 0] < -1e20))
        self.assertTrue(torch.isfinite(scores[:, 1:]).all())

    def test_cross_entropy_backpropagates_into_ltc_and_embedding(self) -> None:
        model = LTCRecommender(
            num_items=8, embedding_dim=5, hidden_size=8, ode_unfolds=3
        )
        item_ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
        timespans = torch.tensor([[2.0, 1.0, 0.0], [3.0, 2.0, 1.0]])
        targets = torch.tensor([3, 6], dtype=torch.long)

        loss = F.cross_entropy(model(item_ids, timespans), targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        embedding_gradient = model.embedding.weight.grad
        self.assertIsNotNone(embedding_gradient)
        self.assertGreater(embedding_gradient.abs().sum().item(), 0)
        ltc_gradients = [
            parameter.grad
            for parameter in model.ltc.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(ltc_gradients)
        self.assertGreater(sum(g.abs().sum().item() for g in ltc_gradients), 0)

    def test_invalid_elapsed_time_shape_or_padding_is_rejected(self) -> None:
        model = LTCRecommender(num_items=6, hidden_size=8)
        item_ids = torch.tensor([[1, 2, 0]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "same shape"):
            model(item_ids, torch.ones((1, 2)))
        with self.assertRaisesRegex(ValueError, "padding positions"):
            model(item_ids, torch.tensor([[1.0, 2.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model(item_ids, torch.tensor([[1.0, -2.0, 0.0]]))

    def test_checkpoint_round_trip_preserves_predictions(self) -> None:
        torch.manual_seed(8)
        model = LTCRecommender(
            num_items=8, embedding_dim=5, hidden_size=8, ode_unfolds=3
        )
        model.eval()
        item_ids = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
        timespans = torch.tensor([[2.0, 5.0], [3.0, 0.0]])
        original = model(item_ids, timespans)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "ltc.pt"
            model.save_checkpoint(checkpoint)
            restored = LTCRecommender.from_checkpoint(checkpoint)

        restored.eval()
        self.assertEqual(restored.ode_unfolds, 3)
        self.assertTrue(torch.equal(original, restored(item_ids, timespans)))


class TrainingDataTests(unittest.TestCase):
    def test_prefix_dataset_caps_history_and_collates_time_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "examples.jsonl"
            rows = [
                {
                    "input_ids": [1, 2, 3, 4],
                    "timespans": [1.5, 2.5, 3.5, 4.5],
                    "target_id": 5,
                },
                {
                    "input_ids": [6, 7],
                    "timespans": [6.5, 7.5],
                    "target_id": 8,
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            dataset = PrefixDataset(path, max_sequence_length=3)
            batch = collate_prefixes([dataset[0], dataset[1]])

        item_ids, timespans, targets = batch
        self.assertEqual(item_ids.tolist(), [[2, 3, 4], [6, 7, 0]])
        self.assertEqual(timespans.tolist(), [[2.5, 3.5, 4.5], [6.5, 7.5, 0.0]])
        self.assertEqual(targets.tolist(), [5, 8])


if __name__ == "__main__":
    unittest.main()
