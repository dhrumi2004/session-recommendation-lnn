import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from src.models import GRURecommender, PopularityRecommender


class PopularityRecommenderTests(unittest.TestCase):
    def test_ranks_training_interaction_frequencies_and_excludes_padding(self) -> None:
        model = PopularityRecommender(num_items=5)
        model.fit([[1, 2, 1], [2, 3]])

        item_ids, scores = model.top_k(3)

        self.assertEqual(item_ids.tolist(), [1, 2, 3])
        self.assertEqual(scores.tolist(), [2.0, 2.0, 1.0])
        self.assertNotIn(0, item_ids.tolist())

    def test_rejects_empty_training_sessions_and_invalid_item_ids(self) -> None:
        model = PopularityRecommender(num_items=3)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model.fit([])
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            model.fit([[0]])

    def test_round_trips_baseline_state(self) -> None:
        original = PopularityRecommender(num_items=4).fit([[1, 2, 1]])

        restored = PopularityRecommender.from_state_dict(original.state_dict())

        self.assertTrue(torch.equal(original.score(), restored.score()))


class GRURecommenderTests(unittest.TestCase):
    def test_forward_handles_right_padded_sequences_and_masks_padding(self) -> None:
        model = GRURecommender(num_items=8, embedding_dim=6, hidden_size=7)
        inputs = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)

        scores = model(inputs)

        self.assertEqual(tuple(scores.shape), (2, 8))
        self.assertTrue(torch.all(scores[:, 0] < -1e20))
        self.assertTrue(torch.isfinite(scores[:, 1:]).all())

    def test_cross_entropy_backpropagates_into_embedding_and_recurrent_weights(self) -> None:
        model = GRURecommender(num_items=8, embedding_dim=6, hidden_size=7)
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
        targets = torch.tensor([3, 6], dtype=torch.long)

        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(model.gru.weight_ih_l0.grad)
        self.assertGreater(model.gru.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_rejects_empty_and_non_right_padded_sequences(self) -> None:
        model = GRURecommender(num_items=5)
        with self.assertRaisesRegex(ValueError, "at least one item"):
            model(torch.tensor([[0, 0]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "right padding"):
            model(torch.tensor([[1, 0, 2]], dtype=torch.long))

    def test_checkpoint_round_trip_preserves_scores_and_configuration(self) -> None:
        torch.manual_seed(11)
        model = GRURecommender(num_items=9, embedding_dim=5, hidden_size=7)
        model.eval()
        inputs = torch.tensor([[1, 2, 0], [5, 0, 0]], dtype=torch.long)
        original_scores = model(inputs)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoints" / "gru.pt"
            model.save_checkpoint(path)
            restored = GRURecommender.from_checkpoint(path)

        restored.eval()
        restored_scores = restored(inputs)
        self.assertEqual(restored.num_items, 9)
        self.assertEqual(restored.embedding_dim, 5)
        self.assertEqual(restored.hidden_size, 7)
        self.assertTrue(torch.equal(original_scores, restored_scores))


if __name__ == "__main__":
    unittest.main()
