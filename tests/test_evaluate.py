import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluate import fit_popularity_from_examples, metrics_from_topk


class RankingMetricTests(unittest.TestCase):
    def test_recall_and_mrr_use_one_relevant_target_per_example(self) -> None:
        ranked = torch.tensor(
            [
                [2, 3, 1, 4, 5, 6, 7, 8, 9, 10],
                [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                [5, 6, 1, 2, 3, 4, 7, 8, 9, 10],
            ]
        )
        targets = torch.tensor([2, 10, 1])

        result = metrics_from_topk(ranked, targets)

        self.assertAlmostEqual(result["Recall@5"], 1.0)
        self.assertAlmostEqual(result["Recall@10"], 1.0)
        self.assertAlmostEqual(result["MRR@5"], (1.0 + 0.25 + 1 / 3) / 3)
        self.assertAlmostEqual(result["MRR@10"], (1.0 + 0.25 + 1 / 3) / 3)

    def test_missing_targets_have_zero_recall_and_reciprocal_rank(self) -> None:
        ranked = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        result = metrics_from_topk(ranked, torch.tensor([20]))

        self.assertEqual(result["Recall@5"], 0.0)
        self.assertEqual(result["Recall@10"], 0.0)
        self.assertEqual(result["MRR@5"], 0.0)
        self.assertEqual(result["MRR@10"], 0.0)

    def test_rejects_incompatible_ranking_shapes_and_cutoffs(self) -> None:
        with self.assertRaisesRegex(ValueError, "one ID"):
            metrics_from_topk(torch.ones((2, 10), dtype=torch.long), torch.ones(1, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "largest cutoff"):
            metrics_from_topk(
                torch.ones((1, 5), dtype=torch.long),
                torch.ones(1, dtype=torch.long),
                cutoffs=(10,),
            )


class PopularityCountTests(unittest.TestCase):
    def test_training_examples_recover_each_mapped_interaction_once(self) -> None:
        rows = [
            {"session_id": "s1", "input_ids": [1], "target_id": 2},
            {"session_id": "s1", "input_ids": [1, 2], "target_id": 1},
            {"session_id": "s2", "input_ids": [2], "target_id": 3},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            model = fit_popularity_from_examples(path, num_items=4)

        self.assertEqual(model.item_counts.tolist(), [0.0, 2.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
