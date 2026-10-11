import unittest

import torch

from src.models import GRURecommender, LTCRecommender, PopularityRecommender
from src.recommend import recommend_items


class RecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = {"100": 1, "200": 2, "300": 3, "400": 4, "500": 5, "600": 6}

    def test_gru_returns_top_five_valid_original_ids_and_probabilities(self) -> None:
        torch.manual_seed(4)
        model = GRURecommender(num_items=7, embedding_dim=4, hidden_size=5)

        recommendations, unknown = recommend_items(
            item_ids=["100", "200"],
            item_to_index=self.mapping,
            model=model,
            model_name="gru",
            device=torch.device("cpu"),
        )

        self.assertEqual(len(recommendations), 5)
        self.assertEqual(unknown, [])
        self.assertEqual(len({entry["item_id"] for entry in recommendations}), 5)
        self.assertTrue(
            all(entry["item_id"] in self.mapping for entry in recommendations)
        )
        self.assertLessEqual(sum(entry["score"] for entry in recommendations), 1.0)

    def test_popularity_uses_saved_training_counts_and_normalized_scores(self) -> None:
        model = PopularityRecommender(num_items=7).fit([[1, 1, 2], [2, 3]])

        recommendations, _ = recommend_items(
            item_ids=["100"],
            item_to_index=self.mapping,
            model=model,
            model_name="popularity",
            device=torch.device("cpu"),
        )

        self.assertEqual(
            [entry["item_id"] for entry in recommendations[:3]],
            ["100", "200", "300"],
        )
        self.assertAlmostEqual(sum(entry["score"] for entry in recommendations), 1.0)

    def test_ltc_accepts_real_timestamps_for_known_history(self) -> None:
        model = LTCRecommender(
            num_items=7, embedding_dim=4, hidden_size=8, ode_unfolds=2
        )

        recommendations, unknown = recommend_items(
            item_ids=["100", "200"],
            item_to_index=self.mapping,
            model=model,
            model_name="ltc",
            device=torch.device("cpu"),
            timestamps_ms=[1_000, 3_500],
        )

        self.assertEqual(len(recommendations), 5)
        self.assertEqual(unknown, [])
        self.assertTrue(all(entry["score"] >= 0 for entry in recommendations))

    def test_unknown_ids_are_reported_and_known_items_still_work(self) -> None:
        model = PopularityRecommender(num_items=7).fit([[1, 2, 3, 4, 5, 6]])

        recommendations, unknown = recommend_items(
            item_ids=["999", "100"],
            item_to_index=self.mapping,
            model=model,
            model_name="popularity",
            device=torch.device("cpu"),
        )

        self.assertEqual(unknown, ["999"])
        self.assertEqual(len(recommendations), 5)

    def test_unknown_only_session_and_invalid_timestamps_fail_clearly(self) -> None:
        model = PopularityRecommender(num_items=7).fit([[1, 2, 3, 4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "none of the supplied"):
            recommend_items(
                item_ids=["999"],
                item_to_index=self.mapping,
                model=model,
                model_name="popularity",
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "chronological order"):
            recommend_items(
                item_ids=["100", "200"],
                item_to_index=self.mapping,
                model=model,
                model_name="popularity",
                device=torch.device("cpu"),
                timestamps_ms=[2000, 1000],
            )


if __name__ == "__main__":
    unittest.main()
