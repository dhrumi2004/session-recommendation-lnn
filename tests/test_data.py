import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import (
    PreprocessConfig,
    construct_sessions,
    load_events_csv,
    preprocess_events,
    save_preprocessed,
)


class DataPipelineTests(unittest.TestCase):
    def test_sessions_are_sorted_and_split_after_inactivity_gap(self) -> None:
        events = pd.DataFrame(
            [
                {"timestamp": 2_000, "visitorid": "v1", "event": "view", "itemid": "b"},
                {"timestamp": 0, "visitorid": "v1", "event": "view", "itemid": "a"},
                {
                    "timestamp": 2_000 + 31 * 60 * 1000,
                    "visitorid": "v1",
                    "event": "view",
                    "itemid": "c",
                },
            ]
        )

        sessions = construct_sessions(events, session_gap_minutes=30)

        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0].item_ids, ("a", "b"))
        self.assertEqual(sessions[1].item_ids, ("c",))

    def test_split_is_chronological_and_keeps_sessions_whole(self) -> None:
        rows = []
        for visitor_number in range(10):
            start = visitor_number * 100_000
            rows.extend(
                [
                    {
                        "timestamp": start + offset,
                        "visitorid": f"v{visitor_number}",
                        "event": "view",
                        "itemid": item_id,
                    }
                    for offset, item_id in enumerate(("x", "y", "x"))
                ]
            )
        examples, item_to_index, report = preprocess_events(
            pd.DataFrame(rows),
            PreprocessConfig(min_item_support=1, train_fraction=0.8),
        )

        session_ids = {
            split: {example.session_id for example in split_examples}
            for split, split_examples in examples.items()
        }
        self.assertFalse(session_ids["train"] & session_ids["validation"])
        self.assertFalse(session_ids["train"] & session_ids["test"])
        self.assertFalse(session_ids["validation"] & session_ids["test"])
        self.assertEqual(set(item_to_index), {"x", "y"})
        self.assertEqual(report["test_targets_unavailable_in_training_vocabulary"], 0)
        self.assertTrue(all(example.input_ids for example in examples["train"]))
        example = examples["train"][0]
        self.assertEqual(len(example.timespans), len(example.input_ids))
        self.assertTrue(all(span >= 0 for span in example.timespans))

    def test_unknown_test_targets_are_counted_and_not_mapped(self) -> None:
        rows = []
        for visitor_number in range(10):
            start = visitor_number * 100_000
            target_item = "rare-test-item" if visitor_number == 9 else "common"
            rows.extend(
                [
                    {
                        "timestamp": start + position,
                        "visitorid": f"v{visitor_number}",
                        "event": "view",
                        "itemid": item_id,
                    }
                    for position, item_id in enumerate(
                        ("common", target_item, "common")
                    )
                ]
            )
        examples, item_to_index, report = preprocess_events(
            pd.DataFrame(rows),
            PreprocessConfig(min_item_support=1, train_fraction=0.8),
        )

        self.assertNotIn("rare-test-item", item_to_index)
        self.assertEqual(report["test_targets_unavailable_in_training_vocabulary"], 1)
        self.assertTrue(
            all(
                example.target_id in set(item_to_index.values())
                for example in examples["test"]
            )
        )

    def test_elapsed_features_use_only_observed_history_timestamps(self) -> None:
        rows = [
            {
                "timestamp": visitor_number * 1_000_000 + offset,
                "visitorid": f"v{visitor_number}",
                "event": "view",
                "itemid": item_id,
            }
            for visitor_number in range(10)
            for offset, item_id in ((0, "x"), (1_000, "y"), (100_000, "z"))
        ]
        examples, _, _ = preprocess_events(
            pd.DataFrame(rows),
            PreprocessConfig(min_item_support=1, train_fraction=0.8),
        )
        two_item_prefix = next(
            example
            for example in examples["train"]
            if len(example.input_ids) == 2
        )

        self.assertEqual(two_item_prefix.timespans, (1.0, 1.0))

    def test_csv_loader_drops_invalid_rows_and_checks_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "events.csv"
            csv_path.write_text(
                "timestamp,visitorid,event,itemid\n"
                "1000,1,view,10\n"
                "invalid,2,view,11\n"
                "2000,,view,12\n",
                encoding="utf-8",
            )
            events, report = load_events_csv(csv_path)
            self.assertEqual(len(events), 1)
            self.assertEqual(report["invalid_rows_removed"], 2)

            csv_path.write_text("timestamp,visitorid,itemid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event"):
                load_events_csv(csv_path)

    def test_processed_files_can_be_written(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "timestamp": visitor_number * 100_000 + position,
                    "visitorid": f"v{visitor_number}",
                    "event": "view",
                    "itemid": item_id,
                }
                for visitor_number in range(10)
                for position, item_id in enumerate(("x", "y", "x"))
            ]
        )
        examples, mapping, report = preprocess_events(
            events, PreprocessConfig(min_item_support=1)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            save_preprocessed(temp_dir, examples, mapping, report)
            self.assertTrue((Path(temp_dir) / "train_examples.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "item_mapping.json").exists())
            self.assertTrue((Path(temp_dir) / "preprocessing_report.json").exists())


if __name__ == "__main__":
    unittest.main()
