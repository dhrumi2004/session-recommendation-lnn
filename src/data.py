"""Load RetailRocket events and create leakage-conscious session examples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "visitorid", "event", "itemid"}


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for timestamp-based session construction and data splitting."""

    session_gap_minutes: int = 30
    min_session_length: int = 2
    min_item_support: int = 5
    train_fraction: float = 0.8
    validation_fraction: float = 0.1

    def validate(self) -> None:
        if self.session_gap_minutes <= 0:
            raise ValueError("session_gap_minutes must be positive")
        if self.min_session_length < 2:
            raise ValueError("min_session_length must be at least 2")
        if self.min_item_support < 1:
            raise ValueError("min_item_support must be positive")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must sum to less than 1")


@dataclass(frozen=True)
class Session:
    session_id: str
    visitor_id: str
    start_timestamp: int
    end_timestamp: int
    item_ids: tuple[str, ...]
    item_timestamps: tuple[int, ...]


@dataclass(frozen=True)
class Example:
    session_id: str
    input_ids: tuple[int, ...]
    target_id: int
    timespans: tuple[float, ...]


def load_events_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read the behavioral events CSV and discard invalid interaction rows."""
    events = pd.read_csv(
        path,
        dtype={"visitorid": "string", "event": "string", "itemid": "string"},
    )
    missing_columns = REQUIRED_COLUMNS.difference(events.columns)
    if missing_columns:
        raise ValueError(
            "Events CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    input_rows = len(events)
    events = events.loc[:, ["timestamp", "visitorid", "event", "itemid"]].copy()
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce")
    for column in ("visitorid", "event", "itemid"):
        events[column] = events[column].str.strip()
    valid = (
        events["timestamp"].notna()
        & events["visitorid"].notna()
        & events["visitorid"].ne("")
        & events["event"].notna()
        & events["event"].ne("")
        & events["itemid"].notna()
        & events["itemid"].ne("")
    )
    events = events.loc[valid].copy()
    events["timestamp"] = events["timestamp"].astype("int64")
    events["_input_order"] = range(len(events))
    return events, {
        "input_rows": input_rows,
        "invalid_rows_removed": input_rows - len(events),
        "valid_event_rows": len(events),
    }


def construct_sessions(
    events: pd.DataFrame, session_gap_minutes: int = 30
) -> list[Session]:
    """Split each visitor's chronologically ordered events after inactivity gaps."""
    if session_gap_minutes <= 0:
        raise ValueError("session_gap_minutes must be positive")
    missing_columns = REQUIRED_COLUMNS.difference(events.columns)
    if missing_columns:
        raise ValueError(
            "Events data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if events.empty:
        return []

    ordered = events.loc[:, ["timestamp", "visitorid", "itemid"]].copy()
    ordered["timestamp"] = pd.to_numeric(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp", "visitorid", "itemid"])
    ordered["visitorid"] = ordered["visitorid"].astype(str).str.strip()
    ordered["itemid"] = ordered["itemid"].astype(str).str.strip()
    ordered = ordered.loc[
        ordered["visitorid"].ne("") & ordered["itemid"].ne("")
    ].copy()
    if ordered.empty:
        return []
    if "_input_order" not in events.columns:
        ordered["_input_order"] = range(len(ordered))
    else:
        ordered["_input_order"] = events.loc[ordered.index, "_input_order"]

    ordered = ordered.sort_values(
        ["visitorid", "timestamp", "_input_order"], kind="mergesort"
    )
    visitor_groups = ordered.groupby("visitorid", sort=False)["timestamp"]
    gap = visitor_groups.diff()
    new_session = gap.isna() | (gap > session_gap_minutes * 60 * 1000)
    ordered["_session_number"] = new_session.groupby(ordered["visitorid"]).cumsum()

    grouped = ordered.groupby(
        ["visitorid", "_session_number"], sort=False, observed=True
    ).agg(
        start_timestamp=("timestamp", "min"),
        end_timestamp=("timestamp", "max"),
        item_ids=("itemid", list),
        item_timestamps=("timestamp", list),
    )
    sessions = [
        Session(
            session_id="",
            visitor_id=str(visitor_id),
            start_timestamp=int(row.start_timestamp),
            end_timestamp=int(row.end_timestamp),
            item_ids=tuple(str(item_id) for item_id in row.item_ids),
            item_timestamps=tuple(int(ts) for ts in row.item_timestamps),
        )
        for visitor_id, row in grouped.iterrows()
    ]
    sessions.sort(
        key=lambda session: (
            session.start_timestamp,
            session.end_timestamp,
            session.visitor_id,
        )
    )
    return [
        replace(session, session_id=f"s{index:08d}")
        for index, session in enumerate(sessions)
    ]


def split_sessions_chronologically(
    sessions: list[Session],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[dict[str, list[Session]], int]:
    """Split complete sessions into non-overlapping chronological time windows.

    Session end times determine the cut points. Any session crossing either
    cut point is discarded so interactions from a session cannot straddle
    splits or place an earlier interaction in a later split.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")
    if len(sessions) < 3:
        raise ValueError("At least three sessions are needed for three-way splitting")

    ordered = sorted(
        sessions,
        key=lambda session: (
            session.end_timestamp,
            session.start_timestamp,
            session.session_id,
        ),
    )
    count = len(ordered)
    train_count = max(1, min(count - 2, int(count * train_fraction)))
    validation_count = max(
        1,
        min(count - train_count - 1, int(count * validation_fraction)),
    )
    first_cut = ordered[train_count - 1].end_timestamp
    second_cut = ordered[train_count + validation_count - 1].end_timestamp

    splits = {"train": [], "validation": [], "test": []}
    discarded = 0
    for session in ordered:
        if session.end_timestamp <= first_cut:
            splits["train"].append(session)
        elif session.start_timestamp > first_cut and session.end_timestamp <= second_cut:
            splits["validation"].append(session)
        elif session.start_timestamp > second_cut:
            splits["test"].append(session)
        else:
            discarded += 1

    empty_splits = [name for name, split in splits.items() if not split]
    if empty_splits:
        raise ValueError(
            "Chronological splitting produced an empty "
            + ", ".join(empty_splits)
            + " split. The dataset may have tied/overlapping session times."
        )
    return splits, discarded


def _filter_training_sessions(
    sessions: list[Session], config: PreprocessConfig
) -> tuple[list[Session], Counter[str]]:
    """Filter infrequent items using only training sessions."""
    current = sessions
    while True:
        frequencies: Counter[str] = Counter(
            item_id for session in current for item_id in session.item_ids
        )
        frequent_items = {
            item_id
            for item_id, frequency in frequencies.items()
            if frequency >= config.min_item_support
        }
        filtered = [
            replace(
                session,
                item_ids=tuple(item_id for item_id in session.item_ids
                               if item_id in frequent_items),
                item_timestamps=tuple(
                    timestamp
                    for item_id, timestamp in zip(
                        session.item_ids, session.item_timestamps
                    )
                    if item_id in frequent_items
                ),
            )
            for session in current
        ]
        filtered = [
            session
            for session in filtered
            if len(session.item_ids) >= config.min_session_length
        ]
        if [session.item_ids for session in filtered] == [
            session.item_ids for session in current
        ]:
            final_frequencies = Counter(
                item_id for session in filtered for item_id in session.item_ids
            )
            return filtered, final_frequencies
        current = filtered
        if not current:
            return [], Counter()


def _make_examples(
    sessions: Iterable[Session],
    item_to_index: dict[str, int],
    min_session_length: int,
) -> tuple[list[Example], int, int]:
    examples: list[Example] = []
    retained_sessions = 0
    unknown_targets = 0
    for session in sessions:
        unknown_targets += sum(
            item_id not in item_to_index for item_id in session.item_ids[1:]
        )
        retained = [
            (item_to_index[item_id], timestamp)
            for item_id, timestamp in zip(
                session.item_ids, session.item_timestamps
            )
            if item_id in item_to_index
        ]
        encoded = tuple(item_id for item_id, _ in retained)
        encoded_timestamps = tuple(timestamp for _, timestamp in retained)
        if len(encoded) < min_session_length:
            continue
        retained_sessions += 1
        for position in range(1, len(encoded)):
            intervals_seconds = tuple(
                1.0
                if index == 0
                else (encoded_timestamps[index] - encoded_timestamps[index - 1])
                / 1000.0
                for index in range(position)
            )
            examples.append(
                Example(
                    session.session_id,
                    encoded[:position],
                    encoded[position],
                    intervals_seconds,
                )
            )
    return examples, retained_sessions, unknown_targets


def preprocess_events(
    events: pd.DataFrame, config: PreprocessConfig | None = None
) -> tuple[dict[str, list[Example]], dict[str, int], dict[str, Any]]:
    """Build training-vocabulary-only examples and a preprocessing report."""
    config = config or PreprocessConfig()
    config.validate()
    sessions = [
        session
        for session in construct_sessions(events, config.session_gap_minutes)
        if len(session.item_ids) >= config.min_session_length
    ]
    if len(sessions) < 3:
        raise ValueError("Fewer than three eligible sessions remain after filtering")

    splits, boundary_discarded = split_sessions_chronologically(
        sessions, config.train_fraction, config.validation_fraction
    )
    train_sessions, train_frequencies = _filter_training_sessions(
        splits["train"], config
    )
    if not train_frequencies:
        raise ValueError(
            "No training items remain; lower min_item_support or "
            "min_session_length."
        )

    ordered_items = sorted(
        train_frequencies,
        key=lambda item_id: (-train_frequencies[item_id], item_id),
    )
    item_to_index = {
        item_id: index for index, item_id in enumerate(ordered_items, start=1)
    }
    examples: dict[str, list[Example]] = {}
    retained_session_counts: dict[str, int] = {}
    unknown_target_counts: dict[str, int] = {"validation": 0, "test": 0}
    examples["train"], retained_session_counts["train"], _ = _make_examples(
        train_sessions, item_to_index, config.min_session_length
    )
    for split_name in ("validation", "test"):
        (
            examples[split_name],
            retained_session_counts[split_name],
            unknown_target_counts[split_name],
        ) = _make_examples(
            splits[split_name], item_to_index, config.min_session_length
        )
    if not examples["train"]:
        raise ValueError("No training examples remain after preprocessing")

    report = {
        "settings": asdict(config),
        "session_count_before_split": len(sessions),
        "sessions_discarded_at_temporal_boundaries": boundary_discarded,
        "sessions_per_split_before_filtering": {
            name: len(split) for name, split in splits.items()
        },
        "sessions_per_split_after_item_mapping": retained_session_counts,
        "examples_per_split": {
            name: len(split_examples) for name, split_examples in examples.items()
        },
        "training_vocabulary_size": len(item_to_index),
        "validation_targets_unavailable_in_training_vocabulary": (
            unknown_target_counts["validation"]
        ),
        "test_targets_unavailable_in_training_vocabulary": unknown_target_counts[
            "test"
        ],
    }
    return examples, item_to_index, report


def save_preprocessed(
    output_dir: str | Path,
    examples: dict[str, list[Example]],
    item_to_index: dict[str, int],
    report: dict[str, Any],
) -> None:
    """Write examples, the training vocabulary, and report as JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, split_examples in examples.items():
        with (output_path / f"{split_name}_examples.jsonl").open(
            "w", encoding="utf-8"
        ) as output_file:
            for example in split_examples:
                json.dump(
                    {
                        "session_id": example.session_id,
                        "input_ids": example.input_ids,
                        "target_id": example.target_id,
                        "timespans": example.timespans,
                    },
                    output_file,
                )
                output_file.write("\n")

    index_to_item = {str(index): item_id for item_id, index in item_to_index.items()}
    with (output_path / "item_mapping.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "padding_index": 0,
                "item_to_index": item_to_index,
                "index_to_item": index_to_item,
            },
            output_file,
            indent=2,
        )
    with (output_path / "preprocessing_report.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(report, output_file, indent=2)


def preprocess_file(
    input_path: str | Path,
    output_dir: str | Path,
    config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    events, input_report = load_events_csv(input_path)
    examples, item_to_index, report = preprocess_events(events, config)
    report = {**input_report, **report}
    save_preprocessed(output_dir, examples, item_to_index, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/events.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--min-item-support", type=int, default=5)
    args = parser.parse_args()
    config = PreprocessConfig(
        session_gap_minutes=args.session_gap_minutes,
        min_session_length=args.min_session_length,
        min_item_support=args.min_item_support,
    )
    print(json.dumps(preprocess_file(args.input, args.output, config), indent=2))


if __name__ == "__main__":
    main()
