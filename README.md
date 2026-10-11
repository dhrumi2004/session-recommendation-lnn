# Session-Based Recommendation System Using Liquid Neural Networks

This undergraduate project studies next-item recommendation from an anonymous
visitor's recent clickstream session. The planned models are a popularity
baseline, a GRU recommender, and a Liquid Time-Constant (LTC) recommender.
Models are implemented in PyTorch. The LTC variant uses `ncps`.

For a plain-language walkthrough of the project structure, data preparation,
models, commands, evaluation, and limitations, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
For a shareable, printable version, see [PROJECT_OVERVIEW.pdf](PROJECT_OVERVIEW.pdf).
For complete clone, dataset, training, and Terminal instructions for
collaborators, see [FRIEND_SETUP.md](FRIEND_SETUP.md).

## Phase 1 and 2 status

- Python 3.10 project environment and scaffold are ready.
- RetailRocket is the selected dataset. Its Kaggle metadata describes
  `events.csv` fields `timestamp`, `visitorid`, `event`, `itemid`, and optional
  `transactionid`; timestamps are Unix milliseconds. Event types are `view`,
  `addtocart`, and `transaction`.
- Dataset license shown in Kaggle metadata: **CC BY-NC-SA 4.0**. Give
  attribution, use the data non-commercially, and follow the share-alike
  condition for adaptations. Review the dataset's current terms before use:
  [Kaggle dataset page](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
  and [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- The dataset archive was extracted locally. The events file is at
  `data/raw/archive/events.csv`; no item-property or category-tree files are
  used by this session-only experiment.
- `src/data.py` now loads events, creates 30-minute inactivity sessions,
  makes strict chronological splits of complete sessions, fits item support
  and vocabulary on training sessions only, maps unknown items out of held-out
  inputs, counts unavailable validation/test targets, and writes JSONL and
  JSON artifacts. Each example also stores real elapsed seconds between
  observed history events only; the first event uses a 1-second reference
  interval. No target timestamp or target-to-history interval is used as an
  input feature.
- Preprocessing tests use synthetic fixtures strictly as software tests. They
  are not recommendation-quality results.
- Real-data preprocessing completed successfully: 2,756,101 event rows loaded,
  with no invalid rows removed; 382,780 eligible sessions were constructed,
  25 sessions crossing temporal boundaries were discarded, and the training
  vocabulary contains 41,261 items. The pipeline produced 685,268 training,
  66,408 validation, and 61,389 test prefix-target examples. There are 22,105
  validation and 27,137 test target occurrences outside the training
  vocabulary; they are counted and excluded from mapped examples.
- Elapsed-time features use only timestamps from observed prefix items. The
  final history item does not use the unknown target's timestamp or the gap to
  the target.
- Phase 3 implements a training-session frequency baseline and a GRU
  recommender that reads right-padded sequences, selects the final valid hidden
  state, and masks padding from candidate scores. Both models were trained and
  evaluated on the real dataset.
- Phase 4 implements an `ncps.torch.LTC` cell, time-aware training, compact
  variable-length batching, validation-loss early stopping, and model
  checkpoints. The LTC uses an embedding, liquid recurrent state, and a
  linear item-scoring head. Its recurrent cell uses conductance/leak dynamics
  integrated with `ode_unfolds` semi-implicit solver steps; `timespans` are
  elapsed seconds between known session events. Padding steps preserve the
  previous hidden state.
- Both recurrent models were then trained for three epochs on the same
  training split with the same 32-dimensional embeddings, 32-unit recurrent
  state, batch size 512, learning rate 0.001, seed 42, and 10-item history cap.
  The best validation-loss epoch was 3 for both (GRU 8.2526, LTC 8.2152).
- Final evaluation is run on all 61,389 mapped test examples with the full
  41,261-item training vocabulary as the candidate universe. The 27,137 test
  target occurrences not represented in that vocabulary are reported but
  excluded from the mapped-example metrics. Actual metrics and plots are saved
  under `artifacts/figures/`.
- `src/recommend.py` loads the trained model, training vocabulary and
  checkpoints without retraining, maps original item IDs, warns about unknown
  inputs, and returns up to five ranked item IDs with scores.

## Dataset setup

The extracted file is currently at `data/raw/archive/events.csv`. If you
extract a replacement dataset under a different folder, pass the corresponding
path with `--input`.

Run preprocessing from this project directory:

```bash
python -m src.data --input data/raw/archive/events.csv --output data/processed
```

Optional thresholds:

```bash
python -m src.data \
  --input data/raw/archive/events.csv \
  --output data/processed \
  --session-gap-minutes 30 \
  --min-session-length 2 \
  --min-item-support 5
```

The default pipeline treats all event types in `events.csv` as item
interactions. It discards invalid timestamps and missing visitor, event, or
item IDs. Sessions are grouped by visitor and split when the timestamp gap is
greater than 30 minutes. Whole sessions are split chronologically into
training, validation, and test windows. Sessions crossing a temporal boundary
are discarded to prevent their events appearing on both sides of a split.
Item frequency and the item-to-index mapping are derived from training data
only; index 0 is reserved for padding.

Generated artifacts:

- `data/processed/train_examples.jsonl`
- `data/processed/validation_examples.jsonl`
- `data/processed/test_examples.jsonl`
- `data/processed/item_mapping.json`
- `data/processed/preprocessing_report.json`

Each JSONL row has one session prefix, next-item target, and elapsed-time
features between observed prefix events. The report includes counts of
validation and test targets not available in the training vocabulary.

## Development checks

Create the project environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the unit tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The popularity baseline expects one integer-indexed item sequence per training
session and counts each interaction occurrence once. The GRU accepts a batch
of right-padded item-index sequences; index 0 is reserved for padding and is
never returned as a candidate. Its checkpoint stores model dimensions along
with model weights.

Train a sequential model with Apple MPS auto-detection (or CPU fallback):

```bash
.venv/bin/python -m src.train --model ltc
.venv/bin/python -m src.train --model gru
```

For a quick software smoke run or a smaller laptop memory budget, set options
such as `--epochs 1 --batch-size 64 --max-sequence-length 30`. Histories and
best checkpoints are saved under `artifacts/checkpoints/`. The sequence cap
keeps the most recent prefix items. Model training and recommendation metrics
are recorded below.

## Evaluation

`src.evaluate` ranks candidates among all 41,261 items observed in the
training vocabulary; padding ID 0 is masked. The popularity baseline ranks by
training interaction count and breaks count ties by item index. All models use
the same held-out examples, with the same 10-item recent-history cap and
cutoffs of 5 and 10.

- **Recall@K:** fraction of evaluated examples whose one correct next item is
  in the first K recommendations.
- **MRR@K:** average reciprocal rank of the correct item when it appears in
  the first K positions; misses contribute zero.

### Observed test results

These results came from the actual RetailRocket test run after correcting the
LTC time features to use observed history only (61,389 examples):

| Model | Recall@5 | Recall@10 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.0064 | 0.0114 | 0.0044 | 0.0051 |
| GRU | 0.1886 | 0.2155 | 0.1455 | 0.1491 |
| LTC | 0.2432 | 0.2680 | 0.1993 | 0.2027 |

On this split and training setup, LTC scores higher than both baselines on
each reported metric. This is evidence only for this particular experiment;
it is not evidence of universal LTC superiority. The catalog omits items
unseen during training, and target occurrences outside that vocabulary were
excluded. Also, only the most recent 10 items were used, despite longer
sessions existing in the data, and each prefix example is weighted equally.

Artifacts:

- `artifacts/figures/test_metrics.json` — complete values and evaluation
  settings
- `artifacts/figures/training_validation_loss.png` — actual training and
  validation loss curves
- `artifacts/figures/test_metrics_comparison.png` — actual test metric
  comparison chart
- `artifacts/checkpoints/popularity.pt` — training interaction counts
- `artifacts/checkpoints/gru.pt` and `artifacts/checkpoints/ltc.pt` — best
  validation-loss checkpoints

## Command-line recommendations

Run inference from the project directory after the checkpoints and mapping
exist. The default model is LTC:

```bash
.venv/bin/python -m src.recommend --items 461686,257040,309778
```

Choose `--model popularity`, `--model gru`, or `--model ltc` as desired. For
LTC, you may provide timestamps in Unix milliseconds, one per item, to use
observed elapsed time:

```bash
.venv/bin/python -m src.recommend \
  --model ltc \
  --items 461686,257040,309778 \
  --timestamps-ms 1433221332117,1433221333117,1433221335117
```

If timestamps are omitted, LTC uses a 1-second reference interval for each
history item. IDs outside the training vocabulary are reported and skipped;
the command fails clearly if none of the supplied IDs are known. Output is
JSON with up to five original item IDs and normalized scores. The scores are
softmax probabilities for neural models and normalized training interaction
counts for popularity; recommendations are ranked over the training
vocabulary.

Actual CLI smoke checks were run against all three saved models using known
RetailRocket item IDs. LTC was also run with timestamp inputs. These
demonstrations verify loading and inference only; recommendation examples are
not manually inserted into this README.

## Source and usage notes

RetailRocket clickstream data is made available through the Kaggle dataset
maintained by Roman Zykov. The dataset description states that interaction IDs
are hashed and characterizes the records as implicit feedback. Dataset
metadata and schema were checked, and the downloaded event CSV was validated
and preprocessed locally. The larger properties files remain unused.
