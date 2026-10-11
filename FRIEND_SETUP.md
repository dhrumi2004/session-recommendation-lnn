# Friend's Setup and Run Guide

This guide is for someone who cloned this GitHub repository and wants to
prepare the data, train the models, and run recommendations from Terminal.
The repository contains the source code, but **does not contain the RetailRocket
dataset, processed data, or trained model checkpoints**. Each person must
download the dataset and create those files locally.

## What the project does

Given a sequence of item IDs from one shopping session, the project ranks up to
five possible next items. It compares:

- **Popularity**: ranks by item frequency in the training data.
- **GRU**: a neural network that learns patterns in the sequence of items.
- **LTC**: a liquid time-constant neural network that also supports time gaps
  between known events.

The returned item IDs are anonymous RetailRocket IDs, not product names. Model
scores rank suggestions; they are not guaranteed click or purchase probabilities.

## 1. Requirements

- macOS, Linux, or Windows.
- Python **3.10 or newer**.
- Git.
- Enough free disk space for the downloaded dataset, processed examples, and
  model files. The raw RetailRocket archive is large (about 1.2 GB in the
  original project setup), and preprocessing creates additional local files.
- A compatible NVIDIA GPU is optional. Apple Silicon can use Apple's MPS
  backend; CPU is also supported. `--device auto` selects an available backend.

Training can be slow on CPU. For a computer with limited memory, use a smaller
batch size such as `64` or `128` in both training commands below. This can
change the exact results and will generally take more steps/time per epoch.

## 2. Clone the repository

Open Terminal (macOS/Linux) or PowerShell (Windows), then run:

```text
git clone https://github.com/dhrumi2004/session-recommendation-lnn.git
cd session-recommendation-lnn
```

Run all commands below from this project directory. The `.venv` environment
created in the next step should be **inside this directory**.

## 3. Create the Python environment and install dependencies

### macOS or Linux

Check that Python is 3.10 or newer, then run:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On Apple Silicon, PyTorch may use MPS automatically when available. No CUDA
installation is needed for a Mac.

### Windows PowerShell

Install Python 3.10 or newer if needed, then run:

```powershell
py -3.10 --version
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the `py -3.10` command is unavailable but `python` is installed, use
`python` instead. Keep using the `.venv` Python path for the remaining
commands so packages are installed in and run from the same environment.

## 4. Download and place the dataset

1. Open the [RetailRocket Ecommerce Dataset page on Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
2. Sign in to Kaggle if it asks, download the dataset archive, and extract it.
3. Find the file named `events.csv`. The other dataset files are not needed by
   this project.
4. In the project folder, create `data/raw/archive/` and place `events.csv`
   there. The final path must be:

   ```text
   session-recommendation-lnn/data/raw/archive/events.csv
   ```

   The project does not upload or include the dataset in Git. The Kaggle page
   lists the dataset as **CC BY-NC-SA 4.0**; check and follow its current
   license terms, including attribution and non-commercial use.

## 5. Preprocess the data

### macOS or Linux

```bash
.venv/bin/python -m src.data \
  --input data/raw/archive/events.csv \
  --output data/processed
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m src.data `
  --input data/raw/archive/events.csv `
  --output data/processed
```

This may take a while because the raw file has millions of events. It creates
the train, validation, and test examples, the item-ID mapping, and a
preprocessing report under `data/processed/`. By default it groups events into
sessions using a 30-minute inactivity gap and builds the item vocabulary from
training data only.

If this step reports that the CSV cannot be found, check the exact filename and
folder path above. If the file is somewhere else, change the `--input` path to
its actual location.

## 6. Train the models

These commands use the experiment settings used for the project's reported
comparison: three epochs, batch size 512, 32-dimensional embeddings, hidden
size 32, and a 10-item history cap. Training saves checkpoints locally under
`artifacts/checkpoints/`.

### macOS or Linux

```bash
.venv/bin/python -m src.train --model gru --epochs 3 \
  --batch-size 512 --embedding-dim 32 --hidden-size 32 \
  --max-sequence-length 10 --seed 42 --device auto

.venv/bin/python -m src.train --model ltc --epochs 3 \
  --batch-size 512 --embedding-dim 32 --hidden-size 32 \
  --max-sequence-length 10 --seed 42 --device auto
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m src.train --model gru --epochs 3 `
  --batch-size 512 --embedding-dim 32 --hidden-size 32 `
  --max-sequence-length 10 --seed 42 --device auto

.\.venv\Scripts\python.exe -m src.train --model ltc --epochs 3 `
  --batch-size 512 --embedding-dim 32 --hidden-size 32 `
  --max-sequence-length 10 --seed 42 --device auto
```

Run both commands if you want to compare GRU and LTC. Each prints its progress
and saves its best validation-loss checkpoint. A smaller batch can help with
limited memory; use the same batch size for both models for a fairer comparison.

## 7. Run recommendations in Terminal

Use original RetailRocket item IDs from a single session, in the order the
visitor interacted with them. The example IDs below are a demonstration; replace
them with IDs that are present in this dataset and in the training vocabulary.

### macOS or Linux

```bash
.venv/bin/python -m src.recommend --model ltc \
  --items 461686,257040,309778
```

Compare with GRU:

```bash
.venv/bin/python -m src.recommend --model gru \
  --items 461686,257040,309778
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m src.recommend --model ltc `
  --items 461686,257040,309778

.\.venv\Scripts\python.exe -m src.recommend --model gru `
  --items 461686,257040,309778
```

The output is JSON containing the selected model and up to five recommendations,
ranked highest score first. LTC can also use event timestamps in Unix
milliseconds if you have one real timestamp for each input item. For example:

```bash
.venv/bin/python -m src.recommend --model ltc \
  --items ITEM1,ITEM2,ITEM3 \
  --timestamps-ms TIMESTAMP1,TIMESTAMP2,TIMESTAMP3
```

Use actual timestamp values in chronological order; do not copy the placeholders.
If timestamps are omitted, LTC uses a 1-second reference interval. Unknown input
IDs are reported and skipped; the command gives an error if none of the IDs are
in the training vocabulary. The default is to return up to five items.

## 8. Evaluate and view the metrics

After preprocessing and training **both** GRU and LTC, run:

### macOS or Linux

```bash
.venv/bin/python -m src.evaluate \
  --device auto --batch-size 128 --max-sequence-length 10
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m src.evaluate `
  --device auto --batch-size 128 --max-sequence-length 10
```

Evaluation writes the metrics and plots under `artifacts/figures/`. Recall@K
measures whether the correct next item appears among the first K suggestions.
MRR@K gives more credit when the correct item is ranked nearer the top.

In the original experiment, LTC scored higher than GRU on all four test metrics.
That comparison applies to that particular data split and training setup; it
does not prove LTC is always better. Some held-out target items are not in the
training vocabulary and therefore cannot be recommended or included in the
mapped-item metrics.

Evaluation also creates the popularity baseline checkpoint. You can then
compare it with the same input:

### macOS or Linux

```bash
.venv/bin/python -m src.recommend --model popularity \
  --items 461686,257040,309778
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m src.recommend --model popularity `
  --items 461686,257040,309778
```

## 9. Optional: run the tests

### macOS or Linux

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Common problems

- **`No such file or directory: .venv/...`**: make sure the Terminal's current
  directory is the cloned `session-recommendation-lnn` folder, and create the
  virtual environment by following Step 3.
- **`events.csv` not found**: verify it is at
  `data/raw/archive/events.csv`, or pass its actual path to `--input`.
- **Checkpoint not found when recommending**: finish Step 6 for the model
  selected with `--model`. A cloned repository does not include checkpoints.
- **Unknown item IDs**: use original RetailRocket item IDs, not internal model
  indexes. The project only recognizes IDs found in its training vocabulary.
- **Out of memory or slow training**: set `--batch-size 64` or `128` in both
  training commands. CPU is supported but can take longer.

## Files generated locally

- `data/processed/`: mapped examples, item mapping, and preprocessing report.
- `artifacts/checkpoints/`: trained GRU/LTC weights, histories, and (after
  evaluation) popularity counts.
- `artifacts/figures/`: evaluation JSON and charts.

These generated files and the downloaded dataset are ignored by Git, so they
remain on the friend's computer and do not get pushed to the public repository.
For the full project explanation, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md) and
the [project overview PDF](PROJECT_OVERVIEW.pdf).
