# BirdCLEF 2026

Kaggle competition: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

Acoustic species identification in the Pantanal wetlands of South America.
Multi-label classification of 650+ bird species from passive acoustic monitoring recordings.

**Submission deadline**: June 3, 2026

---

## Project Structure

```
kaggle-birdclef-2026/
├── notebooks/          # Kaggle notebooks
│   ├── eda/            # Exploratory data analysis
│   ├── training/       # Model training
│   └── inference/      # Inference & submission
├── src/birdclef/       # Utility modules (uploaded as Kaggle Dataset)
├── scripts/            # Kaggle push/pull helpers
└── docs/               # Documentation
```

## Setup

```bash
poetry install
```

Requires `~/.kaggle/kaggle.json` with your Kaggle API credentials.

## Workflow

### Push a notebook to Kaggle

```bash
poetry run python scripts/push_notebook.py notebooks/eda/eda-01-overview.ipynb
```

On first run, a `kernel-metadata.json` is generated — edit the `id` field to set your Kaggle username, then run again to push.

### Pull notebook output from Kaggle

```bash
poetry run python scripts/pull_notebook.py your-username/eda-01-overview
```

## Task

- **Input**: Audio recordings (.ogg) from passive acoustic monitoring
- **Output**: Multi-label bird species predictions per 5-second clip
- **Evaluation**: Custom multilabel classification metric
