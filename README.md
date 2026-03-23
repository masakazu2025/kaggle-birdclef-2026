# BirdCLEF 2026

Kaggle competition: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

Acoustic species identification in the Pantanal wetlands of South America.
Multi-label classification of 650+ bird species from passive acoustic monitoring recordings.

**Submission deadline**: June 3, 2026

---

## Project Structure

```
kaggle-birdclef-2026/
├── notebooks/
│   ├── eda/                    # Exploratory data analysis
│   ├── training/               # Model training
│   └── inference/              # Inference & submission
│       └── <kernel-name>/
│           ├── <kernel-name>.ipynb
│           └── kernel-metadata.json
├── src/birdclef/               # Utility modules
├── scripts/
│   ├── push_notebook.py        # Push notebook to Kaggle
│   └── pull_notebook.py        # Pull notebook from Kaggle
└── docs/
```

## Setup

```bash
poetry install
```

Kaggle authentication via `KAGGLE_API_TOKEN` environment variable (or `~/.kaggle/kaggle.json`).

## Workflow

### Pull a notebook from Kaggle

```bash
poetry run python scripts/pull_notebook.py masakazum/<kernel-name> --output-dir notebooks/inference
```

Pulls the notebook and `kernel-metadata.json` into `notebooks/inference/<kernel-name>/`.

### Push a notebook to Kaggle

```bash
poetry run python scripts/push_notebook.py notebooks/inference/<kernel-name>/<kernel-name>.ipynb
```

On first run, `kernel-metadata.json` is generated — edit if needed, then run again to push.
Push creates a new Kaggle kernel version and starts execution automatically.

## Task

- **Input**: Audio recordings (.ogg) from passive acoustic monitoring
- **Output**: Multi-label bird species predictions per 5-second clip
- **Evaluation**: Custom multilabel classification metric
- **Location**: Pantanal wetlands, South America (150,000+ km²)
