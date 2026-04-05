# BirdCLEF 2026 - Claude Code Instructions

## 実験ワークフロー

パラメータを変更したら、以下の順番で進める。確認なしに実行してよい。

### 1. トレーニングノートをKaggleにpush
```
python scripts/push_notebook.py notebooks/training/birdclef-2026-pytorch-baseline-training/birdclef-2026-pytorch-baseline-training.ipynb
```

### 2. インフェレンスノートのEXP_IDを更新してKaggleにpush
- `notebooks/inference/.../inference.ipynb` の `EXP_ID` を新しいIDに更新
- `python scripts/push_notebook.py notebooks/inference/birdclef-2026-pytorch-baseline-inference/birdclef-2026-pytorch-baseline-inference.ipynb`

### 3. サブミット
- Kaggle上でインフェレンスノートを実行してサブミット（手動）

### 4. 実験ログ更新
- `docs/experiments/log.md` にLBスコアを記録

### 5. GitにコミットしてPR→マージ
- 新しいブランチを切る（例: `edit/training-baseline/YYYYMMDD-description`）
- 変更ファイルをコミット（ブランチの目的に合うものだけ）
- GitHubにpushしてPR作成

## Kaggle push について

- `scripts/push_notebook.py` を使う。UTF-8問題は `_compat` で解決済み。
- トレーニング・インフェレンス両方をpushすること。

## ノートブック編集

- 必ず git branch を切ってから編集する
- 設定値は `# cell: config` セルの `EPOCHS`, `LR`, `BACKBONE` 等を変更する

## Git ワークフロー

- ブランチ → コミット → PR → マージ の順番を守る
- ブランチの目的と関係ないファイルは別ブランチに分けてコミットする
