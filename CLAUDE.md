# BirdCLEF 2026 - Claude Code Instructions

## Gitルール

- 確認なしに自由にコミット・pushしてよい。
- 同じテーマの修正が出たら新しいコミットを積まず `git commit --amend` + `git push --force-with-lease` で前のコミットに統合する。
- ブランチ → コミット → PR → マージ の順番を守る。
- ブランチの目的と関係ないファイルは別ブランチに分けてコミットする。

## 実験ワークフロー

パラメータを変更したら以下の順番で進める。Kaggle pushはユーザー確認なしに実行してよい。

### 1. トレーニングノートをKaggleにpush
```
python scripts/push_notebook.py notebooks/training/birdclef-2026-pytorch-baseline-training/birdclef-2026-pytorch-baseline-training.ipynb
```

### 2. Kaggle上でトレーニングを実行（手動）
- 完了後、EXP_IDを取得する

### 3. インフェレンスノートのEXP_IDを更新
- `notebooks/inference/.../inference.ipynb` の `EXP_ID` を新しいIDに更新

### 4. インフェレンスノートをKaggleにpush
```
python scripts/push_notebook.py notebooks/inference/birdclef-2026-pytorch-baseline-inference/birdclef-2026-pytorch-baseline-inference.ipynb
```

### 5. サブミット
- Kaggle上でインフェレンスノートを実行してサブミット（手動）

### 6. 実験ログ更新・GitにコミットしてPR
- `docs/experiments/log.md` にLBスコアを記録
- コミット・PR作成まで一気に進めてよい

## ノートブック編集

- 必ず git branch を切ってから編集する
- 設定値は `# cell: config` セルの `EPOCHS`, `LR`, `BACKBONE` 等を変更する
