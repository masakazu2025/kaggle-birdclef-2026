# BirdCLEF 2026 - Claude Code Instructions

## Gitルール

- 1テーマ = 1ブランチ = 1PR。PR内に複数コミットがあるのは自然でよい。
- コミットは自由に積んでよい。pushとPR作成はセットで、ユーザーの指示で行う。push前にコミット一覧をユーザーに見せて確認する。
- ブランチの目的と関係ないファイルは別ブランチに分けてコミットする。
- ブランチ命名規則: `notebook/training/YYYYMMDD-description`、`notebook/inference/YYYYMMDD-description`、`script/description`、`docs/description`

## 実験ワークフロー

パラメータを変更したら以下の順番で進める。Kaggle pushはユーザー確認なしに実行してよい。

### 1. 実験ログに仮説を記録
- `docs/experiments/log.md` に1行追記する（スコア後は空欄でよい）
- フォーマット: `| NNN | YYYY-MM-DD | テーマ | 変更内容 | スコア前 | | |`

### 2. トレーニングノートをKaggleにpush
```
python scripts/push_notebook.py notebooks/training/birdclef-2026-pytorch-baseline-training/birdclef-2026-pytorch-baseline-training.ipynb
```

### 3. Kaggle上でトレーニングを実行（手動）
- 完了後、EXP_IDを取得する

### 4. インフェレンスノートのEXP_IDを更新
- `notebooks/inference/.../inference.ipynb` の `EXP_ID` を新しいIDに更新

### 5. インフェレンスノートをKaggleにpush
```
python scripts/push_notebook.py notebooks/inference/birdclef-2026-pytorch-baseline-inference/birdclef-2026-pytorch-baseline-inference.ipynb
```

### 6. サブミット
- Kaggle上でインフェレンスノートを実行してサブミット（手動）

### 7. 実験ログのスコア後を記入・GitにコミットしてPR
- `docs/experiments/log.md` のスコア後・備考を埋める
- コミット・PR作成まで一気に進めてよい

## ノートブック編集

- 必ず git branch を切ってから編集する
- 設定値は `# cell: config` セルの `EPOCHS`, `LR`, `BACKBONE` 等を変更する
- **セルの挿入・削除を行う場合は、必ず事前にReadでノートブック全体を読んでセルIDと順序を把握する**
- **挿入・削除後も必ずReadで再確認し、消えたセルや重複がないか検証する**
- 過去にTrainerクラスやimportセルを誤って消した事故が複数回あった。insert操作は特に注意
