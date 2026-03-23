# セットアップ

## 必要環境

- Python 3.12+
- Poetry

## インストール

```bash
poetry install
```

## Kaggle認証

`KAGGLE_API_TOKEN` 環境変数を設定する（Kaggle の Settings → API Tokens から発行）。

または `~/.kaggle/kaggle.json` に Legacy API Credentials を配置する。

# ワークフロー

## Kaggleからノートブックをpull

```bash
poetry run python scripts/pull_notebook.py masakazum/<kernel-name> --output-dir notebooks/inference
```

`notebooks/inference/<kernel-name>/` にノートブックと `kernel-metadata.json` が保存される。

## Kaggleへノートブックをpush

```bash
poetry run python scripts/push_notebook.py notebooks/inference/<kernel-name>/<kernel-name>.ipynb
```

初回実行時は `kernel-metadata.json` が生成される。必要に応じて編集後、再度実行してpush。
pushすると新しいKaggleカーネルバージョンが作成され、自動で実行が開始される。
