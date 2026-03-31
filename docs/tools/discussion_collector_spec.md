# Discussion収集スクリプト 仕様

## 概要

Kaggle APIを使ってBirdCLEF 2026のDiscussionを収集・整理し、ファイル化する。

## ファイル構成

```
docs/discussions/
  index.md                          # 一覧表
  20260331_discussion-title.md      # 個別ファイル（初回取得日_スラッグ）
```

## index.md の形式

| 日付 | タイトル | Votes | 一読すべきか | リンク |
|------|----------|-------|------------|--------|
| 2026-03-31 | タイトル | 12 | Yes | [link](ファイルパス) |

- Votesの降順でソート

## 個別ファイルの形式

```markdown
# タイトル

- **原文URL**: https://www.kaggle.com/...
- **初回取得日**: 2026-03-31
- **最終更新日**: 2026-03-31
- **Votes**: 12

## 要約

（概要）

## 議論されているパラメータ・値

- パラメータ名: 値（例: lr: 1e-4）

## 考慮すべきこと

- 箇条書き

## 一読すべきか

**Yes** / No

理由: ...
```

## スクリプトの動作

1. Kaggle APIで全Discussionを取得（votes数も含む）
2. 既存ファイルがあれば内容を更新（最終更新日を書き換え、votes更新）
3. 新規ファイルがあれば新規作成（ファイル名: `YYYYMMDD_スラッグ.md`）
4. index.mdをvotes降順で再生成
5. 要約・分類はClaudeに実施させる

## 実行方法

```bash
python scripts/collect_discussions.py
```

## 実行タイミング

- 手動実行（Discussionを読むタイミングで都度実行）
- 将来的にcron化も検討

## 取得項目（Kaggle API）

- タイトル
- 投稿日・更新日
- Votes数
- 本文
- 原文URL
