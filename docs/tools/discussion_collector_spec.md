# Discussion収集スクリプト 仕様

## 概要

`kagglesdk` を使ってBirdCLEF 2026のDiscussionを収集・整理し、ファイル化する。
要約はスクリプトでは行わず、Claude Codeとの会話で別途実施する。

## ファイル構成

```
docs/discussions/
  index.md                          # 一覧テーブル（スクリプトが更新）
  raw/
    20260331_topic-slug.md          # 生データ：本文＋コメント（スクリプトが更新）
  summary/
    20260331_topic-slug.md          # 要約（Claude Codeとの会話で生成・更新）
```

## index.md の形式

| 日付 | タイトル | Votes | 最終更新日時 | 要約待ち | リンク |
|------|----------|-------|------------|---------|--------|
| 2026-03-13 | Acknowledge Bird Sound Recordists... | 45 | 2026-03-28T14:22Z | | [raw](raw/...) [summary](summary/...) |
| 2026-03-23 | tricks in BirdCLEF+ 2026 | 20 | 2026-04-01T12:25Z | ✓ | [raw](raw/...) |

- **日付**：Topicの初回投稿日（createTime の日付部分）
- **最終更新日時**：rawファイルに記録されている `topic.updateTime`（コメント追記で更新される）
- **要約待ち**：後述のロジックで判定。Trueのとき `✓`、FalseはBlank
- **リンク**：rawリンクは常に表示。summaryファイルがない場合はsummaryリンクなし
- Votes降順でソート

## 要約待ちの判定ロジック

```
要約待ち = True  ← summaryファイルが存在しない
要約待ち = True  ← raw.topic.updateTime > summary.target_update_time
要約待ち = False ← それ以外
```

`target_update_time` は summaryファイルのfrontmatterに記録する（後述）。

## rawファイルの更新ロジック

スクリプト実行時に以下を行う：

1. Kaggle APIから最新の `topic.updateTime` を取得
2. rawファイルに記録されている `topic.updateTime` と比較
3. API > raw ならrawファイルを再取得・上書き
4. 同じならスキップ（APIリクエストを最小化）

summaryとは完全に独立した処理。

## rawファイルの形式

```markdown
# タイトル

- **原文URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/{id}
- **投稿日時**: 2026-03-23T20:47:18.723Z
- **更新日時**: 2026-04-01T12:25:43.950Z  ← topic.updateTime（取得時点）
- **Votes**: 20
- **投稿者**: hengck23

## 本文

（messageStripped の内容）

## コメント

### コメント1（votes: 2 / hengck23 / 2026-03-30T00:51Z）

（コメント本文）

### コメント2（votes: 1 / user2 / 2026-03-29T10:00Z）

（コメント本文）
```

## summaryファイルの形式

```markdown
---
topic_id: 684148
target_update_time: 2026-04-01T12:25:43.950Z
---

# タイトル

- **原文URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/{id}
- **Votes**: 20
- **一読すべきか**: Yes / No

## 要約

（概要）

## 議論されているパラメータ・値

- パラメータ名: 値（例: lr: 1e-4）

## 考慮すべきこと

- 箇条書き
```

summaryファイルは生きているドキュメント。追記・内容変更・再要約すべて可。
要約後は `target_update_time` を更新する（更新時のrawの `topic.updateTime` を記録）。

## 取得・API仕様

### 使用API

`kagglesdk` の `KaggleClient.search.search_api_client.list_entities()` を使用。
Kaggle公式パッケージ（`kaggle`）にはフォーラム系メソッドが存在しないため、
内部SDKである `kagglesdk` を直接利用する。

### 認証

`~/.kaggle/kaggle.json` の `username` と `key` を使用（HTTP Basic認証）。

### トピック取得

```python
from kagglesdk import KaggleClient
from kagglesdk.search.types.search_api_service import (
    ListEntitiesRequest, ListEntitiesFilters, ApiSearchDiscussionsFilters,
)
from kagglesdk.discussions.types.search_discussions import (
    SearchDiscussionsSourceType, WriteUpInclusionType,
)

disc_filter = ApiSearchDiscussionsFilters()
disc_filter.source_type = SearchDiscussionsSourceType.SEARCH_DISCUSSIONS_SOURCE_TYPE_COMPETITION
disc_filter.write_up_inclusion_type = WriteUpInclusionType.WRITE_UP_INCLUSION_TYPE_EXCLUDE

filters = ListEntitiesFilters()
filters.query = "birdclef-2026"
filters.discussion_filters = disc_filter

req = ListEntitiesRequest()
req.filters = filters
req.page_size = 100
```

- `documentType == "TOPIC"` のもののみ抽出
- `next_page_token` でページネーション（全件取得）

### コメント取得

各トピックのタイトルで再検索し、`documentType == "COMMENT"` を抽出。  
`discussionDocument.newCommentUrl`（例: `/competitions/birdclef-2026/discussion/684148#...`）から
`topic_id` を抽出して親トピックと照合する。

### 取得フィールド

| フィールド | 内容 |
|-----------|------|
| `id` | Topic/Comment ID |
| `title` | トピックタイトル |
| `votes` | Votes数 |
| `createTime` | 初回投稿日時（ISO 8601） |
| `updateTime` | 最終更新日時（ISO 8601）。コメント追記で更新される模様 |
| `ownerUser.displayName` | 投稿者名 |
| `discussionDocument.messageStripped` | 本文（マークダウン除去済み全文） |
| `discussionDocument.newCommentUrl` | コメントURL（topic_id抽出に使用） |

### レート制御

サーバー負荷軽減のため、リクエスト間に `time.sleep(1)` を挿入する。

## スクリプトの動作

1. Kaggle APIで全Topicを取得（ページネーションで全件）
2. 各Topicについて：
   a. rawファイルが存在しない → 新規作成
   b. rawファイルあり・APIの `updateTime` > rawの記録 → rawを更新（コメント含め再取得）
   c. rawファイルあり・同じ → スキップ
3. index.mdを再生成（Votes降順）
   - 要約待ちフラグをsummaryファイルのfrontmatterから判定

## 一読すべきか（summaryファイル内）

**初期閾値**：Votes ≥ 5 を `Yes` とする。全件取得後に実態を見て調整可。

## 実行方法

```bash
poetry run python scripts/collect_discussions.py
```

## 実行タイミング

- 手動実行（Discussionを読むタイミングで都度実行）
- 将来的にskill化も検討
