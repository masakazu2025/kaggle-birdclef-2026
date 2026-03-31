# セッション引き継ぎ - BirdCLEF 2026

## 実施中の実験
- **実験名:** EXP-001: backboneをB0→B3に変更
- **目的:** B3に変えるだけでスコアが上がるか確認する（ベースライン0.759）

## タスク一覧

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 1 | Inference notebookを要約して読む | 完了 | docs/notebooks/inference_summary.md |
| 2 | Training notebookを要約して読む | 完了 | docs/notebooks/training_summary.md |
| 3 | 重要パラメータをnotebook上部にまとめる | 完了 | 両notebookのcell: configセルに集約済み |
| 4 | EXP-001: B3学習完了後にInferenceのEXP_IDを更新してsubmit | 進行中 | Training実行中（4〜5時間かかる見込み）。完了後にKaggleのoutputからexp_idを取得し、inference cell: configのEXP_IDを書き換えてpush→submit |
| 5 | Discussion収集スクリプトを作る | 未着手 | 仕様書: docs/tools/discussion_collector_spec.md |
| 6 | 実験結果をlog.mdに記録する | 未着手 | docs/experiments/log.md（スコア待ち） |

## 未決定事項
- EXP-001のスコアを見てから、次の仮説（lr調整 or SpecAugment追加）を決める
