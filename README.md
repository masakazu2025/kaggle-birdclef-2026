# BirdCLEF 2026

Kaggleコンペ: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

南米パンタナール湿地帯における音響種識別。
受動的音響モニタリング（PAM）録音から650種以上の鳥類をマルチラベル分類する。

**提出締め切り**: 2026年6月3日

---

## タスク概要

- **入力**: 受動的音響モニタリングの音声録音（.ogg）
- **出力**: 5秒クリップごとの鳥類種のマルチラベル予測
- **評価指標**: マルチラベル分類向けカスタム指標
- **対象地域**: 南米パンタナール湿地帯（15万km²以上）

## プロジェクト構成

```
kaggle-birdclef-2026/
├── notebooks/
│   ├── eda/          # 探索的データ分析
│   ├── training/     # モデル学習
│   └── inference/    # 推論・提出
├── src/birdclef/     # ユーティリティモジュール
├── scripts/          # Kaggle push/pull スクリプト
└── docs/             # 開発ドキュメント
```
