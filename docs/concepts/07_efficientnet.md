# EfficientNet / EfficientNetV2

## EfficientNetとは

Googleが2019年に発表した画像分類モデルファミリー。「精度・速度・モデルサイズのバランスが良い」ことで有名で、Kaggleでよく使われる。

---

## スケーリングの考え方

従来のモデルは「深さ（層数）」「幅（チャンネル数）」「解像度」のどれか1つを増やすだけだった。EfficientNetは**3つを同時に均等にスケールアップ**することで効率的に精度を上げた（Compound Scaling）。

---

## B0〜B4の違い

| モデル | パラメータ数 | 精度（ImageNet） | 推論速度 |
|---|---|---|---|
| B0 | 5.3M | 77.1% | 速い |
| B1 | 7.8M | 79.1% | ↓ |
| B2 | 9.1M | 80.1% | ↓ |
| B3 | 12M | 81.6% | ↓ |
| B4 | 19M | 82.9% | 遅い |

**BirdCLEFでの考え方:** B0はベースライン用、B3はバランスが良く上位解法でよく使われる。B4以上はGPUメモリとの兼ね合いが必要。

---

## EfficientNetV2とは

2021年にGoogleが発表した改良版。V1より学習速度が速く、同精度でより小さいモデルサイズを実現。

BirdCLEFのベースラインは`tf_efficientnetv2_b0`（TensorFlow版の重みを使ったV2のB0）。

---

## `backbone`の変更方法

```python
# ベースライン
BACKBONE = "tf_efficientnetv2_b0"

# B3に変更（1行だけ）
BACKBONE = "tf_efficientnetv2_b3"

# B4に変更
BACKBONE = "tf_efficientnetv2_b4"
```

変更した場合、モデルが大きくなるため：
- 学習時間が増える（B0の約2〜3倍）
- バッチサイズを下げる必要が出る場合がある

---

## なぜB3がよく効くのか

B0は軽すぎてMel Spectrogramの複雑なパターンを十分に学習できないことが多い。B3は精度とコストのバランスが良く、Kaggleの音声コンペでよく上位解法に登場する。
