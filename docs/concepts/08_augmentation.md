# Augmentation（データ拡張）

## Augmentationとは

学習データを人工的に変形・加工して水増しする手法。過学習を防ぎ、モデルの汎化性能を上げる。

---

## Mixup

```python
lam = Beta(alpha, alpha)  # 混合比率をサンプリング
x_mixed = lam * x1 + (1-lam) * x2
y_mixed = lam * y1 + (1-lam) * y2
```

2つのサンプルを線形補間して新しいサンプルを作る手法。

**BirdCLEFでの設定:**
- `alpha=0.5`: Beta(0.5, 0.5)から混合比率をサンプリング。0か1に近い値が多くなる
- `theta=0.8`: ラベルが0.8以上になったら1に丸める
- **Mel Spectrogramに適用**: 音声波形ではなく変換後の画像に対して適用している

**効果:** モデルが特定のサンプルに過度に依存することを防ぐ。

---

## SpecAugment

音声・Mel Spectrogram専用のaugmentation。フェーズ2で追加予定。

```
時間マスキング（Time Masking）:  横方向のランダムな帯域をゼロにする
周波数マスキング（Freq Masking）: 縦方向のランダムな帯域をゼロにする
```

**なぜ有効か:** 音声の一部が欠けても認識できるように、ロバストなモデルになる。ASR（音声認識）でも標準的な手法。

```python
import torchaudio.transforms as T

time_masking = T.TimeMasking(time_mask_param=80)
freq_masking = T.FrequencyMasking(freq_mask_param=30)

mel_spec = time_masking(mel_spec)
mel_spec = freq_masking(mel_spec)
```

---

## その他の音声Augmentation

| 手法 | 内容 | 効果 |
|---|---|---|
| ノイズ付加 | ランダムなノイズを加える | 環境ノイズへの耐性 |
| 音量変化 | 音量をランダムに変える | 録音レベルの違いへの耐性 |
| ピッチシフト | 音程をずらす | 種内の個体差への耐性 |
| タイムシフト | 時間方向にずらす | セグメント切り出し位置への耐性 |
| バックグラウンドノイズ | 別の音声を重ねる | 野外録音の環境音への耐性 |

---

## BirdCLEFでの優先順位

1. **Mixup**（実装済み）: ベースラインに含まれている
2. **SpecAugment**（フェーズ2）: 音声コンペの鉄板。効果が大きい
3. **ノイズ付加**: Discussionで有効と報告されることが多い
