# 転移学習（Transfer Learning）

## 基本的な考え方

大量のデータで学習済みのモデルの知識を、別のタスクに流用する手法。

```
ImageNet（約120万枚の画像）で学習済みモデル
  ↓ 重みを初期値として使う
BirdCLEFのMel Spectrogramで追加学習
```

ゼロから学習するより少ないデータ・時間・計算資源で高精度を達成できる。

---

## なぜ音声タスクに画像モデルが使えるのか

Mel Spectrogramは「画像」として扱えるため、ImageNetで学習した「エッジ・テクスチャ・パターンを認識する能力」がそのまま使える。鳥の鳴き声のパターンも、視覚的なパターンと同様に認識できる。

---

## Fine-tuning（ファインチューニング）

事前学習済みモデルをそのタスク用に追加学習すること。BirdCLEFでは：

1. `pretrained=True` でImageNetの重みを読み込む
2. 最終層（分類層）をクラス数234に合わせて差し替える
3. 全体を学習データで追加学習する

---

## BirdCLEFでの設定

```python
self.backbone = timm.create_model(
    'tf_efficientnetv2_b3',
    pretrained=True,      # ImageNetの事前学習済み重みを使う
    num_classes=234,      # 最終層を234クラスに差し替え
    in_chans=1,           # 入力チャンネル数（Mel Spec はグレースケール）
)
```

- **学習時**: `pretrained=True`（ImageNetの重みから開始）
- **推論時**: `pretrained=False` + `.pth`ファイルを読み込む

---

## timmとは

PyTorchの事前学習済みモデルライブラリ。EfficientNet, ResNet, ConvNeXt, ViTなど数百種類のモデルを1行で呼び出せる。Kaggleで非常によく使われる。

```python
import timm
model = timm.create_model('tf_efficientnetv2_b3', pretrained=True, num_classes=234)
```
