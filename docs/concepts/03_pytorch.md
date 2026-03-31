# PyTorch

## PyTorchとは

Facebookが開発したディープラーニングフレームワーク。NumPyに近い感覚でGPU計算ができ、研究・Kaggleで広く使われている。

---

## テンソル（Tensor）

PyTorchの基本データ構造。NumPyのndarrayのGPU対応版。

```python
import torch

# 作成
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.zeros(3, 4)       # 3×4のゼロ行列
x = torch.randn(2, 3)       # 正規分布からランダム生成

# 形状確認
x.shape    # torch.Size([2, 3])
x.dim()    # 2（次元数）

# GPU転送
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# NumPy変換
x.cpu().numpy()
```

**次元の意味（BirdCLEFでの例）:**
- `(B, T)`: バッチ×時間（音声波形）
- `(B, C, H, W)`: バッチ×チャンネル×高さ×幅（画像・Mel Spectrogram）

---

## nn.Module

全てのモデル・変換クラスの基底クラス。`__init__`で層を定義し、`forward`で処理を記述する。

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

model = MyModel()
output = model(input_tensor)  # forward()が自動で呼ばれる
```

**BirdCLEFでの使われ方:**
- `Spectrogram(nn.Module)`: 音声→画像変換
- `BirdModel(nn.Module)`: EfficientNetをラップした分類モデル
- `Mixup(nn.Module)`: Augmentation

---

## Dataset と DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __len__(self):
        return len(self.data)  # データ総数を返す

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # 1サンプルを返す

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for x, y in loader:
    # x: (32, ...) バッチが自動で作られる
    pass
```

**DataLoaderの主要引数:**
| 引数 | 意味 |
|---|---|
| `batch_size` | 1回の更新で使うサンプル数 |
| `shuffle` | エポックごとにデータをシャッフルするか |
| `num_workers` | データ読み込みの並列数 |
| `pin_memory` | CPUメモリをピン留め。GPU転送が速くなる |
| `drop_last` | 最後の端数バッチを捨てるか |

---

## 学習ループ

```python
model = BirdModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
loss_fn = nn.BCEWithLogitsLoss()

# 1エポックの学習
model.train()  # 学習モードに切り替え（Dropoutなどが有効になる）
for x, y in train_loader:
    optimizer.zero_grad()   # 勾配をリセット（必須）
    logits = model(x)       # 順伝播（予測）
    loss = loss_fn(logits, y)  # 損失計算
    loss.backward()         # 逆伝播（勾配計算）
    optimizer.step()        # パラメータ更新

# 検証
model.eval()  # 評価モードに切り替え（Dropout無効）
with torch.no_grad():  # 勾配計算不要（メモリ節約）
    for x, y in val_loader:
        logits = model(x)
        pred = logits.sigmoid()
```

**`model.train()` と `model.eval()` の違い:**
- `train()`: Dropout有効、BatchNorm が学習統計を使う
- `eval()`: Dropout無効、BatchNorm が保存済み統計を使う
- **忘れると結果が変わるので必ず切り替える**

---

## 勾配と逆伝播

```python
# requires_grad=True のテンソルは勾配が追跡される
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
x.grad  # tensor([4.0]) ← dy/dx = 2x = 4
```

**学習ループでの流れ:**
1. `optimizer.zero_grad()`: 前回の勾配をクリア（しないと累積される）
2. `loss.backward()`: lossから各パラメータへの勾配を計算
3. `optimizer.step()`: 勾配をもとにパラメータを更新

---

## モデルの保存と読み込み

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 読み込み
model = BirdModel()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
```

- `state_dict()`: モデルの全パラメータ（重み）をdictとして取得
- `map_location`: 保存時と異なるデバイスで読み込む際に指定

---

## よく使う関数・操作

```python
# 次元操作
x.unsqueeze(0)    # 新しい次元を追加 (3,) → (1, 3)
x.squeeze(0)      # 次元を削除 (1, 3) → (3,)
x.reshape(2, -1)  # 形状変更（-1は自動計算）
x.view(2, -1)     # reshapeに近い（メモリ連続の場合のみ）

# 統計
x.mean()
x.max()
x.min(dim=-1).values  # 最後の次元方向の最小値

# スタック
torch.stack([a, b, c])     # 新しい次元でスタック
torch.cat([a, b, c], dim=0) # 既存の次元で結合
torch.concat([a, b], dim=0) # catのエイリアス

# デバイス
x.to(device)
x.cpu()
x.detach().cpu().numpy()  # 勾配を切り離してNumPyに変換
```

---

## BirdCLEFでのデータフロー

```
音声ファイル(.ogg)
  ↓ torchaudio.load()
テンソル (1, T) ← チャンネル×サンプル数
  ↓ wav[0] で (T,) に、5秒に切り出し
テンソル (DUR,) = (160000,)
  ↓ DataLoaderがバッチ化
テンソル (B, DUR) = (32, 160000)
  ↓ Spectrogram.forward()
テンソル (B, 1, 256, 256) ← Mel Spectrogram画像
  ↓ BirdModel.forward()
テンソル (B, 234) ← 各種の生スコア（logits）
  ↓ BCEWithLogitsLoss（学習時）/ Sigmoid（推論時）
テンソル (B, 234) ← 各種の確率（0〜1）
```
