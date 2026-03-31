# Training Notebook 詳細解説

対象: `notebooks/training/birdclef-2026-pytorch-baseline-training/`

---

## cell: imports

```python
import os
import random
import numpy as np
import pandas as pd

import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold

os.makedirs("history", exist_ok=True)
os.makedirs("models", exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**何をしている:** ライブラリの読み込みと、再現性確保のための`set_seed`関数を定義する。

**押さえるべき概念:**
- `DataLoader`: データセットをバッチ単位で読み込む仕組み。シャッフル・並列読み込みも担う
- `StratifiedKFold`: クラスの比率を保ちながらデータを分割するKFold。各foldで全クラスが均等に含まれる
- **再現性（set_seed）**: 乱数シードを固定することで、実行するたびに同じ結果が得られる。実験の比較に必須
- `torch.backends.cudnn.deterministic = True`: GPU演算の決定論的動作を強制する。少し遅くなるが再現性が上がる

---

## cell: config

```python
BACKBONE    = "tf_efficientnetv2_b3"
PRETRAINED  = True
EPOCHS      = 32
LR          = 5e-4
BATCH_SIZE  = 32
SEED        = 2
DROPOUT     = 0.2
TRAIN_ONLY  = False

N_MELS      = 256
F_MIN       = 20
F_MAX       = 16000
N_FFT       = 2048
TARGET_SIZE = (256, 256)

MIX_ALPHA   = 0.5
MIX_THETA   = 0.8
```

**何をしている:** 実験で変える値を一箇所にまとめた設定セル。ここだけ変えて実験する。

**パラメータと効果:**

| パラメータ | 現在値 | 意味・変えると |
|---|---|---|
| `BACKBONE` | b3 | モデルの種類。b0→b3→b4と大きくなるほど精度↑・時間↑ |
| `LR` | 5e-4 | 学習率。大きすぎると発散、小さすぎると収束が遅い |
| `EPOCHS` | 32 | 全データを何周学習するか |
| `BATCH_SIZE` | 32 | 1回の更新で使うサンプル数。大きいほど安定するがメモリ増 |
| `DROPOUT` | 0.2 | 過学習防止。学習中にランダムにニューロンを無効化する割合 |
| `TRAIN_ONLY` | False | Trueにすると全データで学習（最終提出前に使う） |
| `MIX_ALPHA` | 0.5 | Mixupの強さ。大きいほど強くミックスされる |
| `MIX_THETA` | 0.8 | ラベルをこの値以上で1に丸める閾値 |

---

## cell: spectrogram-class

```python
class Spectrogram(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=256, hop_length=512,
                 f_min=20, f_max=16000, channels=1, norm="slaney",
                 mel_scale="htk", target_size=(256, 256), top_db=80.0, delta_win=5):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(...)
        self.resize = torchvision.transforms.Resize(size=target_size)

    def power_to_db(self, S):
        log_spec = 10.0 * torch.log10(S.clamp(min=1e-10))
        log_spec -= 10.0 * torch.log10(torch.tensor(1e-10).to(S))
        if self.top_db is not None:
            max_val = log_spec.flatten(-2).max(dim=-1).values[..., None, None]
            log_spec = torch.maximum(log_spec, max_val - self.top_db)
        return log_spec

    def forward(self, x):
        mel_spec = self.mel_transform(x)      # 波形 → Mel Spectrogram
        mel_spec = self.power_to_db(mel_spec) # dB変換
        mel_spec = mel_spec.unsqueeze(1).repeat(1, self.channels, 1, 1)
        mel_spec = self.resize(mel_spec)       # (B, C, 256, 256)にリサイズ
        # 0〜1に正規化
        mins = mel_spec.view(B, C, -1).min(dim=-1).values[..., None, None]
        maxs = mel_spec.view(B, C, -1).max(dim=-1).values[..., None, None]
        mel_spec = (mel_spec - mins) / (maxs - mins + 1e-7)
        return mel_spec
```

**何をしている:** 音声波形をMel Spectrogramに変換するクラス。InferenceのSpectrogramと同じ。

※詳細はinference_summary.mdのcell: spectrogram-classを参照。

**Trainingでの違い:** Inferenceと同じクラスだが、`resize=True`が固定（推論側は引数で制御できる）。

---

## cell: birddataset-class

```python
class BirdDataset(Dataset):
    PATH = '/kaggle/input/competitions/birdclef-2026/'

    def __init__(self, is_train=True, fold=0, config={}):
        # train.csv / taxonomy.csvを読み込む
        # StratifiedKFoldでfold分割
        # 学習/検証データのパスとラベルを格納

    def make_labels(self, X):
        # primary_labelをone-hotベクトルに変換
        out = np.zeros(len(self.LABELS)).astype(bool)
        out[self.LABELS.index(X)] = True
        return out

    def load_sound(self, filepath, DUR=5*32000):
        wav, sr = torchaudio.load(filepath)
        if len(wav) < DUR:
            # 短い場合: ゼロパディング（ランダム位置に配置）
        else:
            # 長い場合: 学習時はランダムクロップ、検証時は先頭から切り出し
        return wav

    def __getitem__(self, idx):
        audio = self.load_sound(path, DUR=int(5 * self.config['sr']))
        labels = self.labels[idx]
        return audio, torch.tensor(labels, dtype=torch.float32)

def create_dataloaders(config, fold=0):
    # 学習用・検証用DataLoaderを生成して返す
```

**何をしている:** `train_audio/`の音声ファイルを読み込み、5秒クリップとラベルのペアを返すデータセットクラス。

**処理の流れ:**
1. `train.csv`からファイルパスと`primary_label`を取得
2. StratifiedKFoldで5分割し、指定foldの学習/検証インデックスを取得
3. `__getitem__`で呼ばれるたびに音声を読み込んで5秒に切り出す
4. ラベルをone-hotベクトルに変換して返す

**押さえるべき概念:**
- **one-hotベクトル**: クラス数分の長さで、該当クラスだけ1、それ以外は0のベクトル。例：234種のうちクラス5なら`[0,0,0,0,0,1,0,...,0]`
- **ランダムクロップ**: 学習時に音声のランダムな位置から5秒を切り出す。毎回違う部分が使われるのでaugmentationとしても機能する
- **ゼロパディング**: 5秒未満の音声をゼロで埋めて5秒にする。ランダム位置に配置することでaugmentationになる
- **fold**: 5分割したデータのうち1つを検証データ、残り4つを学習データとして使う。現状はfold=0固定
- `drop_last=True`: バッチサイズに満たない最後のバッチを捨てる。バッチサイズを統一するため

---

## cell: dataset-init

```python
ds = BirdDataset()
```

**何をしている:** データセットの動作確認用インスタンス生成。実際の学習には使わない（Trainerの中でcreate_dataloadersが呼ばれる）。

---

## cell: auc-metric

```python
from sklearn.metrics import roc_auc_score

def AUC(targets, outputs, verbose=False):
    targets = (targets>0).astype(float)
    scored_classes = (np.sum(targets, axis=0) > 0)  # 正例が存在するクラスのみ
    auc = roc_auc_score(targets[:,scored_classes], outputs[:,scored_classes], average='macro')
    return auc
```

**何をしている:** 検証データに対してMacro ROC-AUCスコアを計算する関数。

**押さえるべき概念:**
- **ROC-AUC**: 分類モデルの性能指標。0.5がランダム、1.0が完璧。閾値に依存しない
- **Macro平均**: 各クラスのAUCを平均する。クラス間の不均衡に影響されにくい
- `scored_classes`: 検証データに正例が存在するクラスのみでスコアを計算する。正例がないクラスはAUCが計算できないため除外
- このAUCはvalidation用の指標であり、Public LBのスコアとは異なる（参考値として使う）

---

## cell: birdmodel-class

```python
import timm

class BirdModel(nn.Module):
    def __init__(self, config=None):
        self.config = {
            'backbone': 'tf_efficientnetv2_b0',
            'dropout': 0.1,
            'pretrained': True,
            'channels': 1,
            'num_labels': 234,
            ...
        }
        if config: self.config.update(config)

        self.backbone = timm.create_model(
            self.config['backbone'],
            pretrained=self.config['pretrained'],
            num_classes=self.config['num_labels'],
            global_pool=self.config['backbone_pooling'],
            in_chans=self.config['channels'],
            drop_rate=self.config['dropout'],
        )

    def forward(self, x):
        labels = self.backbone(x)
        return labels  # ← Sigmoidはまだかけていない（lossの中でかける）
```

**何をしている:** EfficientNetV2をバックボーンとした鳥種分類モデルの定義。

**押さえるべき概念:**
- **Inferenceとの違い**: Trainingでは`forward`の出力はlogits（生のスコア）。Sigmoidは`BCEWithLogitsLoss`の中で計算される。Inferenceでは`nn.Sigmoid()`を明示的に呼んでいる
- **logits**: Sigmoid/Softmaxをかける前の生のスコア。`BCEWithLogitsLoss`はlogitsを受け取る設計で、数値的に安定している
- **`pretrained=True`**: 学習時はImageNetの事前学習済み重みから始める（転移学習）

---

## cell: mixup-class

```python
class Mixup(nn.Module):
    def __init__(self, alpha=0.5, theta=1):
        self.alpha = alpha
        self.theta = theta

    def forward(self, x, y):
        lam = torch.tensor(np.random.beta(alpha, alpha, batch_size))
        lam = torch.maximum(lam, 1-lam)  # 常に0.5以上にする
        idx = torch.randperm(batch_size)  # バッチ内でランダムにペアを作る

        x = lam * x + (1-lam) * x[idx]          # 画像をブレンド
        y = lam * y + (1-lam) * y[idx]           # ラベルもブレンド
        y[y >= theta] = 1                         # 閾値以上は1に丸める
        return x, y
```

**何をしている:** バッチ内の2サンプルを混ぜ合わせるAugmentation（データ拡張）。

**処理の流れ:**
1. Beta分布から混合比率`lam`をサンプリング（0.5〜1.0の範囲）
2. バッチ内でランダムなペアを作る（`idx`）
3. 画像とラベルを`lam`の比率で混ぜる
4. ラベルが`theta`以上なら1に丸める

**押さえるべき概念:**
- **Mixup**: 2つのサンプルを線形補間して新しいサンプルを作るAugmentation手法。過学習を防ぎ汎化性能を上げる効果がある
- **Beta分布**: `alpha`パラメータで形状が変わる確率分布。`alpha=0.5`だと0または1に近い値が多くサンプルされる
- `lam = maximum(lam, 1-lam)`: 常に「元のサンプルが優勢」になるよう、混合比率を0.5以上に制限している
- `theta=0.8`: ラベルが0.8以上になったら1とみなす。あいまいなラベルを作りすぎないための工夫
- **Mel Spectrogramに適用**: 音声ではなく、変換後の画像に対してMixupを適用している

---

## cell: cfg-trainer-class

```python
CFG = {
    'seed': 2,
    'batch_size': 32,
    'lr': 5e-4,
    'loss': nn.BCEWithLogitsLoss(),
    'mel': {'n_mels':256, 'f_min':20, ...},
    'metrics': [AUC],
    'scheduler': True,
    'model': BirdModel,
    ...
}

class Trainer:
    def __init__(self, config={}, fold=0):
        self.config = CFG.copy()
        self.config.update(config)  # experiment-configで上書きされる
        self.exp_id = hashlib.sha256(str(time.time()).encode()).hexdigest()

    def train_one_epoch(self, epoch):
        for x, y in train_loader:
            x = self.mel(x)         # 波形 → Mel Spectrogram
            x, y = mix(x, y)        # Mixup適用
            logits = self.model(x)  # 予測
            L = self.loss_fn(logits, y)  # loss計算
            L.backward()            # 勾配計算
            self.optimizer.step()   # パラメータ更新

    def validate(self):
        for x, y in val_loader:
            x = self.mel(x)
            logits = self.model(x)
            pred = logits.sigmoid()  # 確率に変換
        return AUC(target, pred), loss

    def train(self, epochs):
        # optimizer, scheduler, modelを初期化
        # train_one_epoch → validate を epochs回繰り返す
        # 毎エポックhistory.csvに記録
        # 最後にmodels/{exp_id}.pthを保存
```

**何をしている:** 学習ループ全体を管理するTrainerクラスと、デフォルト設定CFGの定義。

**押さえるべき概念:**
- **BCEWithLogitsLoss**: Binary Cross Entropy Loss。multilabel分類の標準的な損失関数。各クラスを独立した2値分類として扱う
- **AdamW**: 重み減衰（Weight Decay）付きのAdamオプティマイザ。過学習防止に有効
- **CosineAnnealingLR**: 学習率をコサイン曲線に沿って徐々に下げるスケジューラ。`eta_min=1e-8`が最小値
- **勾配計算（backward）**: モデルのパラメータをどう変えればlossが下がるかを計算する操作
- **exp_id**: 実行時刻のハッシュ値。実験を一意に識別する。InferenceではこのIDでモデルを指定する
- **CFG vs config**: CFGがデフォルト値、experiment-configセルのconfigが実験ごとの上書き値。`CFG.copy().update(config)`で合成される

**学習の1ステップを図示:**
```
音声波形
  ↓ Spectrogram（Mel変換）
画像(256×256)
  ↓ Mixup（2サンプルをブレンド）
拡張画像
  ↓ BirdModel（EfficientNetV2）
logits（234次元）
  ↓ BCEWithLogitsLoss（正解ラベルと比較）
loss
  ↓ backward（勾配計算）
  ↓ optimizer.step（パラメータ更新）
```

---

## cell: experiment-config

```python
config = {
    'seed':       SEED,
    'batch_size': BATCH_SIZE,
    'backbone':   BACKBONE,
    'pretrained': PRETRAINED,
    'dropout':    DROPOUT,
    'train_only': TRAIN_ONLY,
    'loss':       nn.BCEWithLogitsLoss(),
    'model':      BirdModel,
    'mel': {
        'n_mels':      N_MELS,
        'f_min':       F_MIN,
        'f_max':       F_MAX,
        'n_fft':       N_FFT,
        'target_size': TARGET_SIZE,
    },
    'mix': {
        'alpha': MIX_ALPHA,
        'theta': MIX_THETA,
    },
}
```

**何をしている:** cell: configで定義した変数をTrainerに渡すdictに整形する。

**押さえるべき概念:**
- このconfigがCFGを上書きする。ここに書いた値だけが実験で変わる
- `history.csv`にこのconfigの全値が記録され、Inferenceで自動的に引き継がれる

---

## cell: run-training

```python
trainer = Trainer(config=config)
model = trainer.train(epochs=EPOCHS)
```

**何をしている:** Trainerを初期化して学習を開始する。

**出力:**
- `models/{exp_id}.pth`: 学習済みモデルの重み
- `history/{exp_id}.csv`: 各エポックのloss・AUC・設定値の記録

**押さえるべき概念:**
- `exp_id`はTrainer初期化時に自動生成される。学習完了後にKaggleのoutputから確認し、InferenceのEXP_IDに設定する
- 学習終了後、最後にもう一度`validate()`が呼ばれてfinal scoreが記録される
