# Inference Notebook 詳細解説

対象: `notebooks/inference/birdclef-2026-pytorch-baseline-inference/`

---

## cell: imports

```python
import os
import random
import numpy as np
import pandas as pd
import plotly.express as px

import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
```

**何をしている:** 必要なライブラリを読み込む。

**押さえるべき概念:**
- `torch`: PyTorchの本体。テンソル計算・モデル定義・学習の全てを担う
- `torchaudio`: 音声ファイルの読み込み・変換（MelSpectrogram等）
- `torchvision`: 画像変換（Resize等）。音声を画像として扱うために使う
- `nn.Module`: PyTorchのモデル基底クラス。全てのモデル・変換クラスはこれを継承する

---

## cell: config

```python
# 使用するモデルのID（Trainingノートの出力から取得）
EXP_ID = "695f28562ef126fff22b6e7b1e82d451bdf7184bce4409fac7706ce02f32624c"

# TTA（推論時に同じ入力を何回通すか）。1=なし、2以上で精度が上がる場合あり
N_REPEAT = 1
```

**何をしている:** 実験で変える値を一箇所にまとめた設定セル。

**押さえるべき概念:**
- `EXP_ID`: Trainingノートが実行時に生成するハッシュ値。どのモデルを使うかを特定する
- `N_REPEAT` (TTA: Test Time Augmentation): 同じ入力を複数回モデルに通して予測を平均する手法。1回より安定したスコアが出ることが多い。増やすほど時間がかかる

---

## cell: check-input-files

```python
import os
for root, dirs, files in os.walk(
    "/kaggle/input/birdclef-2026-pytorch-baseline-training/"
):
    for f in files:
        print(os.path.join(root, f))
```

**何をしている:** Kaggleのinputフォルダにあるファイル一覧を表示する。デバッグ・確認用。

**押さえるべき概念:**
- Kaggleのnotebookは `/kaggle/input/` にデータセットをマウントして使う。ファイルが存在するか確認するのが目的

---

## cell: spectrogram-class

```python
class Spectrogram(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=256, hop_length=512, f_min=20, f_max=16000,
                 channels=1, norm="slaney", mel_scale="htk", target_size=(256, 256), top_db=80.0, delta_win=5):
        super().__init__()
        self.channels = channels
        self.top_db = top_db

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            mel_scale=mel_scale,
            pad_mode="reflect",
            power=2.0,
            norm=norm,
            center=True,
        )
        self.resize = torchvision.transforms.Resize(size=target_size)

    def power_to_db(self, S):
        amin = 1e-10
        log_spec = 10.0 * torch.log10(S.clamp(min=amin))
        log_spec -= 10.0 * torch.log10(torch.tensor(amin).to(S))
        if self.top_db is not None:
            max_val = log_spec.flatten(-2).max(dim=-1).values[..., None, None]
            log_spec = torch.maximum(log_spec, max_val - self.top_db)
        return log_spec

    def forward(self, x, resize=True):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        mel_spec = self.mel_transform(x)          # (B, n_mels, time)
        mel_spec = self.power_to_db(mel_spec)
        mel_spec = mel_spec.unsqueeze(1).repeat(1, self.channels, 1, 1)
        if resize: mel_spec = self.resize(mel_spec)   # (B, C, H, W)

        B, C = mel_spec.shape[:2]
        flat = mel_spec.view(B, C, -1)
        mins = flat.min(dim=-1).values[..., None, None]
        maxs = flat.max(dim=-1).values[..., None, None]
        mel_spec = (mel_spec - mins) / (maxs - mins + 1e-7)

        if squeeze:
            mel_spec = mel_spec.squeeze(0)
        return mel_spec
```

**何をしている:** 音声波形（1次元の数値列）をMel Spectrogram（2次元画像）に変換するクラス。

**処理の流れ:**
1. `MelSpectrogram`: 波形 → Mel Spectrogram（周波数×時間の行列）
2. `power_to_db`: 強度をdB（対数スケール）に変換。人間の聴覚は対数的なため
3. `top_db`クリッピング: 最大値から80dB以上小さい部分を切り捨て。ダイナミックレンジを圧縮
4. `Resize`: (256, 256)にリサイズ。モデルへの入力サイズを統一する
5. 正規化: 各画像を0〜1の範囲に正規化

**押さえるべき概念:**
- **Mel Spectrogram**: 周波数軸を人間の聴覚特性（Melスケール）に合わせた時間-周波数表現。横軸=時間、縦軸=周波数、色の濃さ=音の強さ
- **Melスケール**: 低周波数ほど細かく、高周波数ほど粗く分割する。人間が低音の違いを敏感に感じることに対応
- **dB変換**: 音のエネルギーは対数スケールで知覚される。変換することでモデルが学習しやすくなる
- **n_fft**: FFT（高速フーリエ変換）の窓サイズ。大きいほど周波数解像度↑・時間解像度↓
- **hop_length**: 窓をずらすステップ幅。小さいほど時間解像度↑・計算コスト↑
- **n_mels**: Melフィルタの数。多いほど周波数を細かく見られる

**パラメータと効果:**

| パラメータ | 現在値 | 変えると |
|---|---|---|
| `n_mels` | 256 | 増やすと細かい周波数情報を拾えるがメモリ増 |
| `f_min` | 20Hz | 鳥の鳴き声は500Hz〜が多い。上げるとノイズ除去効果 |
| `f_max` | 16000Hz | 人の可聴域上限。鳥種によっては下げても良い |
| `top_db` | 80 | 下げると静かな音を捨てる。上げると背景ノイズも残る |

---

## cell: birdmodel-class

```python
import timm

class BirdModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'scale': 1,
            'backbone_pooling': 'avg',
            'backbone': 'tf_efficientnetv2_b0',
            'dropout': 0.1,
            'pretrained': True,
            'channels': 1,
            'num_labels': 234,
        }
        if config: self.config.update(config)

        self.backbone = timm.create_model(
            self.config['backbone'],
            pretrained=self.config['pretrained'],
            num_classes=self.config['num_labels'],
            global_pool=self.config['backbone_pooling'],
            in_chans=1,
            drop_rate=self.config['dropout'],
        )
        feature_dim = self.backbone.num_features

    def forward(self, x):
        labels = self.backbone(x)
        return labels
```

**何をしている:** 画像分類モデル（EfficientNetV2）をバックボーンとした鳥種分類モデルの定義。

**処理の流れ:**
1. `timm.create_model`: 事前学習済みモデルを読み込む
2. `forward`: Mel Spectrogram画像を受け取り、234種のスコアを出力

**押さえるべき概念:**
- **転移学習（Transfer Learning）**: ImageNetで学習済みの重みを初期値として使う。ゼロから学習するより少ないデータ・時間で高精度になる
- **EfficientNetV2**: Googleが設計した画像分類モデルファミリー。B0〜B4とサイズ違いがあり、数字が大きいほど精度↑・計算コスト↑
- **Global Average Pooling**: 特徴マップを空間方向に平均してベクトルに圧縮する操作。`backbone_pooling='avg'`がこれ
- **timm**: PyTorchの事前学習済みモデルライブラリ。数百種類のモデルを1行で呼び出せる
- **in_chans=1**: 入力チャンネル数。Mel Spectrogramはグレースケール（1ch）なのでこの値
- **推論時は`pretrained=False`**: 学習済みのweightファイル(.pth)を後でロードするため

---

## cell: birddataset-class

```python
class BirdDataset(Dataset):
    PATH = '/kaggle/input/competitions/birdclef-2026/'
    TEST_PATH = PATH + 'test_soundscapes/'
    TRAIN_PATH = PATH + 'train_soundscapes/'
    taxonomy = pd.read_csv(PATH+'taxonomy.csv')

    LABELS = list(np.unique(taxonomy.primary_label))
    CLASSES = list(np.unique(taxonomy.class_name))
    BATCH_SIZE = 32
    DUR = 5
    SR = 32000

    def __init__(self, split_size=0.2, seed=2, n_repeat=1, is_train=True):
        paths = [self.TEST_PATH+x for x in os.listdir(self.TEST_PATH) if '.ogg' in x]
        if len(paths)==0:
            paths = [self.TRAIN_PATH+x for x in os.listdir(self.TRAIN_PATH) if '.ogg' in x]
            paths = sorted(paths)[:16]
        self.paths = paths.copy()

    def __len__(self):
        return len(self.paths)
```

**何をしている:** テスト音声ファイルの一覧を管理するクラス。

**処理の流れ:**
1. `test_soundscapes/` フォルダの`.ogg`ファイルを取得
2. テストデータがなければ `train_soundscapes/` の先頭16ファイルで代替（デバッグ用）

**押さえるべき概念:**
- **soundscapes**: 野外に設置したマイクで録音した環境音。複数の種が同時に鳴いている可能性がある（multilabel）
- **train_audio vs train_soundscapes**: `train_audio`は種ごとの短いクリップ音源、`train_soundscapes`はフィールド録音の長い音源。評価はsoundscapesに対して行われる
- `DUR=5`: 1セグメントの長さ（秒）。60秒の音声を5秒×12セグメントに分割する
- `SR=32000`: サンプリングレート（Hz）。1秒間に32000個のサンプルを取る

---

## cell: dataset-init

```python
dataset = BirdDataset()
```

**何をしている:** BirdDatasetのインスタンスを作成。テストファイルの一覧が`dataset.paths`に格納される。

---

## cell: load-model

```python
history = pd.read_csv(f"/kaggle/input/birdclef-2026-pytorch-baseline-training/history/{EXP_ID}.csv")
model_path = f"/kaggle/input/birdclef-2026-pytorch-baseline-training/models/{EXP_ID}.pth"
config = {x: history.iloc[0][x] for x in history.columns[7:]}
```

**何をしている:** TrainingノートのEXP_IDをもとに、学習済みモデルと設定値を読み込む。

**押さえるべき概念:**
- `history.csv`: Trainingノートが各エポックのloss・スコア・設定値を記録したファイル。`columns[7:]`以降が設定値
- `config = {x: history.iloc[0][x] ...}`: 1行目（最初のエポック）の設定値を取得。Mel Spectrogram等のパラメータが入っている
- Training側で設定したパラメータが自動でInferenceに引き継がれる仕組み

---

## cell: predict-function

```python
def predict(filepath):
    wav, sr = torchaudio.load(filepath)
    n_seg = int(60/dataset.DUR)
    wav = wav.float()[:,:dataset.SR*60]
    wav = wav.float().reshape((n_seg, dataset.SR*dataset.DUR)).to(device)

    activation = nn.Sigmoid()
    PREDS = []
    with torch.no_grad():
        mel = torch.stack([spec(wav[i]) for i in range(len(wav))])
        for _ in range(N_REPEAT):
            PREDS.append(activation(model(mel).unsqueeze(0)))
    PREDS = torch.concat(PREDS)
    pred = torch.mean(PREDS, dim=0)

    names = [ID+'_'+t for ID, t in zip(
        [filepath.split('/')[-1].split('.')[0]]*n_seg,
        (np.array(range(n_seg))*dataset.DUR+dataset.DUR).astype(str)
    )]
    pred = pred.cpu().numpy()
    return pred, names
```

**何をしている:** 1つの音声ファイルを受け取り、予測確率と行IDを返す関数。

**処理の流れ:**
1. 音声ファイルを読み込み（60秒）
2. 5秒×12セグメントに分割
3. 各セグメントをMel Spectrogramに変換
4. モデルに通してSigmoid → 確率値（0〜1）
5. N_REPEAT回繰り返して平均（TTA）
6. `row_id`を生成（例: `soundscape_001_5`）

**押さえるべき概念:**
- **Sigmoid**: 実数値を0〜1の確率に変換する関数。multilabel分類で使う（各クラス独立）
- **`torch.no_grad()`**: 推論時は勾配計算不要。メモリ節約・高速化のために使う
- **row_id形式**: `{ファイル名}_{終了秒数}`。60秒を5秒区切りにすると`_5`, `_10`, ..., `_60`

---

## cell: labels-audio

```python
# The model was trained on the files from "train_audio" folder only
LABELS_audio = np.unique(pd.read_csv(dataset.PATH+'train.csv').primary_label)
```

**何をしている:** `train_audio`に実際に音声データがある種のラベル一覧を取得。

**押さえるべき概念:**
- taxonomy（全650種）のうち、モデルが実際に学習した種は234種のみ
- 残りの種は予測値が0のまま提出される
- 将来的にはsoundscapesデータも活用して650種全部を予測することがスコアアップの鍵になる可能性がある

---

## cell: inference-loop

```python
from concurrent.futures import ThreadPoolExecutor
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

cfg = decode_config(config)
cfg.update({'pretrained':False})
spec = Spectrogram(**cfg['mel']).to(device)
model = BirdModel(config=cfg)
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model = model.to(device)

pred = []
names = []
model.eval()
start = time()
with ThreadPoolExecutor(max_workers=4) as executor:
    for p, n in executor.map(predict, dataset.paths):
        pred.append(p)
        names.append(n)
        fps = len(pred)/(time()-start)
        if len(pred)%16==0:
            print(np.round(100*len(pred)/len(dataset)), '%', ...)
pred = np.concatenate(pred, axis=0)
```

**何をしている:** 全テストファイルに対して並列で推論を実行する。

**押さえるべき概念:**
- **ThreadPoolExecutor**: 複数ファイルの処理を並列実行する仕組み。`max_workers=4`で4並列
- `decode_config`: historyから読んだ設定値（文字列）をPythonオブジェクトに変換
- `cfg.update({'pretrained':False})`: 推論時はImageNet事前学習重みは不要。学習済み.pthをロードするため
- `model.eval()`: BatchNormやDropoutを推論モードに切り替える。必須

---

## cell: print-shapes

```python
print("pred.shape:", pred.shape)
print("len(LABELS_audio):", len(LABELS_audio))
print("len(dataset.LABELS):", len(dataset.LABELS))
```

**何をしている:** 予測結果の形状を確認する。デバッグ用。

- `pred.shape`: `(セグメント数, 234)`の形。全ファイル×12セグメント分

---

## cell: build-submission

```python
Pred = pd.DataFrame(np.zeros((len(pred), 234)), columns=dataset.LABELS)
Pred[dataset.LABELS] = pred
Pred.insert(0, 'row_id', np.concatenate(names, axis=0))
```

**何をしている:** 予測結果をsubmission形式のDataFrameに整形する。

**押さえるべき概念:**
- `np.zeros(...)`: 全種を0で初期化してから、学習した234種の予測値を上書きする
- submission.csvの形式: `row_id`列 + 234種の確率列

---

## cell: save-submission

```python
Pred.to_csv('submission.csv', index=False)
```

**何をしている:** submission.csvとして保存する。このファイルをKaggleに提出する。
