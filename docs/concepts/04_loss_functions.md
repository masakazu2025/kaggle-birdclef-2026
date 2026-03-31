# 損失関数（Loss Functions）

## BirdCLEFで使っている損失関数

### BCEWithLogitsLoss（Binary Cross Entropy with Logits）

multilabel分類の標準的な損失関数。

```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, targets)
```

---

## Binary Cross Entropy（BCE）とは

各クラスを独立した「いる/いない」の2値分類として扱う損失関数。

```
1クラスについての損失:
  loss = -[y * log(p) + (1-y) * log(1-p)]

  y=1（正解はいる）で p=0.9（予測確率0.9）→ loss小
  y=1（正解はいる）で p=0.1（予測確率0.1）→ loss大
  y=0（正解はいない）で p=0.1（予測確率0.1）→ loss小
```

全クラス（234種）の平均がfinal loss。

---

## WithLogitsとは

`BCELoss` はSigmoid後の確率を受け取る。
`BCEWithLogitsLoss` はSigmoid前のlogitsを受け取り、内部でSigmoidを計算する。

**なぜWithLogitsを使うのか:**
- 数値的に安定している（log(0)が発生しにくい）
- 計算効率が良い
- → **学習時は常にWithLogitsを使うのが標準的**

---

## SoftmaxとSigmoidの違い

| | Sigmoid | Softmax |
|---|---|---|
| 出力 | 各クラス独立に0〜1 | 全クラスの合計が1 |
| 用途 | multilabel分類 | multiclass分類（どれか1つ） |
| BirdCLEFでは | ✅ 複数種が同時に鳴く | ❌ 1種しか鳴かない前提になる |

---

## 損失と評価指標の違い

- **損失関数（loss）**: 学習中に最小化するもの。BCEWithLogitsLoss
- **評価指標（metric）**: スコアを測るもの。Macro ROC-AUC

損失が下がっても評価指標が必ずしも上がるわけではない。Public LBのスコアが最終的な基準。
