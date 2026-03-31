# Optimizer と Learning Rate Scheduler

## Optimizer（最適化アルゴリズム）

損失関数を最小化するためにモデルのパラメータを更新するアルゴリズム。

---

## AdamW

BirdCLEFで使用しているoptimizer。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
```

### Adamとは
勾配の**平均（1次モーメント）**と**分散（2次モーメント）**を追跡し、パラメータごとに学習率を自動調整する。

- 学習率の手動調整が少なくて済む
- 収束が速い
- Kaggleで最もよく使われるoptimizer

### AdamとAdamWの違い
- **Adam**: L2正則化がパラメータ更新に混入する問題がある
- **AdamW**: 重み減衰（Weight Decay）を適切に分離。過学習防止効果が高い

---

## 学習率（Learning Rate）

1回のパラメータ更新でどれだけ動かすかを決めるハイパーパラメータ。

```
lr = 5e-4 = 0.0005
```

| lr | 挙動 |
|---|---|
| 大きすぎる（1e-2〜） | 発散する。lossが下がらない |
| 適切（1e-4〜5e-4） | 安定して収束する |
| 小さすぎる（1e-6〜） | 収束が遅すぎる |

**BirdCLEFでの考え方:** B3に変えた場合、`lr=5e-4`はやや大きい可能性がある。`1e-4`に下げることで安定することもある。

---

## CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-8
)
```

学習率をコサイン曲線に沿って徐々に下げるスケジューラ。

```
エポック0:   lr = 5e-4（最大）
エポック16:  lr ≈ 2.5e-4（中間）
エポック32:  lr = 1e-8（最小 = eta_min）
```

**なぜ有効か:**
- 学習初期は大きい学習率で素早く探索
- 学習後期は小さい学習率で細かく収束させる
- 急激に下げるより滑らかに下げる方が安定する

---

## BirdCLEFでのスケジューラの位置

```python
for epoch in range(epochs):
    train_one_epoch()
    validate()
    scheduler.step()  # ← エポックごとに学習率を更新
```

`scheduler.step()`は**エポックの最後**に呼ぶ（バッチごとではない）。
