# ROC-AUC

## AUCとは

**Area Under the Curve**の略。ROC曲線の下の面積。分類モデルの性能を表す指標。

- **0.5**: ランダム（当てずっぽう）と同じ
- **1.0**: 完璧な予測
- **0.7〜0.8**: 一般的な「まあまあ使える」水準

---

## ROC曲線とは

閾値を0〜1まで変えたときの「真陽性率（TPR）vs 偽陽性率（FPR）」をプロットした曲線。

```
真陽性率（TPR）= 正解がいると予測 / 本当にいる数
偽陽性率（FPR）= 正解がいないのにいると予測 / 本当にいない数
```

閾値に依存せずにモデルの「識別能力」を測れる点が優れている。

---

## Macro ROC-AUC

BirdCLEFで使用している評価指標。

```python
def AUC(targets, outputs):
    scored_classes = (np.sum(targets, axis=0) > 0)  # 正例があるクラスのみ
    auc = roc_auc_score(targets[:,scored_classes], outputs[:,scored_classes], average='macro')
    return auc
```

- **Macro平均**: 各クラス（234種）のAUCを単純平均する
- **scored_classes**: 検証データに正例が存在するクラスのみ対象

---

## BirdCLEFのスコアとの関係

- Notebookで表示される `val_AUC` は検証データのROC-AUC
- Public LBのスコアもROC-AUCベースだが、**テストデータが異なるため値は違う**
- `val_AUC`が高くてもPublic LBが低い場合 → 過学習の可能性

---

## 現状のスコア感

| スコア | 状況 |
|---|---|
| 0.759 | ベースライン（B0） |
| 0.78〜0.80 | フェーズ1目標 |
| 0.82〜0.85 | フェーズ2目標（銅メダル圏） |
| 0.87〜 | 銀メダル圏（目安） |
