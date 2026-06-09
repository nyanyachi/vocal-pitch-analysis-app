# Vocal Pitch Analysis App

Python と Streamlit で開発した個人向けボーカル分析・成長記録アプリです。

## 概要

このアプリは、自分の歌声を基準ボーカルと比較しながら、長期的な成長を記録するためのツールです。

他人と競争するためではなく、過去の自分と現在の自分を比較することを目的としています。

## 主な機能

### ボーカル分析

* 基準ボーカル WAV アップロード
* 自分のボーカル WAV アップロード
* librosa.pyin() によるピッチ抽出
* ピッチ比較グラフ
* キー差推定
* 自動キー補正
* 自動タイミング補正
* Cent単位の音程誤差計算
* Accuracyスコア
* Stabilityスコア
* 区間別分析
* 最高音・最低音検出

### 成長記録機能 (V2.5)

* records.json 保存
* 成長履歴ダッシュボード
* 平均 Accuracy
* 平均 Stability
* 平均キー差
* 曲フィルター
* 過去と現在の比較
* ボーカルプロフィール生成

## 使用技術

* Python
* Streamlit
* Librosa
* NumPy
* Pandas
* Matplotlib

## 起動方法

```bash
streamlit run app.py
```

## 今後の予定

### V3

* ボーカルプロフィール強化
* 基準音源品質評価
* 成長グラフ
* 音域分析
* AIコーチング機能
* 多言語対応

## 作者

Nyanyachi
