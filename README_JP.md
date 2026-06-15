# Vocal Pitch Analysis App

Python + Streamlit で開発された個人向けボーカル分析・成長記録アプリです。

## 概要

このプロジェクトは、基準となるボーカル音声と自分の歌声を比較し、時間の経過とともにどのように成長したかを記録するために作られました。

他人と競争するためのアプリではなく、過去の自分と現在の自分を比較しながら成長を確認することを目的としています。

## 主な機能

### ボーカル分析

* 基準ボーカル WAV ファイルのアップロード
* 自分のボーカル WAV ファイルのアップロード
* librosa.pyin() によるピッチ検出
* ピッチ比較グラフ
* 自動キー差推定
* 内部キー補正
* 自動タイムアライメント
* Cent ベースのピッチ誤差計算
* Accuracy スコア計算
* Stability スコア計算
* 区間別 Accuracy 分析
* 最高音・最低音検出
* ピッチ誤差の可視化

### 成長記録

* records.json への保存
* ボーカル履歴ダッシュボード
* 平均 Accuracy 計算
* 平均 Stability 計算
* 平均キー差計算
* 楽曲フィルター
* 過去録音と最新録音の比較
* 個人ボーカルプロフィール生成

### リアルタイムチューナー (V3.0)

* リアルタイムマイクピッチ検出
* 周波数(Hz)表示
* 音名表示 (C4, A3 など)
* Cent差表示
* リアルタイムチューニング判定

  * Perfect
  * Good
  * High
  * Low
* リアルタイム Cent Bar
* Pitch Smoothing
* Note Stabilization
* Pitch Hold システム
* 自動マイクノイズ補正
* リアルタイム Stability 計算

## 使用技術

* Python
* Streamlit
* Librosa
* NumPy
* Pandas
* Matplotlib
* SoundDevice

## インストール

```bash
pip install -r requirements.txt
```

## 実行方法

### メインアプリ

```bash
streamlit run app.py
```

### リアルタイムチューナーテスト

```bash
python real_time_pitch.py
```

## プロジェクト構成

```text
app.py
real_time_pitch.py
realtime_tuner_engine.py
translations.py
requirements.txt
README.md
README_KR.md
README_JP.md
```

## ロードマップ

### V3.0

* リアルタイムチューナー MVP
* リアルタイムピッチ検出
* Stability モニタリング
* 自動ノイズ補正

### V3.5

* Always On Top オーバーレイ
* VRChat 練習サポート

### V4.0

* 練習セッションモード
* ボーカル実績システム
* 高度なボーカルプロフィール
* 成長分析ダッシュボード

## 作者

Nyanyachi
