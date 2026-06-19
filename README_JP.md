# Vocal Pitch Analysis App

Python + Streamlit で開発された個人向けボーカル分析・成長記録アプリです。

## プロジェクト概要

Vocal Pitch Analysis App は、ボーカル録音を分析し、基準となる歌声と比較しながら長期的な成長を記録できる個人向けボーカル分析アプリケーションです。

このプロジェクトの目的は、他人と比較したり音痴判定を行ったりすることではありません。

過去の自分と現在の自分を比較し、練習を通じてどのように成長したかを客観的なデータで確認できるよう設計されています。

録音分析だけでなく、リアルタイムチューナーやデスクトップオーバーレイ機能も提供しており、練習中にリアルタイムで音程を確認することができます。

---

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

* 周波数 (Hz) 表示

* 音名表示 (C4, A3 など)

* Cent 差表示

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

### デスクトップオーバーレイチューナー (V3.5)

* Always-On-Top オーバーレイウィンドウ
* リアルタイム音名表示
* リアルタイム Cent 表示
* Cent Bar 可視化
* Stability モニタリング
* マイクデバイス選択
* ウィンドウ位置の自動保存
* 透明度設定
* Windows 実行ファイル (EXE) 配布対応

### Lite バージョン (V3.5.1)

* FFT ベースの軽量ピッチ検出エンジン

* オーバーレイ専用軽量ビルド

* パッケージサイズ最適化

  * 121 MB → 21.8 MB

* 高速起動

* ポータビリティ向上

---

## 使用技術

* Python
* Streamlit
* Tkinter
* Librosa
* NumPy
* Pandas
* Matplotlib
* SoundDevice

---

## インストール

```bash
pip install -r requirements.txt
```

---

## 実行方法

### メインアプリ

```bash
streamlit run app.py
```

### リアルタイムチューナー

```bash
python real_time_pitch.py
```

### デスクトップオーバーレイチューナー

```bash
python overlay_tuner.py
```

---

## プロジェクト構成

```text
app.py
overlay_tuner.py
real_time_pitch.py
realtime_tuner_engine.py
translations.py
record_utils.py
records.json
requirements.txt
README.md
README_KR.md
README_JP.md
```

---

## リリース

GitHub Releases から Windows 実行ファイル (EXE) をダウンロードできます。

現在提供中のバージョン:

* Vocal Pitch Analysis App V3.5 Overlay Tuner
* Vocal Pitch Analysis App V3.5.1 Lite（推奨）

Lite バージョンでは FFT ベースの軽量エンジンを採用し、パッケージサイズを 121 MB から 21.8 MB に削減しました。

---

## ロードマップ

### V3.0

* リアルタイムチューナー MVP
* リアルタイムピッチ検出
* Stability モニタリング
* 自動ノイズ補正

### V3.5

* Always-On-Top オーバーレイ
* デスクトップチューナー UI
* マイクデバイス選択
* EXE 配布

### V3.6

* OVR Toolkit 連携テスト
* XSOverlay 表示テスト
* VR 環境での練習サポート

### V4.0

* Practice Mode
* Free Practice
* Long Tone Practice
* Section Practice
* One-Take Practice
* 練習履歴保存
* 成長ダッシュボード強化

---

## プロジェクト理念

このプロジェクトは音痴判定アプリではありません。

他人と比較するためのツールでもありません。

過去の自分と現在の自分を比較し、練習記録を積み重ねながら成長を確認することを目的としています。

歌の練習をより楽しく、継続しやすくするための個人向け成長支援ツールを目指しています。

---

## 作者

Nyanyachi
