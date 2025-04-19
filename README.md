# 📈 FFT Viewer for Acceleration Data

このアプリは、CSV形式の加速度データからZ軸成分を選んで、FFT（高速フーリエ変換）解析を行い、スペクトルを可視化するStreamlitアプリです。

## 🚀 機能概要

- CSVファイルのアップロード（Shift-JIS, UTF-8など自動判定）
- 任意の列の選択（例：Z軸加速度）
- サンプリング周波数の指定（例：1000Hz）
- FFTの実行とスペクトルプロットの表示（Matplotlib）
- 対数スケール表示対応
- FFT結果の最大ピーク周波数を表示

## 📝 使用方法

1. 画面左のサイドバーからCSVファイルをアップロード
2. 対象列（Z軸など）を選択
3. サンプリング周波数（Hz）を入力
4. 「FFT実行」ボタンを押すと結果がグラフで表示されます

## 🧪 サンプルデータについて

本アプリは以下のようなCSVファイルに対応しています：

timestamp,acc_x,acc_y,acc_z 0.000,-0.02,0.01,0.98 0.001,-0.01,0.02,0.97 ...


- ヘッダー付き（1行目が列名）
- 任意の加速度列（Z軸など）を選択可
- サンプリングは等間隔（例：1kHz）で取得されたもの

## 🧰 使用ライブラリ

- [streamlit](https://streamlit.io/)
- pandas
- numpy
- matplotlib
- scipy

## 💻 開発環境

- Python 3.9.x（ローカル）
- Streamlit Cloud（Python 3.12.x 環境）

## 📁 ファイル構成

. ├── FFT_only.py # Streamlitアプリ本体 ├── requirements.txt # 必要パッケージ一覧 └── README.md # この説明ファイル


## 🔮 今後の追加予定

- 複数列FFT表示
- 結果のCSVエクスポート
- ピーク検出と注釈表示

## 🧑‍💻 作者

みねやん（@GPTコラボ開発）

---

StreamlitでのFFT分析を手軽に。  
データがあれば、すぐに解析できます。
