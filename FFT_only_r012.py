# coding: utf-8
'''
FFT_only_r0**.py
# 2025.4.19 r011

# csvファイルをアップロードし, FFT分析結果をグラフで表示
# Streamlit用 url: https://app-dsjptfkljmnx3rny74xnib.streamlit.app/
# ピークサーチ機能なし
# csvファイルの冒頭を表示し, データ開始行およびデータ列を指定する

### 改善点
# データ分析範囲を指定できるようにする（サンプル数：2^15個？）
- セレクタとしては「全範囲」「データ個数」秒数」を選べるようにする
- データ個数は2^n個とし、10 <= n <= 20とする（プルダウンリスト？）
- 秒数を入力したら、サンプリング周波数×秒数個のデータを抽出
- 抽出個数が元データの個数を上回ってもエラーとはせず、警告表示だけにする
### 要改善点
- データ分析開始地点を選べるようにする

### 将来構想
- wavファイル対応とする
- ピークサーチ機能を追加する
-- 設計ピークリストを計算する
- ピークサーチ結果をグラフで表示する
- ピークサーチ結果をCSVで保存する
- ピークサーチ結果をデータベースに保存する

'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import List, Tuple, Optional
import soundfile as sf

# --- Encoding checker ---
class EncodingChecker:
    def __init__(self, file_contents: bytes, encodings=None):
        self.file_contents = file_contents
        self.encodings = encodings or ['shift_jis', 'utf-8-sig', 'cp932', 'cp775', 'utf-8']
        self.encoding = self._detect_encoding()

    def _detect_encoding(self):
        for enc in self.encodings:
            try:
                self.file_contents.decode(enc)
                return enc
            except Exception:
                continue
        raise ValueError("全てのエンコーディングで読み込みに失敗しました。")

# --- FFT処理クラス ---
class GetFFT:
    def __init__(self, data, sampling_freq):
        self.data = data
        self.sampling_freq = sampling_freq
        self.results = self._frequency_component()

    def _frequency_component(self) -> Tuple[List[float], List[float]]:
        sampling_interval = 1 / self.sampling_freq
        fft_result = np.fft.fft(self.data)
        frequencies = np.fft.fftfreq(len(self.data), d=sampling_interval)
        positive = frequencies[:len(frequencies)//2]
        result = np.abs(fft_result)[:len(frequencies)//2]
        return positive[positive > 0], result[positive > 0]

# --- UI 関数たち ---
def preview_csv_head(text_content: str) -> Tuple[pd.DataFrame, int]:
    lines = text_content.splitlines()
    preview_lines = lines[:20]
    split_rows = [line.split(',') for line in preview_lines]
    max_columns = max(len(row) for row in split_rows)
    normalized = [row + [''] * (max_columns - len(row)) for row in split_rows]
    df_preview = pd.DataFrame(normalized)
    df_preview.index.name = "行番号"
    df_preview.columns = [f"列 {i}" for i in range(max_columns)]
    st.subheader("CSVファイルの冒頭20行")
    st.dataframe(df_preview)
    return df_preview, max_columns

def sidebar_skiprows() -> int:
    with st.sidebar:
        return st.number_input("データ開始行", value=0, min_value=0)

def sidebar_usecols(df_preview: pd.DataFrame) -> List[int]:
    with st.sidebar:
        available = list(range(df_preview.shape[1]))
        selected = st.multiselect("表示する列を選択", options=available, default=[0], format_func=lambda x: f"列 {x}")
        if not selected:
            st.warning("少なくとも1つの列を選択してください")
            st.stop()
        return selected

def sidebar_fft_settings(usecols: List[int], df: pd.DataFrame, rate: Optional[int] = None) -> Tuple[int, int, dict]:
    with st.sidebar:
        col = st.selectbox("FFT分析する列", options=usecols, format_func=lambda x: f"列 {x}")
        
        if rate is None:
            rate = st.number_input("サンプリング周波数 (Hz)", value=1000, min_value=1)
        else:
            # WAVの場合 → 表示のみ＆変更不可
            st.markdown(f"<div style='color:gray;'>サンプリング周波数 (Hz)</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.2em;'>{rate} Hz</div>", unsafe_allow_html=True)
        
        st.markdown("---")  # 区切り線
        st.markdown("### データ抽出設定")
        
        # データ抽出方法の選択
        selection_type = st.radio(
            "抽出方法",
            options=["全範囲", "データ個数", "秒数"],
            horizontal=True
        )

        # 選択方法に応じた入力フィールドの表示
        data_points = None
        if selection_type == "データ個数":
            n_power = st.selectbox(
                "データ個数 (2^n個)",
                options=list(range(10, 21)),
                format_func=lambda x: f"2^{x} = {2**x}点"
            )
            data_points = calculate_data_points(selection_type, n_power=n_power)
        
        elif selection_type == "秒数":
            seconds = st.number_input("秒数", value=1.0, min_value=0.1, step=0.1)
            data_points = calculate_data_points(selection_type, seconds=seconds, sampling_rate=rate)

        # データ点数の表示と警告
        if data_points is not None:
            total_points = len(df[col])
            st.info(f"選択データ点数: {data_points}点")
            if not validate_data_range(total_points, data_points):
                st.warning(f"警告: 選択点数がデータ総数({total_points}点)を超えています")

        return col, rate, {
            "selection_type": selection_type,
            "data_points": data_points
        }

def handle_fft_execution(df, fft_column, samplerate, extraction_settings):
    with st.sidebar:
        if st.button("FFT分析実行"):
            # データの抽出
            data = df[fft_column]
            points = extraction_settings["data_points"]
            
            if points is not None:
                # 指定された点数でデータを抽出
                data = data.iloc[:points]
                if len(data) < points:
                    st.warning(f"警告: 要求された点数({points}点)に対して、実際のデータ点数は{len(data)}点です")
            
            # FFT実行
            freqs, result = GetFFT(data, samplerate).results
            
            # 現在の周波数表示範囲を保持
            current_max_freq = (
                st.session_state.get("max_freq", int(freqs.max()))
                if "max_freq" in st.session_state
                else int(freqs.max())
            )
            
            # 結果の保存
            st.session_state.fft_done = True
            st.session_state.frequencies = freqs
            st.session_state.fft_result = result
            st.session_state.fft_column = fft_column
            st.session_state.max_freq = min(current_max_freq, int(freqs.max()))  # 現在値と新しい最大値の小さい方を使用
            
            # 使用したデータ点数の表示
            st.info(f"FFT実行データ点数: {len(data)}点")

def show_time_series_plot(df: pd.DataFrame, usecols: List[int], extraction_settings: dict):
    st.subheader("選択列の時系列データ")
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # データプロット
    for col in usecols:
        ax.plot(df[col], label=f'Column {col}', alpha=0.7)
    
    # 抽出範囲の可視化
    if extraction_settings["selection_type"] != "全範囲" and extraction_settings["data_points"] is not None:
        points = extraction_settings["data_points"]
        ax.axvspan(0, points, color='red', alpha=0.1)  # 抽出範囲を半透明の赤で表示
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)  # 開始位置
        ax.axvline(x=points, color='red', linestyle='--', alpha=0.5)  # 終了位置
        
        # 抽出範囲の情報をテキストで表示
        if extraction_settings["selection_type"] == "データ個数":
            ax.text(points/2, ax.get_ylim()[1], f'Selected: {points}pts',
                   horizontalalignment='center', verticalalignment='bottom',
                   color='red', alpha=0.7, bbox=dict(facecolor='white', alpha=0.7))
        else:  # "秒数"の場合
            seconds = points / extraction_settings.get("sampling_rate", 1000)
            ax.text(points/2, ax.get_ylim()[1], 
                   f'Selected: {seconds:.1f}sec ({points}pts)',
                   horizontalalignment='center', verticalalignment='bottom',
                   color='red', alpha=0.7, bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xlabel('Sample')
    ax.set_ylabel('Acceleration')
    ax.legend(fontsize=16)
    ax.grid(True)
    st.pyplot(fig)

def show_fft_result_graph():
    st.subheader(f"列 {st.session_state.fft_column} のFFT分析結果")
    yscale = st.radio("**縦軸の目盛**", options=['linear', 'log'],
                      format_func=lambda x: '線形目盛' if x == 'linear' else '対数目盛',
                      horizontal=True, index=0)

    col1, col2 = st.columns([1, 3])
    with col1:
        max_freq = st.number_input("**周波数表示範囲 (Hz)**",
                                   min_value=1,
                                   max_value=int(st.session_state.frequencies.max()),
                                   value=st.session_state.max_freq,  # 保存された値を使用
                                   step=1,                                   
                                   key="max_freq_input")
    with col2:
        st.markdown(" ")  # 高さ合わせ用
        if st.button("グラフを更新", key="update_graph"):
            st.session_state.max_freq = max_freq

    if 'max_freq' not in st.session_state:
        st.session_state.max_freq = max_freq

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(st.session_state.frequencies,
            st.session_state.fft_result,
            label="FFT Result", alpha=0.7)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel("Frequency (Hz)", fontsize=16)
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Column {st.session_state.fft_column} FFT Result")
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_yscale(yscale)
    ax.set_xlim(0, st.session_state.max_freq)
    st.pyplot(fig)

def calculate_data_points(selection_type: str, n_power: int = None, seconds: float = None, sampling_rate: int = None) -> int:
    """データ点数を計算する関数"""
    if selection_type == "全範囲":
        return None  # Noneを返すことで、全範囲を示す
    elif selection_type == "データ個数":
        return 2 ** n_power
    else:  # "秒数"の場合
        return int(seconds * sampling_rate)

def validate_data_range(total_points: int, selected_points: int) -> bool:
    """選択されたデータ点数が有効かチェックする関数"""
    if selected_points is None:  # 全範囲選択の場合
        return True
    return selected_points <= total_points

# --- アプリ本体 ---
# --- アプリ本体 ---
def main():
    st.title("FFT分析アプリケーション")
    # FFT設定の前でグラフ描画エリアを用意
    plot_area = st.empty()
    
    with st.sidebar:
        st.header("設定")
        file_type = st.selectbox("ファイル形式を選択", ["CSV", "WAV"])

        if file_type == "CSV":
            uploaded_file = st.file_uploader("CSVファイルを選択してください", type=['csv'])
        else:
            uploaded_file = st.file_uploader("WAVファイルを選択してください", type=['wav'])

    if uploaded_file:
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.clear()
            st.session_state.last_uploaded_file = uploaded_file

        try:
            if file_type == "CSV":
                file_contents = uploaded_file.read()
                encoding = EncodingChecker(file_contents).encoding
                text_content = file_contents.decode(encoding)

                df_preview_raw, _ = preview_csv_head(text_content)
                skiprows = sidebar_skiprows()

                string_data = io.StringIO(text_content)
                string_data.seek(0)
                df_preview = pd.read_csv(string_data, nrows=5, skiprows=skiprows,
                                         encoding=encoding, header=None, on_bad_lines='skip', sep=',', skipinitialspace=True)
                st.subheader(f"データ開始行({skiprows}行目)以降のプレビュー")
                st.dataframe(df_preview)

                usecols = sidebar_usecols(df_preview)

                string_data.seek(0)
                df = pd.read_csv(string_data, skiprows=skiprows, usecols=usecols,
                                 encoding=encoding, header=None, on_bad_lines='skip', sep=',', skipinitialspace=True)

                rate = None  # CSVはあとでUI入力

                # ✅ CSVの場合は「時系列表示ボタン」で表示
                with st.sidebar:
                    if st.button("時系列データを表示", key="show_raw_data"):
                        st.session_state.show_raw = True
                        st.session_state.df = df
                        st.session_state.usecols = usecols

                if st.session_state.get("show_raw", False):
                    show_time_series_plot(st.session_state.df, st.session_state.usecols, {
                        "selection_type": "全範囲",
                        "data_points": None,
                        "sampling_rate": 1000  # 仮の表示値（本当はrate UIで指定される）
                    })

            else:
                # 🔹 WAV読み込み（自動で1枚グラフ表示）
                data, samplerate = sf.read(io.BytesIO(uploaded_file.read()))
                st.success("WAVファイルの読み込みに成功しました")
                st.write(f"サンプリング周波数: {samplerate} Hz")
                st.write(f"データ点数: {len(data)}")

                if data.ndim == 1:
                    df = pd.DataFrame({0: data})
                else:
                    df = pd.DataFrame(data)

                usecols = list(df.columns)
                rate = samplerate

                # ✅ WAVはアップロード後すぐに時系列グラフ表示
                show_time_series_plot(df, usecols, {
                    "selection_type": "全範囲",
                    "data_points": None,
                    "sampling_rate": rate
                })

            # ✅ 共通：FFT設定UI（rate は WAVなら固定、CSVなら UI入力）
            fft_column, samplerate, extraction_settings = sidebar_fft_settings(usecols, df, rate=rate)
            extraction_settings["sampling_rate"] = samplerate

            # ✅ 抽出設定を反映したグラフを必ず再描画（赤帯表示）
            show_time_series_plot(df, usecols, extraction_settings)

            # ✅ 共通：FFT実行ボタンで処理実行
            handle_fft_execution(df, fft_column, samplerate, extraction_settings)

            if st.session_state.get("fft_done", False):
                show_fft_result_graph()

        except Exception as e:
            st.error(f"データの読み込み中にエラーが発生しました: {e}")
            st.stop()

if __name__ == "__main__":
    main()
