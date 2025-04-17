# coding: utf-8
'''
FFT_only_r0**.py
# 2025.4.17 r010デプロイ

# csvファイルをアップロードし, FFT分析結果をグラフで表示
# Streamlit用 url: https://app-dsjptfkljmnx3rny74xnib.streamlit.app/
# ピークサーチ機能なし
# csvファイルの冒頭を表示し, データ開始行およびデータ列を指定する

### 要改善点
# データ分析範囲を指定できるようにする（サンプル数：2^15個？）

'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import List, Tuple

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

def sidebar_fft_settings(usecols: List[int]) -> Tuple[int, int]:
    with st.sidebar:
        col = st.selectbox("FFT分析する列", options=usecols, format_func=lambda x: f"列 {x}")
        rate = st.number_input("サンプリング周波数 (Hz)", value=1000, min_value=1)
        return col, rate

def handle_fft_execution(df, fft_column, samplerate):
    with st.sidebar:
        if st.button("FFT分析実行"):
            freqs, result = GetFFT(df[fft_column], samplerate).results
            st.session_state.fft_done = True
            st.session_state.frequencies = freqs
            st.session_state.fft_result = result
            st.session_state.fft_column = fft_column
            st.session_state.max_freq = int(freqs.max())

def show_time_series_plot():
    st.subheader("選択列の時系列データ")
    fig, ax = plt.subplots(figsize=(15, 5))
    for col in st.session_state.usecols:
        ax.plot(st.session_state.df[col], label=f'Column {col}', alpha=0.7)
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
                                   value=int(st.session_state.frequencies.max()),
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

# --- アプリ本体 ---
def main():
    st.title("FFT分析アプリケーション")

    with st.sidebar:
        st.header("設定")
        uploaded_file = st.file_uploader("CSVファイルを選択してください", type=['csv'])

    if uploaded_file:
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.clear()
            st.session_state.last_uploaded_file = uploaded_file

        try:
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

            with st.sidebar:
                if st.button("時系列データを表示", key="show_raw_data"):
                    st.session_state.show_raw = True
                    st.session_state.df = df
                    st.session_state.usecols = usecols

            fft_column, samplerate = sidebar_fft_settings(usecols)

            if st.session_state.get("show_raw", False):
                show_time_series_plot()

            handle_fft_execution(df, fft_column, samplerate)

            if st.session_state.get("fft_done", False):
                show_fft_result_graph()

        except Exception as e:
            st.error(f"データの読み込み中にエラーが発生しました: {e}")
            st.stop()

if __name__ == "__main__":
    main()
