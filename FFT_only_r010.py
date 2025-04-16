# coding: utf-8
'''
FFT_only_r001.py
# 2025.4.14 r001デプロイ

# csvファイルをアップロードし, FFT分析結果をグラフで表示
# Streamlit用 url: https://app-dsjptfkljmnx3rny74xnib.streamlit.app/
# ピークサーチ機能なし
# csvファイルの冒頭を表示し, データ開始行およびデータ列を指定する

### 要改善点
# 新しくファイルを読み込めば, 時系列データグラフ以下はリセットさせる
# データ分析範囲を指定できるようにする（サンプル数：2^15個？）

'''


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
import csv
from typing import List, Tuple, Dict, Any, Optional

class EncodingChecker:
    def __init__(self, file_contents: bytes, encodings=None):
        self.file_contents = file_contents
        self.encodings = encodings or ['shift_jis', 'utf-8-sig', 'cp932', 'cp775', 'utf-8']
        self.encoding = self._detect_encoding()

    def _detect_encoding(self):
        for enc in self.encodings:
            try:
                decoded_content = self.file_contents.decode(enc)
                return enc
            except Exception:
                continue
        raise ValueError("全てのエンコーディングで読み込みに失敗しました。")

class GetFFT:
    def __init__(self, data, sampling_freq):
        self.data = data
        self.sampling_freq = sampling_freq
        self.results = self._frequency_component()

    def _frequency_component(self) -> Tuple[List[float], List[float]]:
        sampling_interval = 1 / self.sampling_freq
        fft_result = np.fft.fft(self.data)
        frequencies = np.fft.fftfreq(len(self.data), d=sampling_interval)

        positive_frequencies = frequencies[:len(frequencies)//2]
        positive_fft_result = np.abs(fft_result)[:len(frequencies)//2]
        filtered_frequencies = positive_frequencies[positive_frequencies > 0]
        filtered_fft_result = positive_fft_result[positive_frequencies > 0]

        return filtered_frequencies, filtered_fft_result

def plot_fft(frequencies, fft_result, graph_title, freq_max=100, acc_min=0, acc_max=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, freq_max)
    ax.set_ylim(acc_min, acc_max)
    ax.plot(frequencies, fft_result, label="FFT Result", alpha=0.7)
    ax.set_title(graph_title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration")
    ax.grid(True)
    ax.legend()
    return fig

def preview_csv_head(text_content: str) -> Tuple[pd.DataFrame, int]:
    lines = text_content.splitlines()
    preview_lines = lines[:20]
    split_rows = [line.split(',') for line in preview_lines]
    max_columns = max(len(row) for row in split_rows)
    normalized_rows = [row + [''] * (max_columns - len(row)) for row in split_rows]
    df_preview_raw = pd.DataFrame(normalized_rows)
    df_preview_raw.index.name = "行番号"
    df_preview_raw.columns = [f"列 {i}" for i in range(max_columns)]
    
    st.subheader("CSVファイルの冒頭20行")
    st.dataframe(df_preview_raw)
    return df_preview_raw, max_columns

def sidebar_skiprows() -> int:
    with st.sidebar:
        skiprows = st.number_input("データ開始行", value=0, min_value=0)
    return skiprows

def sidebar_usecols(df_preview: pd.DataFrame) -> List[int]:
    with st.sidebar:
        available_columns = list(range(df_preview.shape[1]))
        usecols = st.multiselect(
            "表示する列を選択",
            options=available_columns,
            default=[0],
            format_func=lambda x: f"列 {x}"
        )
        if not usecols:
            st.warning("少なくとも1つの列を選択してください")
            st.stop()
    return usecols

def sidebar_fft_settings(usecols: List[int]) -> Tuple[int, int]:
    with st.sidebar:
        fft_column = st.selectbox(
            "FFT分析する列",
            options=usecols,
            format_func=lambda x: f"列 {x}"
        )
        samplerate = st.number_input("サンプリング周波数 (Hz)", value=1000, min_value=1)
    return fft_column, samplerate

def handle_fft_execution(df, fft_column, samplerate):
    with st.sidebar:
        if st.button("FFT分析実行"):
            filtered_frequencies, filtered_fft_result = GetFFT(df[fft_column], samplerate).results
            st.session_state.fft_done = True
            st.session_state.frequencies = filtered_frequencies
            st.session_state.fft_result = filtered_fft_result
            st.session_state.fft_column = fft_column
            st.session_state.max_freq = int(filtered_frequencies.max())

def show_fft_result_graph():
    st.subheader(f"列 {st.session_state.fft_column} のFFT分析結果")

    yscale = st.radio(
        "**縦軸の目盛**",
        options=['linear', 'log'],
        format_func=lambda x: '方眼目盛' if x == 'linear' else '対数目盛',
        horizontal=True,
        index=0
    )

    col_freq_input, col_button = st.columns([1, 3])

    with col_freq_input:
        max_freq = st.number_input(
            label="**周波数表示範囲 (Hz)**",
            min_value=1,
            max_value=int(st.session_state.frequencies.max()),
            value=int(st.session_state.frequencies.max()),
            step=1,
            key="max_freq_input"
        )

    with col_button:
        st.markdown(" ")  # ← これでボタンの位置を下に下げる
        update_pressed = st.button("グラフを更新", key="update_graph")
        if update_pressed:
            st.session_state.max_freq = max_freq

    if 'max_freq' not in st.session_state:
        st.session_state.max_freq = max_freq

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(st.session_state.frequencies,
            st.session_state.fft_result,
            label="FFT Result",
            alpha=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Column {st.session_state.fft_column} FFT Result")
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.set_yscale(yscale)
    ax.set_xlim(0, st.session_state.max_freq)
    st.pyplot(fig)

def show_time_series_plot():
    st.subheader("選択列の時系列データ")
    fig_raw, ax_raw = plt.subplots(figsize=(15, 5))
    for col in st.session_state.usecols:
        ax_raw.plot(st.session_state.df[col], label=f'Column {col}', alpha=0.7)
    ax_raw.set_xlabel("Sample")
    ax_raw.set_ylabel("Acceleration")
    ax_raw.legend(fontsize=16)
    ax_raw.grid(True)
    st.pyplot(fig_raw)

def main():
    st.title("FFT分析アプリケーション")

    with st.sidebar:
        st.header("設定")
        uploaded_file = st.file_uploader("CSVファイルを選択してください", type=['csv'])

    if uploaded_file is not None:
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.clear()
            st.session_state.last_uploaded_file = uploaded_file

    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read()
            encoding_checker = EncodingChecker(file_contents)
            text_content = file_contents.decode(encoding_checker.encoding)

            # ファイル読み込み後の最初
            df_preview_raw, max_columns = preview_csv_head(text_content)

            # 1. skiprows設定（先に取る）
            skiprows = sidebar_skiprows()

            # 2. データ開始行以降のプレビュー読み込み
            string_data = io.StringIO(text_content)
            string_data.seek(0)
            df_preview = pd.read_csv(
                string_data,
                nrows=5,
                skiprows=skiprows,
                encoding=encoding_checker.encoding,
                header=None,
                on_bad_lines='skip',
                sep=',',
                skipinitialspace=True
            )
            st.subheader(f"データ開始行({skiprows}行目)以降のプレビュー")
            st.dataframe(df_preview)

            # 3. usecols選択（df_previewに基づく）
            usecols = sidebar_usecols(df_preview)

            string_data.seek(0)
            df = pd.read_csv(
                string_data,
                skiprows=skiprows,
                usecols=usecols,
                encoding=encoding_checker.encoding,
                header=None,
                on_bad_lines='skip',
                sep=',',
                skipinitialspace=True
            )

            with st.sidebar:
                if st.button("時系列データを表示", key="show_raw_data"):
                    st.session_state.show_raw = True
                    st.session_state.df = df
                    st.session_state.usecols = usecols
                
            fft_column, samplerate = sidebar_fft_settings(usecols)

            if 'show_raw' in st.session_state and st.session_state.show_raw:
                show_time_series_plot()

            # FFT分析ボタンと処理
            handle_fft_execution(df, fft_column, samplerate)

            if 'fft_done' in st.session_state and st.session_state.fft_done:
                show_fft_result_graph()

        except Exception as e:
            st.error(f"データの読み込み中にエラーが発生しました: {e}")
            st.stop()

if __name__ == "__main__":
    main()