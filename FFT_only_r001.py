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

def main():
    st.title("FFT分析アプリケーション")

    with st.sidebar:
        st.header("設定")
        uploaded_file = st.file_uploader("CSVファイルを選択してください", type=['csv'])

    if uploaded_file is not None:
        try:
            file_contents = uploaded_file.read()
            encoding_checker = EncodingChecker(file_contents)
            text_content = file_contents.decode(encoding_checker.encoding)

            # 1. CSVファイルの冒頭15行プレビュー（表形式）
            lines = text_content.splitlines()
            preview_lines = lines[:15]
            split_rows = [line.split(',') for line in preview_lines]
            max_columns = max(len(row) for row in split_rows)
            normalized_rows = [row + [''] * (max_columns - len(row)) for row in split_rows]
            df_preview_raw = pd.DataFrame(normalized_rows)
            df_preview_raw.index.name = "行番号"
            df_preview_raw.columns = [f"列 {i}" for i in range(max_columns)]
            st.subheader("🔍 CSVファイルの冒頭15行")
            st.dataframe(df_preview_raw)

            # 文字列を再度読み込み用に変換
            string_data = io.StringIO(text_content)

            # サイドバー：データ開始行を指定
            with st.sidebar:
                skiprows = st.number_input("データ開始行", value=0, min_value=0)

            # 2. データ開始行以降のプレビュー（表形式）
            string_data.seek(0)
            df_preview = pd.read_csv(
                string_data,
                nrows=15,
                skiprows=skiprows,
                encoding=encoding_checker.encoding,
                header=None,
                on_bad_lines='skip',
                sep=',',
                skipinitialspace=True
            )
            st.subheader(f"📊 データ開始行({skiprows}行目)以降のプレビュー")
            st.dataframe(df_preview)

            # 列選択
            with st.sidebar:
                if df_preview.shape[1] > 0:
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

            # フルデータ読み込み
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
                # 時系列データ表示ボタン
                if st.button("時系列データを表示", key="show_raw_data"):
                    st.session_state.show_raw = True
                    st.session_state.df = df
                    st.session_state.usecols = usecols
                
                fft_column = st.selectbox(
                    "FFT分析する列",
                    options=usecols,
                    format_func=lambda x: f"列 {x}"
                )
                samplerate = st.number_input("サンプリング周波数 (Hz)", value=1000, min_value=1)

            # 時系列データのグラフ表示
            if 'show_raw' in st.session_state and st.session_state.show_raw:
                st.subheader("📈 選択列の時系列データ")
                fig_raw, ax_raw = plt.subplots(figsize=(15, 5))
                for col in st.session_state.usecols:
                    ax_raw.plot(st.session_state.df[col], label=f'Column {col}', alpha=0.7)
                ax_raw.set_xlabel('Sample')
                ax_raw.set_ylabel('Amplitude')
                ax_raw.legend()
                ax_raw.grid(True)
                st.pyplot(fig_raw)

            with st.sidebar:
                if st.button("FFT分析実行"):
                    filtered_frequencies, filtered_fft_result = GetFFT(df[fft_column], samplerate).results
                    st.session_state.fft_done = True
                    st.session_state.frequencies = filtered_frequencies
                    st.session_state.fft_result = filtered_fft_result
                    st.session_state.fft_column = fft_column
                    st.session_state.max_freq = int(filtered_frequencies.max())

            if 'fft_done' in st.session_state and st.session_state.fft_done:
                st.subheader(f"列 {st.session_state.fft_column} のFFT分析結果")
                yscale = st.radio(
                    "縦軸の目盛り",
                    options=['linear', 'log'],
                    format_func=lambda x: '方眼目盛' if x == 'linear' else '対数目盛',
                    horizontal=True,
                    index=0
                )
                col1, col2 = st.columns([2, 1])
                with col1:
                    max_freq = st.number_input(
                        "周波数表示範囲 (Hz)",
                        min_value=1,
                        max_value=int(st.session_state.frequencies.max()),
                        value=int(st.session_state.frequencies.max()),
                        step=1
                    )
                with col2:
                    st.write("")
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
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Column {st.session_state.fft_column} FFT Result")
                ax.grid(True)
                ax.legend()
                ax.set_yscale(yscale)
                ax.set_xlim(0, st.session_state.max_freq)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"データの読み込み中にエラーが発生しました: {e}")
            st.stop()

if __name__ == "__main__":
    main()
