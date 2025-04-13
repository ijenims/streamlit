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
                # バイトデータを指定のエンコーディングでデコード
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
    
    # サイドバーにファイルアップロードと設定項目をまとめる
    with st.sidebar:
        st.header("設定")
        uploaded_file = st.file_uploader("CSVファイルを選択してください", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # ファイルの内容を一度だけ読み込んでメモリに保持
            file_contents = uploaded_file.read()
            
            # エンコーディングチェッカーの初期化
            encoding_checker = EncodingChecker(file_contents)
            
            # バイトデータをデコード
            text_content = file_contents.decode(encoding_checker.encoding)
            
            # 区切り文字を自動検出
            dialect = csv.Sniffer().sniff(text_content[:1024])

            # StringIOを使用してテキストデータをCSVとして読み込む
            string_data = io.StringIO(text_content)

            # サイドバーに各種設定を配置
            with st.sidebar:
                # データ開始行の設定
                skiprows = st.number_input("データ開始行", value=0, min_value=0)

                # プレビュー用のデータフレーム作成
                df_preview = pd.read_csv(
                    string_data,
                    nrows=5,
                    encoding=encoding_checker.encoding,
                    header=None,
                    on_bad_lines='skip',
                    sep=dialect.delimiter
                )

                # StringIOを先頭に戻す
                string_data.seek(0)

                # 利用可能な列を表示
                if df_preview.shape[1] > 0:
                    available_columns = list(range(df_preview.shape[1]))
                    
                    # 複数列の選択を可能にする
                    usecols = st.multiselect(
                        "表示する列を選択",
                        options=available_columns,
                        default=[0],
                        format_func=lambda x: f"列 {x}"
                    )

                    if not usecols:
                        st.warning("少なくとも1つの列を選択してください")
                        st.stop()

            # メイン画面にデータプレビューを表示
            st.subheader("データプレビュー")
            st.dataframe(df_preview)

            # 完全なデータ読み込み
            string_data.seek(0)
            df = pd.read_csv(
                string_data,
                skiprows=skiprows,
                usecols=usecols,
                encoding=encoding_checker.encoding,
                header=None,
                on_bad_lines='skip',
                sep=dialect.delimiter
            )

            with st.sidebar:
                # FFT分析用の列を1つ選択
                fft_column = st.selectbox(
                    "FFT分析する列",
                    options=usecols,
                    format_func=lambda x: f"列 {x}"
                )

                # サンプリング周波数の入力
                samplerate = st.number_input("サンプリング周波数 (Hz)", 
                                           value=1000, 
                                           min_value=1)

                # FFT分析実行ボタン
                if st.button("FFT分析実行"):
                    # FFT実行
                    filtered_frequencies, filtered_fft_result = GetFFT(df[fft_column], samplerate).results
                    
                    # セッションステートにFFT結果を保存
                    st.session_state.fft_done = True
                    st.session_state.frequencies = filtered_frequencies
                    st.session_state.fft_result = filtered_fft_result
                    st.session_state.fft_column = fft_column
                    
                    # 初期の最大周波数を設定
                    st.session_state.max_freq = int(filtered_frequencies.max())

            # メイン画面にグラフを表示
            st.subheader("選択された列のデータ")
            fig_raw, ax_raw = plt.subplots(figsize=(15, 5))
            for col in usecols:
                ax_raw.plot(df[col], label=f'Column {col}', alpha=0.7)
            ax_raw.set_xlabel('Sample')
            ax_raw.set_ylabel('Amplitude')
            ax_raw.legend()
            ax_raw.grid(True)
            st.pyplot(fig_raw)

            # FFT実行済みの場合、グラフ表示部分を実行
            if 'fft_done' in st.session_state and st.session_state.fft_done:
                st.subheader(f"列 {st.session_state.fft_column} のFFT分析結果")

                # グラフ設定用のコントロール
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Y軸スケールの選択（ラジオボタン）
                    yscale = st.radio(
                        "縦軸の目盛り",
                        options=['linear', 'log'],
                        format_func=lambda x: '方眼目盛' if x == 'linear' else '対数目盛',
                        horizontal=True,
                        index=0
                    )
                
                with col2:
                    # 横軸の最大値設定
                    max_freq = st.number_input(
                        "周波数表示範囲 (Hz)",
                        min_value=1,
                        max_value=int(st.session_state.frequencies.max()),
                        value=int(st.session_state.frequencies.max()),
                        step=1
                    )

                # グラフ更新ボタン
                if st.button("グラフを更新", key="update_graph"):
                    st.session_state.max_freq = max_freq
                
                # 初回表示時の設定
                if 'max_freq' not in st.session_state:
                    st.session_state.max_freq = max_freq
                
                # グラフ描画
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
                
                # スケール設定
                ax.set_yscale(yscale)
                ax.set_xlim(0, st.session_state.max_freq)
                
                st.pyplot(fig)

        except Exception as e:
            st.error(f"データの読み込み中にエラーが発生しました: {e}")
            st.stop()

if __name__ == "__main__":
    main()
