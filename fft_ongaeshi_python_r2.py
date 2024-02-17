import streamlit as st
# import plotly.express as px
import pandas as pd
import numpy as np
# from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 画面サイズをwideに設定
# st.set_page_config(layout='wide')


### 関数定義
# 顕著なピークを検出するための基準周波数リスト（これの算出はノウハウにつき非開示）
base_frequencies = [3.28, 6.58, 9.94, 13.42, 20.62, 24.33, 28.23, 36.97, 17.03]


# 窓関数を使用して特定の周波数範囲内のピークを見つける関数を定義
def find_peak_within_window(frequencies, fft_result, center_freq, window_width=1.5):
    lower_bound = center_freq - window_width
    upper_bound = center_freq + window_width
    window_indices = (frequencies >= lower_bound) & (frequencies <= upper_bound)
    window_frequencies = frequencies[window_indices]
    window_fft_result = fft_result[window_indices]
    peaks_in_window, _ = find_peaks(window_fft_result, height=max(window_fft_result)*0.1)
    if peaks_in_window.size > 0:
        max_peak_index = np.argmax(window_fft_result[peaks_in_window])
        return window_frequencies[peaks_in_window[max_peak_index]]
    else:
        return None


# 検出されたピークをグラフに表示し、グラフを保存する関数を定義
def plot_fft_and_mark_peaks(frequencies, fft_result):   # , peak_freqs
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(frequencies, fft_result, label="FFT Result")

    #for peak in peak_freqs:
    #    if peak:  # Noneでない値のみプロット
    #        ax.plot(peak, np.interp(peak, frequencies, fft_result), "x", color='red')
    #        ax.text(peak, np.interp(peak, frequencies, fft_result) + 5, f'{peak:.2f} Hz', color='red', ha='center')
    
    ax.set_title("FFT Analysis with Marked Peaks")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# 曲げ剛性を算出する関数
def virtual_flexural_rigidity(w):
    Dt = np.sqrt(4 * w / (rho * np.pi))    # 理論ケーブル径 m
    I = np.pi * Dt ** 4 / 64    # 断面2次モーメント m4
    EI = E * I    # 曲げ剛性 N・m2

    return Dt, I, EI


## ケーブル諸元のインプット
#
# 定数定義
rho = 7850    # density kg/m3
E = 196000_000000    # young's modulus N/m2

# DataFrame作成 'inp'
# セッションステートでデータフレームを初期化
if 'inp' not in st.session_state:
    st.session_state.inp = pd.DataFrame({'c_No': 'C0', 'w': 4.8, 'T': 47.45, 'L': 22.573, 
                                        'Dt': 0, 'I': 0, 'EI': 0, 'xi': 0,
                                        '1st': 0, '2nd': 0}, index=[0])


### User Interface
## サイドバー
# ファイルアップロードウィジェット
uploaded_file = st.sidebar.file_uploader("CSVファイルを選択してください", type='csv')

c_no = st.sidebar.text_input('Cable No.')
weight = st.sidebar.number_input('単位重量w[kgf/m]')
Tension = st.sidebar.number_input('設計張力T[Ton]')
Length = st.sidebar.number_input('支間長L[m]')

if st.sidebar.button('追加'):
    Dt, I, EI = virtual_flexural_rigidity(weight)

    new_row = {'c_No': c_no, 'w': weight, 'T': Tension, 'L': Length,
                'Dt': Dt, 'I': I, 'EI': EI, 'xi': 0,
                }
    st.session_state.inp = st.session_state.inp.append(new_row, ignore_index=True)


## メイン画面設定
# アプリケーションのタイトルを設定
st.title('FFT Analyzer')

# DataFrame の画面表示
st.text('ケーブルデータ')
st.write(st.session_state.inp)

# DataFrame の保存
if st.button('保存'):
    pass


# ファイルがアップロードされたら処理を実行
if uploaded_file is not None:
    # CSVファイルをPandasのデータフレームに読み込む
    data = pd.read_csv(uploaded_file)

    # Plotlyでグラフを作成  
    # fig = px.line(data, x="time", y="dist", title="振動データ")

    # X軸のみズーム可能に設定 , type="-"
    # fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

    # グラフのレイアウトを設定（ここでは自動サイズ調整を有効にする）
    #fig.update_layout(autosize=True)

    # Streamlitでグラフを表示 , use_container_width=True
    # st.plotly_chart(fig)

    # FFTを実行し、周波数成分を取得
    time_diffs = data['time'].diff().dropna()
    mean_sampling_interval = time_diffs.mean()
    fft_result = np.fft.fft(data['dist'])
    frequencies = np.fft.fftfreq(len(data), d=mean_sampling_interval)

    # 正の周波数成分のみを取得し、0 Hzを除外
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_fft_result = np.abs(fft_result)[:len(frequencies)//2]
    filtered_frequencies = positive_frequencies[positive_frequencies > 0]
    filtered_fft_result = positive_fft_result[positive_frequencies > 0]

    # 各基準周波数に対して窓内でピークを検出
    # final_peaks = [find_peak_within_window(filtered_frequencies, filtered_fft_result, freq) for freq in base_frequencies]

    # グラフを描画
    plot_fft_and_mark_peaks(filtered_frequencies, filtered_fft_result)  # , final_peaks

    ## ピークリストを表示
    # 検出されたピークのリストを表示
    #final_peaks = [peak for peak in final_peaks if peak is not None]  # Noneを除外
    #final_peaks.sort()  # リストをソート

    # ピークリストをデータフレーム化
    #df_nf = pd.DataFrame(final_peaks, columns=['固有振動数nf'])
    #st.write(df_nf)

# この後、固有振動数より張力を算出する関数があるが、ノウハウにつき非開示とする
