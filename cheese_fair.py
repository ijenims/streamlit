import streamlit as st
# import plotly.express as px
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# 画面設定
# st.set_page_config(layout='wide')

# 初期設定
data = pd.DataFrame()


## サイドバー

# ファイルアップロードウィジェット
uploaded_file = st.sidebar.file_uploader("Excelファイルを選択してください", type='xlsx')


### メイン画面

# アプリケーションのタイトルを設定
st.title('Cheese Fair 2024')


# ファイルがアップロードされたら処理を実行
if uploaded_file is not None:
    # ExcelファイルをPandasのデータフレームに読み込む
    data = pd.read_excel(uploaded_file, sheet_name='input').T
    
    # 最初の行を列名に設定
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)

    # 日付文字列をdatetime型に変換
    # datetime型から必要なフォーマット（'m/d'）に変換
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.strftime('%m/%d')


    ## chart_dataの作成
    chart_cols = data.columns[4:]
    # st.text(chart_cols)
    df = pd.DataFrame()
    df = data.groupby('date')[chart_cols].sum()

    ## chart表示
    # 日毎の注文数（グラフ）
    st.text('日毎の注文数')
    st.bar_chart(df, height=500)

    # チーズ毎の注文数（グラフ）
    st.text('チーズ毎の注文数')
    st.bar_chart(df.T, height=500)

    # chart_dataの表示
    st.text('グラフデータ')
    st.dataframe(df)

    # data表示
    st.text('入力データ')
    st.dataframe(data)


    # グラフ表示
    #fig, ax = plt.subplots(figsize=(15,4))
    #ax.plot(data['dist'])
    #st.pyplot(fig)






