import streamlit as st
import time

st.title('streamlit 超入門')

# 2023.4.15
st.write('Progress Bar')
'Start!!'

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)

'Done!!!'

st.write('Display Image')

st.write('Interactive Widgets')

left_column, right_column = st.columns(2)
button = left_column.button('display Right Column')
if button:
    right_column.write("Here's Right Column")

exp = st.expander('Question')
exp.write('contents')

text = st.text_input('your hobby?')
condition = st.slider('Your condition?', 0, 100, 50)

'Your hobby is ', text
'Condition : ', condition

option = st.selectbox(
    'your number',
    list(range(1, 11))
)

'your number is ', option, '!'

if st.checkbox('show Image'):

    img = Image.open('Malgoire-Mattheus-L.jpg')
    st.image(img, caption='matthew', use_column_width=True)


# 2023.4.12

st.write('DataFrame')

df = pd.DataFrame({
    '1st': [1, 2, 3, 4],
    '2nd': [10, 20, 30, 40]
})

st.write(df)

st.dataframe(df.style.highlight_max(axis=0),
             width=200, height=200)  # interactive

st.table(df.style.highlight_max(axis=0))  # static

"""
# section
## section
### section

```python
import streamlit as st
import numpy as np
import pandas as pd
```
"""

df1 = pd.DataFrame(
    np.random.rand(100, 2) / [50, 50] + [35.69, 139.7],
    columns=['lat', 'lon']
)

st.map(df1)
