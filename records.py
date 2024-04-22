import lxml
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title('GAMEONE Batter DATA')

# DATE_COLUMN = 'date/time'
DATA_URL = "http://www.gameone.kr/club/info/ranking/hitter?club_idx=7984"

# @st.cache_data
def load_data(url = DATA_URL):
    # data = pd.read_csv(DATA_URL, nrows=nrows)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    # 웹 페이지에서 데이터 가져오기
    response = requests.get(url)
    tables = pd.read_html(response.text)

    # 데이터 확인
    for i, table in enumerate(tables):
        if i == 0:
            above_minPA = table
            # print(above_minPA.head(2))
            above_minPA = above_minPA.rename(columns={'게임수': '경기'}, inplace=False)
            # print(above_minPA.head(2))
        else :
            below_minPA = table
    return above_minPA, below_minPA

# Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()

df_concat = pd.concat([data[0], data[1]], axis = 0).reset_index(drop=True)
df_concat = df_concat.drop(['순위'], axis = 1)

# Notify the reader that the data was successfully loaded.
# data_load_state.text('Loading data...done!')
# data_load_state.text("Done! (using st.cache_data)")

st.subheader('Raw entire data')
st.write(df_concat)

if st.checkbox('Show raw data(Above PA)'):
    st.subheader('규정 타석 이상')
    st.write(data[0])

if st.checkbox('Show raw data(Below PA)'):
    st.subheader('규정 타석 미만')
    st.write(data[1])
