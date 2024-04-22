import lxml
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title('성남(토요)리그 DATA')

# DATE_COLUMN = 'date/time'
team_id_dict = {
    "코메츠 호시탐탐": 7984,
    "Big Hits": 36636,
    "FA Members": 13621,
    "RedStorm": 17375,
    "unknown`s": 33848,
    "그냥하자": 10318,
    "기드온스": 27811,
    "다이아몬스터": 39783,
    "데빌베어스(Devil Bears)": 19135,
    "라이노즈": 41236,
    "미파스": 19757,
    "분당스타즈": 34402,
    "블루레이커즈": 22924,
    "성시야구선교단": 29105,
    "와사비": 14207,
}

team_name = st.selectbox(
    '팀 선택',
    (team_id_dict.keys()))

# st.write('You selected TEAM:', option)

team_id = team_id_dict[team_name]

DATA_URL_B = "http://www.gameone.kr/club/info/ranking/hitter?club_idx={}".format(team_id)
DATA_URL_P = "http://www.gameone.kr/club/info/ranking/pitcher?club_idx={}".format(team_id)

# @st.cache_data
def load_data(url):
    # st.write(url)
    # 웹 페이지에서 데이터 가져오기
    response = requests.get(url)
    tables = pd.read_html(response.text)

    # 데이터 확인
    for i, table in enumerate(tables):
        # 정규 표현식을 사용하여 table 내 '이름' 열의 데이터를 분리
        extracted_df = table['이름'].str.extract(r"(\w+)\((\d+)\)")
        # 새로운 열 이름 지정
        extracted_df.columns = ['성명', '배번']
        extracted_df = extracted_df[['배번', '성명']]
        extracted_df['배번'] = extracted_df['배번'].astype('int')
        table = pd.concat([extracted_df, table.drop(['이름'], axis = 1)], axis = 1)
        if i == 0:
            above_min = table
            if 'hitter' in url :
                above_min = above_min.rename(columns={'게임수': '경기'}, inplace=False)
            else:
                above_min = above_min.rename(columns={'게임수': '경기수'}, inplace=False)                
        else :
            below_min = table
    return above_min, below_min

data_b = load_data(DATA_URL_B)
data_p = load_data(DATA_URL_P)
# st.write(data_p)

df_b_concat = pd.concat([data_b[0], data_b[1]], axis = 0).reset_index(drop=True)
df_b_concat = df_b_concat.drop(['순위'], axis = 1)

df_p_concat = pd.concat([data_p[0], data_p[1]], axis = 0).reset_index(drop=True)
df_p_concat = df_p_concat.drop(['순위'], axis = 1)
#t.write(df_p_concat)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["[전체타자]", "[전체투수]", "타자:규정이상", "타자:규정미만", "투수:규정이상", "투수:규정미달"])

with tab1:
    st.subheader('전체 타자 : {}'.format(team_name))
    st.dataframe(df_b_concat)
    st.write(DATA_URL_B)

with tab2:
    st.subheader('전체 투수 : {}'.format(team_name))
    st.dataframe(df_p_concat) 
    st.write(DATA_URL_P)

with tab3:
   st.subheader('규정타석 이상 : {}'.format(team_name))
   st.write(data_b[0])

with tab4:
   st.subheader('규정타석 미만 : {}'.format(team_name))
   st.write(data_b[1]) 

with tab5:
    st.subheader('규정이닝 이상 : {}'.format(team_name))
    st.write(data_p[0])   

with tab6:
    st.subheader('규정이닝 미달 : {}'.format(team_name))
    st.write(data_p[1])       
