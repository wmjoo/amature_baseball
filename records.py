import lxml
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

st.set_page_config(
        page_title="Baseball Data",
)
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

ALB_URL_SCHD = "http://alb.or.kr/s/schedule/schedule_team_2019.php?id=schedule_team&sc=2&team=%B7%B9%BE%CB%B7%E7%C5%B0%C1%EE&gyear=2024"

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["[전체타자]", "[전체투수]", "타자:규정이상", "타자:규정미만", "투수:규정이상", "투수:규정미달", "안양_일정"])

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

with tab7:
    # # 페이지 URL
    # url = "http://alb.or.kr/s/schedule/schedule_team_2019.php?id=schedule_team&sc=2&team=%B7%B9%BE%CB%B7%E7%C5%B0%C1%EE&gyear=2024"
    
    # # requests를 사용해 웹 페이지 가져오기
    # response = requests.get(url)
    # response.encoding = 'euc-kr' # 인코딩 설정
    
    # # BeautifulSoup 객체 생성
    # soup = BeautifulSoup(response.text, 'html.parser')
    
    # # 모든 테이블을 찾기
    # tables = soup.find_all('table')
    # table = tables[6]
    
    # # 데이터프레임 리스트 생성
    # dataframes = []
    
    # rows = []
    # for tr in table.find_all('tr'):
    #     cells = [td.get_text(strip=True) for td in tr.find_all('td')]
    #     if cells:
    #         rows.append(cells)
    
    # # 데이터프레임 생성
    # df = pd.DataFrame(rows) 
    # dataframes.append(df)
    # df = dataframes[0]
    # del rows
    
    # df.columns = df.iloc[3].tolist()
    # df2 = df.iloc[4:]
    # df3 = df2.loc[:, ~df2.isna().all(0)].sort_values('날짜,시간').reset_index(drop=True)
    # df3.columns = ['No', '일시', '1루', '3루', '대회타이틀', '구장', '진행', '비고']
    # df3['일시'] = df3['일시'].str.replace(' -m', '')
    
    # df3['일자'] = df3['일시'].apply(lambda x: x[:-6])
    # df3['시간'] = df3['일시'].apply(lambda x: x[-5:])
    # df3['1루'] = df3['1루'].apply(lambda x: x[:-1] if x.endswith('d') else x)
    # df3['3루'] = df3['3루'].apply(lambda x: x[:-1] if x.endswith('d') else x)
    # df4 = df3.loc[:, ['No', '일자', '시간', '진행', '1루', '3루', '대회타이틀', '구장', '비고']]
    
    st.subheader('안양리그 일정[2024]')
    # st.write(df4)
    # data_alb_scd = load_data(ALB_URL_SCHD)
    # for i in range(len(data_alb_scd)):
    #     st.write(data_alb_scd[i])
        
