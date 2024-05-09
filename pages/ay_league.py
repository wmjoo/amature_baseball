import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# 
try:
    @st.cache_data
    def get_table_as_dataframe(url, table_index=7):
        # URL에서 HTML 페이지를 가져옵니다.
        response = requests.get(url)
        # HTML을 파싱합니다.
        soup = BeautifulSoup(response.text, 'html.parser')
        # 모든 테이블을 찾습니다.
        tables = soup.find_all('table')
        # # 특정 인덱스의 테이블을 DataFrame으로 변환합니다.
        try:
            df = pd.read_html(str(tables[table_index]))[0]
            return df
        except IndexError:
            print(f"No table at index {table_index}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # URL과 테이블 인덱스를 지정하여 사용합니다.
    ALB_URL_SCHD = "http://alb.or.kr/s/schedule/schedule_team_2019.php?id=schedule_team&sc=2&team=%B7%B9%BE%CB%B7%E7%C5%B0%C1%EE&gyear=2024"
    # 데이터프레임을 가져옵니다.
    df_schd = get_table_as_dataframe(url = ALB_URL_SCHD, table_index = 9) # 7 or 9?
    # df_schd.columns = df_schd.loc[0]
    # df_schd = df_schd.iloc[1:].sort_values('날짜,시간').reset_index(drop=True)
    # df_schd2
    st.subheader('안양리그 경기 일정')
    st.dataframe(df_schd, use_container_width = True, hide_index = True)
except Exception as e:
    st.write('🚧 Under Construction ... 🚧')
    st.write(f"An error occurred: {e}")