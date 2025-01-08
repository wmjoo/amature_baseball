import os
import time
import lxml
import streamlit as st

from datetime import datetime 
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

from streamlit_gsheets import GSheetsConnection
# import pandasql as psql



warnings.filterwarnings('ignore')
st.set_page_config(page_title="Baseball Data")
st.title('Sat League Data')

## 성남리그 팀 딕셔너리 및 영문 그래프용 리스트
team_id_dict_rkA = { # team_id_dict_rookieA
    "코메츠 호시탐탐": 7984, "Big Hits": 36636,    "FA Members": 13621, "RedStorm": 17375,    "unknown`s": 33848, "그냥하자": 10318,
    "기드온스": 27811,    "다이아몬스터": 39783,     "데빌베어스": 19135, "라이노즈": 41236,    "미파스": 19757,    "분당스타즈": 34402,
    "블루레이커즈": 22924,    "성시야구선교단": 29105,    "와사비": 14207, # "SKCC Wings": 4653,
}
team_id_dict_rkB = {
    '보성야구단': 15977,
    '데빌베어스(Devil Bears)': 19135,
    'FA Members': 13621,
    'Team 야놀자': 39918,
    '슈퍼스타즈': 23785,
    'MANNA ECCLESIA': 43133,
    '성남야구선수촌': 7072,
    '라이노즈': 41326,
    '에자이갑스': 23042,
    '실버서울 야구단': 15753,
    '야호 이겨스': 42160,
    '마자야지': 19163,
    '다이아몬스터': 39783,
    'HEAT': 18414
}

team_id_dict = team_id_dict_rkB.copy()
team_id_dict.setdefault('SKCC Wings', 4653) 
rank_calc_except_teams = list(team_id_dict.keys() - team_id_dict_rkB.keys())

team_englist = ['BoseongBaseballTeam', 'DevilBears', 'FAMembers', 'TeamYnj', 'Superstars', 'MANNAECCLESIA', 
                'SeongnamYgssc', 'Rhinos', 'EisaiGabs', 'SilverSeoul', 'Yaho', 'MajaYaji', 'Diamonster', 'HEAT', "KometsHSTT"] #, "SKCC Wings"]

# ["Big Hits", "FA Members", "Red Storm", "unknown`s", "GNHaJa", "Gideons", "Diamonster", "DevilBears",
                #  "Rhinos", "Mifas", "Bundang Stars", "Blue Lakers", "Sungsi YGSG", "Wasabi", "KometsHSTT"] #, "SKCC Wings"]

# 타자 데이터프레임 df에 적용할 자료형 / 컬럼명 딕셔너리 정의
hitter_data_types = {
    '성명': 'str', '배번': 'str', '타율': 'float', '경기': 'int', '타석': 'int', '타수': 'int',
    '득점': 'int', '총안타': 'int', '1루타': 'int', '2루타': 'int', '3루타': 'int', '홈런': 'int',
    '루타': 'int', '타점': 'int', '도루': 'int', '도실(도루자)': 'int', '희타': 'int', '희비': 'int',
    '볼넷': 'int', '고의4구': 'int', '사구': 'int', '삼진': 'int', '병살': 'int', '장타율': 'float',
    '출루율': 'float', '도루성공률': 'float', '멀티히트': 'int', 'OPS': 'float', 'BB/K': 'float',
    '장타/안타': 'float', '팀': 'str'
}
hitter_data_KrEn = {
    '성명': 'Name', '배번': 'No', '타율': 'AVG', '경기': 'G', '타석': 'PA', '타수': 'AB',
    '득점': 'R', '총안타': 'H', '1루타': '1B', '2루타': '2B', '3루타': '3B', '홈런': 'HR',
    '루타': 'TB', '타점': 'RBI', '도루': 'SB', '도실(도루자)': 'CS', '희타': 'SH', '희비': 'SF',
    '볼넷': 'BB', '고의4구': 'IBB', '사구': 'HBP', '삼진': 'SO', '병살': 'DP', '장타율': 'SLG', '출루율': 'OBP', '도루성공률': 'SB%', '멀티히트': 'MHit', 'OPS': 'OPS', 'BB/K': 'BB/K',
    '장타/안타': 'XBH/H', '팀': 'Team'
}
hitter_data_EnKr = {'Name': '성명', 'No': '배번', 'AVG': '타율', 'G': '경기', 'PA': '타석', 'AB': '타수', 'R': '득점', 
                    'H': '총안타', '1B': '1루타', '2B': '2루타', '3B': '3루타', 'HR': '홈런', 'TB': '루타', 'RBI': '타점', 
                    'SB': '도루', 'CS': '도실', 'SH': '희타', 'SF': '희비', 'BB': '볼넷', 'IBB': '고의4구', 'HBP': '사구', 'SO': '삼진', 'DP': '병살', 'SLG': '장타율', 'OBP': '출루율', 'SB%': '도루성공률', 'MHit': '멀티히트', 'OPS': 'OPS', 'BB/K': 'BB/K', 'XBH/H': '장타/안타', 'Team': '팀'}
# 투수 데이터프레임 df_pitcher에 적용할 자료형 / 컬럼명 딕셔너리 정의
pitcher_data_types = {
    '성명': 'str', '배번': 'str', '방어율': 'float', '경기수': 'int', '승': 'int', '패': 'int', '세': 'int',
    '홀드': 'int', '승률': 'float', '타자': 'int', '타수': 'int', '투구수': 'int', '이닝': 'float',
    '피안타': 'int', '피홈런': 'int', '희타': 'int', '희비': 'int', '볼넷': 'int', '고의4구': 'int',
    '사구': 'int', '탈삼진': 'int', '폭투': 'int', '보크': 'int', '실점': 'int', '자책점': 'int',
    'WHIP': 'float', '피안타율': 'float', '탈삼진율': 'float', '팀': 'str'
}
pitcher_data_KrEn = {
    '성명': 'Name', '배번': 'No', '방어율': 'ERA', '경기수': 'G', '승': 'W', '패': 'L', '세': 'SV',
    '홀드': 'HLD', '승률': 'WPCT', '타자': 'BF', '타수': 'AB', '투구수': 'P', '이닝': 'IP',
    '피안타': 'HA', '피홈런': 'HR', '희타': 'SH', '희비': 'SF', '볼넷': 'BB', '고의4구': 'IBB',
    '사구': 'HBP', '탈삼진': 'SO', '폭투': 'WP', '보크': 'BK', '실점': 'R', '자책점': 'ER',
    'WHIP': 'WHIP', '피안타율': 'BAA', '피장타율': 'SLG', '피출루율': 'OBP', '피OPS' : 'OPS', '탈삼진율': 'K9', '팀': 'Team'
}
pitcher_data_EnKr = {'Name': '성명', 'No': '배번', 'ERA': '방어율', 'G': '경기수', 'W': '승', 'L': '패', 'SV': '세', 'HLD': '홀드', 'WPCT': '승률', 
                     'BF': '타자', 'AB': '타수', 'P': '투구수', 'IP': '이닝', 'HA': '피안타', 'HR': '피홈런', 'SH': '희타', 'SF': '희비', 'BB': '볼넷', 'IBB': '고의4구', 'HBP': '사구', 
                     'SO': '탈삼진', 'WP': '폭투', 'BK': '보크', 'R': '실점', 'ER': '자책점', 'WHIP': 'WHIP', 'BAA': '피안타율', 'SLG':'피장타율', 'OBP':'피출루율', 'OPS' : '피OPS', 
                     'K9': '탈삼진율', 'Team': '팀'}

################################################################
## User def functions
################################################################
def create_heatmap(data, cmap, input_figsize = (10, 7)):
    plt.figure(figsize=input_figsize)
    sns.heatmap(data, annot=True, fmt=".0f", cmap=cmap, annot_kws={'color': 'black'}, yticklabels=data.index, cbar=False)
    plt.xticks(rotation=45)  # x축 레이블 회전
    plt.yticks(rotation=0)   # y축 레이블 회전
    plt.tight_layout()
    return plt

@st.cache_data
def load_data(team_name, team_id):
    urls = {
        'hitter': f"http://www.gameone.kr/club/info/ranking/hitter?club_idx={team_id}&kind=&season={default_year}",
        'pitcher': f"http://www.gameone.kr/club/info/ranking/pitcher?club_idx={team_id}&kind=&season={default_year}"
    }
    results = {'hitter': [], 'pitcher': []}
    for key, url in urls.items():
        response = requests.get(url)
        tables = pd.read_html(response.text)
        for table in tables:
            extracted_df = table['이름'].str.extract(r"(\w+)\((\d+)\)")
            extracted_df.columns = ['성명', '배번']
            extracted_df['배번'] = extracted_df['배번'].astype(int)
            table = pd.concat([extracted_df, table.drop(['이름'], axis=1)], axis=1)
            # 컬럼명 변경
            if '게임수' in table.columns:
                if key == 'hitter':
                    table.rename(columns={'게임수': '경기'}, inplace=True)
                else:
                    table.rename(columns={'게임수': '경기수'}, inplace=True)

            table['팀'] = team_name  # 팀 이름 컬럼 추가
            table = table.drop('순위', axis = 1)
            table.columns = [col.replace(" ", "") for col in table.columns]
            results[key].append(table)
    return {'hitter': pd.concat(results['hitter'], ignore_index=True), 
            'pitcher': pd.concat(results['pitcher'], ignore_index=True)}

################################################################
## Data Loading
################################################################
sn_standings_url = 'http://www.gameone.kr/league/record/rank?lig_idx=10373'

try:        # Create GSheets connection AND Load Data from google sheets 
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Read Google WorkSheet as DataFrame
    df_hitter = conn.read(worksheet="df_hitter")
    df_pitcher = conn.read(worksheet="df_pitcher")
    time.sleep(2)    
    st.toast('Loaded Data from Cloud!', icon='✅')
except Exception as e: ## 만약 csv 파일 로드에 실패하거나 에러가 발생하면 병렬로 데이터 로딩
    st.error(f"Failed to read data from drive: {e}", icon="🚨") 
    hitters = []
    pitchers = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_data, team_name, team_id): team_name for team_name, team_id in team_id_dict.items()}
        for future in as_completed(futures):
            try:
                result = future.result()
                hitters.append(result['hitter'])
                pitchers.append(result['pitcher'])
            except Exception as exc:
                print(f'Team {futures[future]} generated an exception: {exc}')
    # 모든 데이터를 각각의 데이터프레임으로 합침
    final_hitters_data = pd.concat(hitters, ignore_index=True)
    final_pitchers_data = pd.concat(pitchers, ignore_index=True)

    # 데이터프레임 df의 컬럼 자료형 설정
    df_hitter = final_hitters_data.astype(hitter_data_types)
    # 타자 데이터프레임 컬럼명 영어로
    df_hitter.columns = ['Name', 'No', 'AVG', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 
                         'SB', 'CS', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'DP', 'SLG', 'OBP', 'SB%', 'MHit', 
                         'OPS', 'BB/K', 'XBH/H', 'Team']

    final_pitchers_data.loc[final_pitchers_data.방어율 == '-', '방어율'] = np.nan

    # 투수 데이터프레임 df_pitcher의 컬럼 자료형 설정
    df_pitcher = final_pitchers_data.astype(pitcher_data_types)
    # 투수 데이터프레임 컬럼명 영어로
    df_pitcher.columns = ['Name', 'No', 'ERA', 'G', 'W', 'L', 'SV', 'HLD', 'WPCT', 
                          'BF', 'AB', 'P', 'IP', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'WP', 'BK', 
                        'R', 'ER', 'WHIP', 'BAA', 'K9', 'Team']
    # IP 컬럼을 올바른 소수 형태로 변환
    df_pitcher['IP'] = df_pitcher['IP'].apply(lambda x: int(x) + (x % 1) * 10 / 3).round(2)

    # Create GSheets connection
    conn = st.connection("gsheets", type=GSheetsConnection)

    # click button to update worksheet / This is behind a button to avoid exceeding Google API Quota
    if st.button("Loading Dataset"):
        try:
            df_hitter = conn.create(worksheet="df_hitter", data=df_hitter)
        except Exception as e:
            st.error(f"Failed to save df_hitter: {e}", icon="🚨")        
            df_hitter = conn.update(worksheet="df_hitter", data=df_hitter)
            st.toast('Updete Hitter Data from Web to Cloud!', icon='💾')
        
        try:
            df_pitcher = conn.create(worksheet="df_pitcher", data=df_pitcher)
        except Exception as e:
            st.error(f"Failed to save df_pitcher: {e}", icon="🚨")        
            df_pitcher = conn.update(worksheet="df_pitcher", data=df_pitcher)               
            st.toast('Updete Pitcher Data from Web to Cloud!', icon='💾')
        time.sleep(2)
        st.toast('Saved Data from Web to Cloud!', icon='💾')

################################################################
## UI Tab
################################################################
## 년도 설정
# default_year = 2024
default_year = st.selectbox('년도', [2025, 2024, 2023, 2022, 2021, 2020], key = 'year_selectbox')

## 탭 설정
tab_sn_players, tab_sn_teamwise, tab_sn_viz, tab_schd, tab_dataload, tab_sn_terms = st.tabs(["전체 선수", "팀별 선수", "시각화/통계", "일정", "업데이트", "약어"])

with tab_sn_players:
    tab_sn_players_1, tab_sn_players_2 = st.tabs(["성남:전체타자", "성남:전체투수"])
    with tab_sn_players_1:
        # 출력시 열 순서 변경
        rank_by_cols_h_sorted = ['Team', 'AVG', 'OBP', 'SLG', 'OPS', 'HR', 'SB', 'R', 'H', 'MHit', 
                                    '1B', '2B', '3B', 'TB', 'RBI', 'CS', 'SH', 'SF', 'BB', 'IBB', 
                                    'HBP', 'PA', 'AB', 'SO', 'DP']
        st.subheader('성남 : 전체타자 [{}명]'.format(df_hitter.shape[0]))
        st.dataframe(df_hitter[['No', 'Name'] + rank_by_cols_h_sorted].rename(columns = hitter_data_EnKr, inplace=False), 
                     use_container_width = True, hide_index = True)
        st.subheader('팀별 기록')
        hitter_sumcols = ['PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'DP', 'MHit']
        hitter_grpby = df_hitter.loc[~df_hitter['Team'].isin(rank_calc_except_teams), hitter_sumcols + ['Team']].groupby('Team').sum().reset_index()

        # 타율(AVG), 출루율(OBP), 장타율(SLG), OPS 계산 & 반올림
        hitter_grpby['AVG'] = (hitter_grpby['H'] / hitter_grpby['AB']).round(3)
        hitter_grpby['OBP'] = ((hitter_grpby['H'] + hitter_grpby['BB'] + hitter_grpby['HBP']) / (hitter_grpby['AB'] + hitter_grpby['BB'] + hitter_grpby['HBP'] + hitter_grpby['SF'])).round(3)
        hitter_grpby['SLG'] = (hitter_grpby['TB'] / hitter_grpby['AB']).round(3)
        hitter_grpby['OPS'] = (hitter_grpby['OBP'] + hitter_grpby['SLG']).round(3)
        
        # 'Team' 컬럼 바로 다음에 계산된 컬럼들 삽입
        for col in ['OPS', 'SLG', 'OBP', 'AVG']:
            team_idx = hitter_grpby.columns.get_loc('Team') + 1
            hitter_grpby.insert(team_idx, col, hitter_grpby.pop(col))
  
        # rank_by_ascending, rank_by_descending columns 
        rank_by_ascending_cols_h = ['SO', 'DP', 'CS'] # 낮을수록 좋은 지표들
        rank_by_descending_cols_h = ['AVG', 'OBP', 'SLG', 'OPS', 'PA', 'AB', 'R', 'H', 'MHit', 
                    '1B', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'SH', 'SF', 'BB', 'IBB', 'HBP'] # 높을수록 좋은 지표들
        st.dataframe(hitter_grpby.loc[:, rank_by_cols_h_sorted].rename(columns = hitter_data_EnKr, inplace=False), use_container_width = True, hide_index = True)
        hitter_grpby_rank = pd.concat([
                                        hitter_grpby.Team, 
                                        hitter_grpby[rank_by_descending_cols_h].rank(method = 'min', ascending=False),
                                        hitter_grpby[rank_by_ascending_cols_h].rank(method = 'min', ascending=True)
                                    ], axis = 1)
        hitter_grpby_rank = hitter_grpby_rank.loc[:, rank_by_cols_h_sorted]                                    
        st.write('Ranking')
        st.dataframe(hitter_grpby_rank.rename(columns = hitter_data_EnKr, inplace=False), use_container_width = True, hide_index = True)
        
        ## 히트맵 시각화 팀별 랭킹        
        st.write("Heatmap")
        df = hitter_grpby_rank.drop('Team', axis = 1).copy()  
        df['team_eng'] = team_englist
        df.set_index('team_eng', inplace=True)
        # 커스텀 컬러맵 생성
        colors = ["#8b0000", "#ffffff"]  # 어두운 빨간색에서 하얀색으로
        cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=15)
        # 히트맵 생성
        plt = create_heatmap(df, cmap, input_figsize = (10, 6))
        st.pyplot(plt)


    with tab_sn_players_2:
        # 출력시 열 순서 변경
        rank_by_cols_p_sorted = ['Team', 'IP', 'ERA', 'WHIP', 'H/IP', 'BB/IP', 'SO/IP', 'BAA', 'OBP', 'G', 'W', 'L', 'SV', 'HLD', 
                                 'SO', 'BF', 'AB', 'P', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'R', 'ER', 'K9']  
        st.subheader('성남 : 전체투수 [{}명]'.format(df_pitcher.shape[0]))
        pitcher_sumcols = df_pitcher.select_dtypes(include=['int64']).columns.tolist() + ['IP'] # Sum 컬럼 선택
        
        # 이닝당 삼진/볼넷/피안타 계산 (예제로 삼진(K), 볼넷(BB), 피안타(HA) 컬럼 필요)
        if 'SO' in df_pitcher.columns and 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
            df_pitcher['SO/IP'] = (df_pitcher['SO'] / df_pitcher['IP']).round(2)
            df_pitcher['BB/IP'] = (df_pitcher['BB'] / df_pitcher['IP']).round(2)
            df_pitcher['H/IP'] = (df_pitcher['HA'] / df_pitcher['IP']).round(2)
        
        # WHIP 계산: (볼넷 + 피안타) / 이닝
        if 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
            df_pitcher['WHIP'] = ((df_pitcher['BB'] + df_pitcher['HA']) / df_pitcher['IP']).round(3)
            df_pitcher['OBP'] = (df_pitcher['HA'] + df_pitcher['BB'] + df_pitcher['HBP']) / (df_pitcher['AB'] + df_pitcher['BB'] + df_pitcher['HBP'] + df_pitcher['SF'])
            # df_pitcher['SLG'] = (df_pitcher['HA'] + df_pitcher['2B']*2 + df_pitcher['3B']*3 + df_pitcher['HR']*4) / df_pitcher['AB']
            # df_pitcher['OPS'] = df_pitcher['OBP'] + df_pitcher['SLG']
            # st.write(df_pitcher[['OBP', 'SLG', 'OPS']])

        # None, '', '-'를 NaN으로 변환
        df_pitcher = df_pitcher.replace({None: np.nan, '': np.nan, '-': np.nan}) #, inplace=True)
        st.dataframe(df_pitcher[['No', 'Name'] + rank_by_cols_p_sorted].rename(columns = pitcher_data_EnKr, inplace=False), use_container_width = True, hide_index = True)

        # 팀별로 그룹화하고 정수형 변수들의 합계 계산
        st.subheader('팀별 기록 : 투수')
        pitcher_grpby = df_pitcher.loc[~df_pitcher['Team'].isin(rank_calc_except_teams), :].groupby('Team')[pitcher_sumcols].sum().reset_index()  # 팀별 합계
        # 파생 변수 추가
        # 방어율(ERA) 계산: (자책점 / 이닝) * 9 (예제로 자책점과 이닝 컬럼 필요)
        if 'ER' in df_pitcher.columns and 'IP' in df_pitcher.columns:
            pitcher_grpby['ERA'] = ((pitcher_grpby['ER'] / pitcher_grpby['IP']) * 9).round(3)
        
        # 이닝당 삼진/볼넷/피안타 계산 (예제로 삼진(K), 볼넷(BB), 피안타(HA) 컬럼 필요)
        if 'SO' in df_pitcher.columns and 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
            pitcher_grpby['SO/IP'] = (pitcher_grpby['SO'] / pitcher_grpby['IP']).round(2)
            pitcher_grpby['BB/IP'] = (pitcher_grpby['BB'] / pitcher_grpby['IP']).round(2)
            pitcher_grpby['H/IP'] = (pitcher_grpby['HA'] / pitcher_grpby['IP']).round(2)
            pitcher_grpby['K9'] = (pitcher_grpby['SO/IP'] * 9)
        
        # WHIP 계산: (볼넷 + 피안타) / 이닝
        if 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
            pitcher_grpby['WHIP'] = ((pitcher_grpby['BB'] + pitcher_grpby['HA']) / pitcher_grpby['IP']).round(3)
            pitcher_grpby['BAA'] = (pitcher_grpby['HA'] / pitcher_grpby['AB']).round(3)
            pitcher_grpby['OBP'] = (pitcher_grpby['HA'] + pitcher_grpby['BB'] + pitcher_grpby['HBP']) / (pitcher_grpby['AB'] + pitcher_grpby['BB'] + pitcher_grpby['HBP'] + pitcher_grpby['SF']).round(3)
            # pitcher_grpby['SLG'] = (pitcher_grpby['HA'] + pitcher_grpby['2B']*2 + pitcher_grpby['3B']*3 + pitcher_grpby['HR']*4) / pitcher_grpby['AB']
            # pitcher_grpby['OPS'] = pitcher_grpby['OBP'] + pitcher_grpby['SLG']

        # 'Team' 컬럼 바로 다음에 계산된 컬럼들 삽입
        new_cols = ['K/IP', 'BB/IP', 'H/IP', 'WHIP', 'ERA', 'BAA', 'OBP'] # , 'OPS', 'OBP', 'SLG']
        for col in new_cols:
            if col in pitcher_grpby.columns:
                team_idx = pitcher_grpby.columns.get_loc('Team') + 1
                pitcher_grpby.insert(team_idx, col, pitcher_grpby.pop(col))

        # 결과 확인
        # rank_by_ascending, rank_by_descending columns  
        rank_by_ascending_cols_p = ['ERA', 'WHIP', 'H/IP', 'BB/IP', 'BAA', 'OBP', 'BF', 'AB', 'P', 'HA', 'HR', 
                                    'SH', 'SF', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'R', 'ER'] # 낮을수록 좋은 지표들
        rank_by_descending_cols_p = ['IP', 'G', 'W', 'L', 'SV', 'HLD', 'SO', 'SO/IP', 'K9'] # 높을수록 좋은 지표들
        st.dataframe(pitcher_grpby.loc[:, rank_by_cols_p_sorted].rename(columns = pitcher_data_EnKr, inplace=False), use_container_width = True, hide_index = True)
        pitcher_grpby_rank = pd.concat([
                                        pitcher_grpby.Team, 
                                        pitcher_grpby[rank_by_descending_cols_p].rank(method = 'min', ascending=False),
                                        pitcher_grpby[rank_by_ascending_cols_p].rank(method = 'min', ascending=True)
                                    ], axis = 1)
        st.write('Ranking')
        pitcher_grpby_rank = pitcher_grpby_rank.loc[:, rank_by_cols_p_sorted]
        st.dataframe(pitcher_grpby_rank.rename(columns = pitcher_data_EnKr, inplace=False), use_container_width = True, hide_index = True)

        ## 히트맵 시각화 팀별 랭킹        
        st.write("Heatmap")
        df = pitcher_grpby_rank.drop('Team', axis = 1).copy()
        # team_englist = ["Big Hits", "FA Members", "RedStorm", "unknown`s", "GNHaJa", "Gideons", "Diamon]ster", "DevilBears", "Rhinos", "Mifas", "BundangStars", "BlueLakers", "SungsiYGSG", "Wasabi", "KometsHSTT"]
        df['team_eng'] = team_englist
        df.set_index('team_eng', inplace=True)
        # 커스텀 컬러맵 생성
        colors = ["#8b0000", "#ffffff"]  # 어두운 빨간색에서 하얀색으로
        cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=15)
        # 히트맵 생성
        plt = create_heatmap(df, cmap, input_figsize = (10, 6))
        st.pyplot(plt)

    

with tab_sn_teamwise:
    # HTML display Setting
    span_stylesetting = '<span style="font-size: 11px; color: black; line-height: 5px;">'
    df_h_meandict = {k: round(v, 3) for k, v in df_hitter[rank_by_cols_h_sorted].mean(numeric_only=True).to_dict().items()}
    df_h_mediandict = {k: round(v, 3) for k, v in df_hitter[rank_by_cols_h_sorted].median(numeric_only=True).to_dict().items()}
    df_p_meandict = {k: round(v, 3) for k, v in df_pitcher[rank_by_cols_p_sorted].dropna().mean(numeric_only=True).to_dict().items()}
    df_p_mediandict = {k: round(v, 3) for k, v in df_pitcher[rank_by_cols_p_sorted].dropna().median(numeric_only=True).to_dict().items()}
    team_name = st.selectbox('팀 선택', (team_id_dict.keys()), key = 'selbox_team_b')
    team_id = team_id_dict[team_name]
    tab_sn_teamwise_1, tab_sn_teamwise_2 = st.tabs(["성남:팀별타자", "성남:팀별투수"])

    with tab_sn_teamwise_1: # club_idx={team_id}&kind=&season={default_year}",
        DATA_URL_B = "http://www.gameone.kr/club/info/ranking/hitter?club_idx={}&kind=&season={}".format(team_id, default_year)
        df_hitter_team = df_hitter.loc[df_hitter.Team == team_name].reset_index(drop=True).drop('Team', axis = 1)
        st.subheader('타자 : {} [{}명]'.format(team_name, df_hitter_team.shape[0]))
        st.dataframe(df_hitter_team[['No', 'Name'] + rank_by_cols_h_sorted[1:]].rename(columns = hitter_data_EnKr, inplace=False), 
                     use_container_width = True, hide_index = True)
        st.write(DATA_URL_B)
        df1 = hitter_grpby.loc[hitter_grpby.Team == team_name, rank_by_cols_h_sorted].drop('Team', axis = 1) # , use_container_width = True, hide_index = True)
        df2 = hitter_grpby_rank.loc[hitter_grpby_rank.Team == team_name].drop('Team', axis = 1)
        df1.insert(0, 'Type', 'Records')
        df2.insert(0, 'Type', 'Rank')
        st.write('Entire Mean(Hitters)')
        st.markdown(span_stylesetting + str(df_h_meandict)[1:-1] +'</span>', unsafe_allow_html=True)
        st.write('Entire Median(Hitters)')
        st.markdown(span_stylesetting + str(df_h_mediandict)[1:-1] +'</span>', unsafe_allow_html=True)

        
        st.dataframe(pd.concat([df1, df2], axis = 0).rename(columns = hitter_data_EnKr, inplace=False), 
                     use_container_width = True, hide_index = True)
    with tab_sn_teamwise_2:
        DATA_URL_P = "http://www.gameone.kr/club/info/ranking/pitcher?club_idx={}&kind=&season={}".format(team_id, default_year)
        df_pitcher_team = df_pitcher.loc[df_pitcher.Team == team_name].reset_index(drop=True).drop('Team', axis = 1)
        st.subheader('투수 : {} [{}명]'.format(team_name, df_pitcher_team.shape[0]))
        st.dataframe(df_pitcher_team[['No', 'Name'] + rank_by_cols_p_sorted[1:]].rename(columns = pitcher_data_EnKr, inplace=False), 
                     use_container_width = True, hide_index = True)
        st.write(DATA_URL_P)
        df1 = pitcher_grpby.loc[pitcher_grpby.Team == team_name, rank_by_cols_p_sorted].drop('Team', axis = 1)
        df2 = pitcher_grpby_rank.loc[pitcher_grpby_rank.Team == team_name].drop('Team', axis = 1)
        df1.insert(0, 'Type', 'Records')
        df2.insert(0, 'Type', 'Rank')
        st.write('Entire Mean(Pitchers)')        
        st.markdown(span_stylesetting + str(df_p_meandict)[1:-1] +'</span>', unsafe_allow_html=True)
        st.write('Entire Median(Pitchers)')
        st.markdown(span_stylesetting + str(df_p_mediandict)[1:-1] +'</span>', unsafe_allow_html=True)        
        st.dataframe(pd.concat([df1, df2], axis = 0).rename(columns = pitcher_data_EnKr, inplace=False), 
                     use_container_width = True, hide_index = True)
with tab_sn_viz:
    tab_sn_viz_1, tab_sn_viz_2, tab_sn_viz_3 = st.tabs(["선수별분포", "팀별비교", "통계량"])
    with tab_sn_viz_1: # 개인 선수별 기록 분포 시각화
        #st.subheader('선수별 기록 분포 시각화')    
        df_plot = df_hitter
        tab_sn_viz_col1, tab_sn_viz_col2, tab_sn_viz_col3 = st.columns(3)
        with tab_sn_viz_col1:        # 데이터셋 선택을 위한 토글 버튼
            dataset_choice = st.radio('데이터셋 선택', ('타자', '투수'), key = 'dataset_choice')
        with tab_sn_viz_col2:         # 그래프 유형 선택을 위한 토글 버튼
            graph_type = st.radio('그래프 유형', ('히스토그램', '박스플롯'), key = 'graph_type')
        with tab_sn_viz_col3:
            colsNo = st.selectbox('한 줄에 몇개 표시할까요? (1~4열):', options=[1, 2, 3, 4], index=2)

        # 선택된 데이터셋에 따라 데이터 프레임 설정
        if dataset_choice == '투수':
            df_plot = df_pitcher.copy()
        else:
            df_plot = df_hitter.copy()

        numeric_columns = df_plot.select_dtypes(include=['float', 'int']).columns
        rows = (len(numeric_columns) + colsNo - 1) // colsNo
        fig, axs = plt.subplots(rows, colsNo, figsize=(15, 3 * rows))

        # axs가 1차원 배열일 경우 처리
        if rows * colsNo == 1:
            axs = [axs]
        elif rows == 1 or colsNo == 1:
            axs = axs.flatten()
        else:
            axs = axs.reshape(-1)

        # "Plotting" 버튼 추가
        if st.button('Plotting', key = 'dist_btn'):
            for i, var in enumerate(numeric_columns):
                ax = axs[i]
                if graph_type == '히스토그램':
                    sns.histplot(df_plot[var].dropna(), kde=False, ax=ax)
                    ax.set_title(f'{var}')
                elif graph_type == '박스플롯':
                    sns.boxplot(x=df_plot[var].dropna(), ax=ax)
                    ax.set_title(f'{var}')

            # 빈 서브플롯 숨기기
            for j in range(len(numeric_columns), rows * colsNo):
                axs[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

                ### template_input 
                # plotly - Plotly의 기본 템플릿.     # plotly_white - 배경이 하얀색인 깔끔한 템플릿.     # plotly_dark - 배경이 어두운색인 템플릿.
                # ggplot2 - R의 ggplot2 스타일을 모방한 템플릿.    # seaborn - Python의 seaborn 라이브러리 스타일을 모방한 템플릿.    # simple_white - 매우 단순하고 깨끗한 템플릿.
    
    with tab_sn_viz_2: # tab_sn_vs [레이더 차트]
        teams = list(sorted(team_id_dict.keys())) # Team list applied sorting
        template_input = 'plotly_white'    
        try:
            # '호시탐탐'의 인덱스 찾기
            idx_hstt = teams.index('코메츠 호시탐탐')
        except ValueError:
            idx_hstt = 0

        # st.subheader('팀 간 전력 비교')      
        tab_sn_vs_col1, tab_sn_vs_col2, tab_sn_vs_col3 = st.columns(3)
        with tab_sn_vs_col1:        # 2개 팀을 비교할지 / 전체 팀을 한판에 그릴지 선택하는 토글 버튼
            team_all = st.toggle("Select All Teams")
        with tab_sn_vs_col2:         # # 스트림릿 셀렉트박스로 팀 선택
            if not team_all: #team_selection_rader == 'VS':            # 스트림릿 셀렉트박스로 팀 선택
                team1 = st.selectbox('Select Team 1:', options = teams, index=idx_hstt)
        with tab_sn_vs_col3:  
            if not team_all: #if team_selection_rader == 'VS':            # 스트림릿 셀렉트박스로 팀 선택              
                team2 = st.selectbox('Select Team 2:', options = teams, index=1)
        multisel_h = st.multiselect('공격(타자) 지표 선택',
            [hitter_data_EnKr.get(col, col) for col in rank_by_cols_h_sorted[1:]], 
            ['타율', '출루율', 'OPS', '볼넷', '삼진', '도루'], max_selections = 12
        )
        multisel_p = st.multiselect('수비(투수) 지표 선택',
            # rank_by_cols_p_sorted, 
            [pitcher_data_EnKr.get(col, col) for col in rank_by_cols_p_sorted[1:]],
            ['방어율', 'WHIP', 'H/IP', 'BB/IP', 'SO/IP', '피안타율'], max_selections = 12
        )        
        # "Plotting" 버튼 추가
        if st.button('Plotting', key = 'vs_rader_btn'):
            hitter_grpby_plt = hitter_grpby.rename(columns = hitter_data_EnKr, inplace=False).copy()
            pitcher_grpby_plt = pitcher_grpby.rename(columns = pitcher_data_EnKr, inplace=False) .copy()
            selected_cols_h = ['팀'] + multisel_h # ['AVG', 'OBP', 'OPS', 'BB', 'SO', 'SB']
            selected_cols_p = ['팀'] + multisel_p
            # 데이터 스케일링
            hitter_grpby_plt_scaled = hitter_grpby_plt.rename(columns = hitter_data_EnKr, inplace=False).copy()
            scaler_h = MinMaxScaler()             # 스케일러 초기화
            hitter_grpby_plt_scaled[hitter_grpby_plt_scaled.columns[1:]] = scaler_h.fit_transform(hitter_grpby_plt_scaled.iloc[:, 1:]) # 첫 번째 열 'Team'을 제외하고 스케일링
            pitcher_grpby_plt_scaled = pitcher_grpby_plt.rename(columns = pitcher_data_EnKr, inplace=False).copy()
            scaler_p = MinMaxScaler()             # 스케일러 초기화
            pitcher_grpby_plt_scaled[pitcher_grpby_plt_scaled.columns[1:]] = scaler_p.fit_transform(pitcher_grpby_plt_scaled.iloc[:, 1:]) # 첫 번째 열 'Team'을 제외하고 스케일링
            if team_all: #if team_selection_rader == '전체':
                filtered_data_h = hitter_grpby_plt_scaled
                radar_data_h = filtered_data_h[selected_cols_h].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                fig_h = px.line_polar(radar_data_h, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'공격력')   

                filtered_data_p = pitcher_grpby_plt_scaled
                radar_data_p = filtered_data_p[selected_cols_p].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                fig_p = px.line_polar(radar_data_p, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'수비력')  

            else: # team_selection_rader == 'VS' : 2개팀을 비교할 경우
                # 선택된 팀 데이터 필터링
                filtered_data_h = hitter_grpby_plt_scaled[hitter_grpby_plt_scaled['팀'].isin([team1, team2])]#.rename(columns = hitter_data_EnKr, inplace=False).copy()
                # 레이더 차트 데이터 준비
                radar_data_h = filtered_data_h[selected_cols_h].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                # 레이더 차트 생성
                fig_h = px.line_polar(radar_data_h, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'공격력 : {team1} vs {team2}')
                # 선택된 팀 데이터 필터링
                filtered_data_p = pitcher_grpby_plt_scaled[pitcher_grpby_plt_scaled['팀'].isin([team1, team2])]#.rename(columns = pitcher_data_EnKr, inplace=False).copy()
                # 레이더 차트 데이터 준비
                radar_data_p = filtered_data_p[selected_cols_p].melt(id_vars=['팀'], var_name='Stat', value_name='Value')
                # 레이더 차트 생성
                fig_p = px.line_polar(radar_data_p, r='Value', theta='Stat', color='팀', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'수비력 : {team1} vs {team2}')
            ########################
            ## Chart AND Dataframe display Area
            if not team_all:    #if team_selection_rader == 'VS':  
                df_rader_vs_h = pd.concat([hitter_grpby_plt.loc[hitter_grpby_plt['팀'] == team1, selected_cols_h], 
                                    hitter_grpby_plt.loc[hitter_grpby_plt['팀'] == team2, selected_cols_h]], axis = 0).sort_values('팀')      
                st.dataframe(df_rader_vs_h, use_container_width = True, hide_index = True) 
            else :
                st.dataframe(hitter_grpby_plt[selected_cols_h].sort_values('팀').T, use_container_width = True)    

            if not team_all:    #if team_selection_rader == 'VS':    
                df_rader_vs_p = pd.concat([pitcher_grpby_plt.loc[pitcher_grpby_plt['팀'] == team1, selected_cols_p], 
                                    pitcher_grpby_plt.loc[pitcher_grpby_plt['팀'] == team2, selected_cols_p]], axis = 0).sort_values('팀')           
                st.dataframe(df_rader_vs_p, use_container_width = True, hide_index = True)      
            else :
                st.dataframe(pitcher_grpby_plt[selected_cols_p].sort_values('팀').T, use_container_width = True)  

            tab_sn_vs_col2_1, tab_sn_vs_col2_2 = st.columns(2)   
            with tab_sn_vs_col2_1:            # 차트 보기 [Hitter]
                st.plotly_chart(fig_h, use_container_width=True)
            with tab_sn_vs_col2_2:             # 차트 보기 [Pitcher]
                st.plotly_chart(fig_p, use_container_width=True)
    with tab_sn_viz_3:
        st.write("선수 별 기록 분포 통계량")
        st.write("타자")
        st.dataframe(df_hitter.drop('No', axis = 1).rename(columns = hitter_data_EnKr, inplace=False).describe(), 
                     use_container_width = True, hide_index = False)  
        st.write("투수")
        st.dataframe(df_pitcher.drop('No', axis = 1).rename(columns = pitcher_data_EnKr, inplace=False).describe(), 
                     use_container_width = True, hide_index = False)  

with tab_schd:
    # 일정표 URL 설정
    url = f"http://www.gameone.kr/club/info/schedule/table?club_idx=7984&kind=&season={default_year}"
    st.write(url)
    # HTTP GET 요청
    response = requests.get(url)
    response.raise_for_status()  # 요청이 성공했는지 확인

    # BeautifulSoup을 이용하여 HTML 파싱
    soup = BeautifulSoup(response.content, 'html.parser')

    # 테이블 찾기
    table = soup.find('table', {'class': 'game_table'})  # 테이블의 클래스를 확인하고 지정하세요

    # 테이블 헤더 추출
    headers = [header.text.strip() for header in table.find_all('th')]

    # 테이블 데이터 추출
    rows = []
    for row in table.find_all('tr')[1:]:  # 첫 번째 행은 헤더이므로 제외
        cells = row.find_all('td')
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)

    # pandas DataFrame 생성
    df_schd = pd.DataFrame(rows, columns=headers)
    df_schd = df_schd.sort_values('일시').reset_index(drop=True)
    data = df_schd.게임.str.split('\n').tolist()
    # 최대 열 개수 확인
    max_columns = max(len(row) for row in data)
    # 열 이름 설정
    column_names = [f"col{i+1}" for i in range(max_columns)]
    st.write(pd.DataFrame(data))
    # DataFrame 생성
    df_team = pd.DataFrame(data, columns=column_names).drop(['col3', 'col4', 'col5'], axis =1)
    # DataFrame 출력
    df_schd2 = pd.concat([df_schd.drop(['게임', '분류'], axis =1), df_team], axis = 1)
    # 열 갯수가 6개일 경우, '6' 컬럼을 추가
    if df_schd2.shape[1] == 6:
        df_schd2['6'] = ''  # '' 값을 가진 빈 컬럼을 추가    
    df_schd2.columns = ['일시', '구장', '결과', '선공', '선공점수', '후공', '후공점수']
    df_schd2.구장 = df_schd2.구장.str.replace('야구장', '')
    st.write(df_schd2)
    first_called = df_schd2.선공점수.str.contains('콜드승')
    second_called = df_schd2.후공점수.str.contains('콜드승')
    df_schd2.선공점수 = df_schd2.선공점수.str.replace('콜드승 ', '').str.replace('기권승 ', '').str.replace('몰수승 ', '').replace(r'^\s*$', pd.NA, regex=True).fillna(0).astype('int')  #.replace('', 0).astype('int')
    df_schd2.후공점수 = df_schd2.후공점수.str.replace('콜드승 ', '').str.replace('기권승 ', '').str.replace('몰수승 ', '').fillna(0).astype('int')
    df_schd2['Result'] = ''
    tmp_result = list()
    for i in range(df_schd2.shape[0]):
        # print(i, first_called[i], second_called[i])
        if df_schd2.iloc[i]['선공점수'] > df_schd2.iloc[i]['후공점수']:
            if first_called[i]:
                result = df_schd2.iloc[i]['선공'] + '_콜드승'    
            else :
                result = df_schd2.iloc[i]['선공'] + '_승'
        elif df_schd2.iloc[i]['선공점수'] < df_schd2.iloc[i]['후공점수']:
            if second_called[i]:
                result = df_schd2.iloc[i]['후공'] + '_콜드승'
            else:
                result = df_schd2.iloc[i]['후공'] + '_승'
            # print(i, result)
        elif (df_schd2.iloc[i]['결과'] != '게임대기') & (df_schd2.iloc[i]['선공점수'] == df_schd2.iloc[i]['후공점수']):
            result = '무'
            # print(i, result)
        else:
            result = '경기전'
            # print(i, result)
        tmp_result.append(result)

    df_schd2['Result'] = tmp_result
    df_schd2.loc[df_schd2['Result'].str.contains('호시탐탐_콜드승'), 'Result'] = '콜드승'
    df_schd2.loc[df_schd2['Result'].str.contains('호시탐탐_승'), 'Result'] = '승'
    df_schd2.loc[df_schd2['Result'].str.contains('_승'), 'Result'] = '패'
    df_schd2.loc[df_schd2['Result'].str.contains('_콜드승'), 'Result'] = '콜드패'

    df_schd2 = df_schd2.drop('결과', axis = 1)
    df_schd2.columns = ['일시', '구장', '선공', '선', '후공', '후', '결과']
    st.markdown(soup.find('span', {'class': 'info'}), unsafe_allow_html=True)
    # st.dataframe(df_schd2)
    st.table(df_schd2.reset_index(drop=True))

with tab_dataload:
    user_password_update = st.text_input('Input Password for Update', type='password')
    user_password_update = str(user_password_update)
    if user_password_update == st.secrets["password_update"]: # Correct Password
        st.write('Correct Password')
        st.write('아래 버튼을 누르면 현재 시점의 데이터를 새로 로드합니다.')        
        if st.button('Data Update'):
            hitters = []
            pitchers = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(load_data, team_name, team_id): team_name for team_name, team_id in team_id_dict.items()}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        hitters.append(result['hitter'])
                        pitchers.append(result['pitcher'])
                    except Exception as exc:
                        print(f'Team {futures[future]} generated an exception: {exc}')
            # 모든 데이터를 각각의 데이터프레임으로 합침
            final_hitters_data = pd.concat(hitters, ignore_index=True)
            final_pitchers_data = pd.concat(pitchers, ignore_index=True)

            # 데이터프레임 df의 컬럼 자료형 설정
            df_hitter = final_hitters_data.astype(hitter_data_types)
            # 타자 데이터프레임 컬럼명 영어로
            df_hitter.columns = ['Name', 'No', 'AVG', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 
                                'SB', 'CS', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'DP', 'SLG', 'OBP', 'SB%', 'MHit', 
                                'OPS', 'BB/K', 'XBH/H', 'Team']

            final_pitchers_data.loc[final_pitchers_data.방어율 == '-', '방어율'] = np.nan

            # 투수 데이터프레임 df_pitcher의 컬럼 자료형 설정
            df_pitcher = final_pitchers_data.astype(pitcher_data_types)
            # 투수 데이터프레임 컬럼명 영어로
            df_pitcher.columns = ['Name', 'No', 'ERA', 'G', 'W', 'L', 'SV', 'HLD', 'WPCT', 
                                'BF', 'AB', 'P', 'IP', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'WP', 'BK', 
                                'R', 'ER', 'WHIP', 'BAA', 'K9', 'Team']
            # IP 컬럼을 올바른 소수 형태로 변환
            df_pitcher['IP'] = df_pitcher['IP'].apply(lambda x: int(x) + (x % 1) * 10 / 3).round(2)
            
            ###### GOOGLE SHEETS
            # Create GSheets connection
            conn = st.connection("gsheets", type=GSheetsConnection)

            df_hitter = conn.update(worksheet="df_hitter", data=df_hitter)
            df_pitcher = conn.update(worksheet="df_pitcher", data=df_pitcher)
            time.sleep(3)
            st.toast('Saved Data from Web to Cloud! (Updated)', icon='☁️')
            st.write(df_hitter.shape, "Hitter Data SAVED!")
            st.dataframe(df_hitter, use_container_width = True, hide_index = True)
            st.write(df_pitcher.shape, "Pitcher Data SAVED!")
            st.dataframe(df_pitcher, use_container_width = True, hide_index = True)
    else:
        st.write('Wrong Password!!')


        
with tab_sn_terms:
    st.subheader('야구 기록 설명')
    tab_sn_terms_col1, tab_sn_terms_col2 = st.columns(2)
    # 스트림릿 페이지 제목 설정
    with tab_sn_terms_col1:
        # 타자 데이터 설명
        st.markdown("""
        ### 타자(Hitters) 컬럼명 약어:
        | ENG | KOR | Desc                    |
        |--------------|-------------|--------------------------------|
        | Name         | 성명        | Player's name                  |
        | No           | 배번        | Jersey number                  |
        | AVG           | 타율        | Batting average                |
        | G            | 경기        | Games played                   |
        | PA           | 타석        | Plate appearances              |
        | AB           | 타수        | At bats                        |
        | R            | 득점        | Runs                           |
        | H            | 총안타      | Hits                           |
        | 1B           | 1루타       | Singles                        |
        | 2B           | 2루타       | Doubles                        |
        | 3B           | 3루타       | Triples                        |
        | HR           | 홈런        | Home runs                      |
        | TB           | 루타        | Total bases                    |
        | RBI          | 타점        | Runs batted in                 |
        | SB           | 도루        | Stolen bases                   |
        | CS           | 도실(도루자)| Caught stealing                |
        | SH           | 희타        | Sacrifice hits                 |
        | SF           | 희비        | Sacrifice flies                |
        | BB           | 볼넷        | Walks                          |
        | IBB          | 고의4구     | Intentional walks              |
        | HBP          | 사구        | Hit by pitch                   |
        | SO           | 삼진        | Strikeouts                     |
        | DP           | 병살        | Double plays                   |
        | SLG          | 장타율      | Slugging percentage            |
        | OBP          | 출루율      | On-base percentage             |
        | SB%          | 도루성공률  | Stolen base percentage         |
        | MHit         | 멀티히트    | Multi-hit games                |
        | OPS          | OPS         | On-base plus slugging          |
        | BB/K         | BB/K       | Walks per strikeout            |
        | XBH/H        | 장타/안타   | Extra base hits per hit        |
        | Team         | 팀          | Team name                      |
        """)
    with tab_sn_terms_col2:
        # 투수 데이터 설명
        st.markdown("""
        ### 투수(Pitchers) 컬럼명 약어:
        | ENG | KOR | Desc                    |
        |--------------|-------------|--------------------------------|
        | Name         | 성명        | Player's name                  |
        | No           | 배번        | Jersey number                  |
        | ERA          | 방어율      | Earned run average             |
        | WHIP         | WHIP        | Walks plus hits per inning    |
        | SO/IP        | 이닝 당 탈삼진 | Strikeouts per 1 Inning       |
        | GS           | 경기수      | Games started                  |
        | W            | 승          | Wins                           |
        | L            | 패          | Losses                         |
        | SV           | 세          | Saves                          |
        | HLD          | 홀드        | Holds                          |
        | BF           | 타자        | Batters faced                  |
        | AB           | 타수        | At bats against                |
        | P            | 투구수      | Pitches thrown                 |
        | HA           | 피안타      | Hits allowed                   |
        | HR           | 피홈런      | Home runs allowed              |
        | SH           | 희생타        | Sacrifice hits allowed         |
        | SF           | 희생플라이     | Sacrifice flies allowed        |
        | BB           | 볼넷        | Walks allowed                  |
        | IBB          | 고의4구     | Intentional walks allowed      |
        | HBP          | 사구        | Hit by pitch allowed           |
        | SO           | 탈삼진      | Strikeouts                     |
        | WP           | 폭투        | Wild pitches                   |
        | BK           | 보크        | Balks                          |
        | R            | 실점        | Runs allowed                   |
        | ER           | 자책점      | Earned runs allowed            |
        | IP           | 이닝        | Innings pitched                |    
        | SO/IP        | 이닝 당 탈삼진 | Strikeouts per 1 Inning       |
        """)
