import lxml
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Baseball Data",)
st.title('Saturday League Data')

# DATE_COLUMN = 'date/time'
team_id_dict = {
    "코메츠 호시탐탐": 7984, "Big Hits": 36636,    "FA Members": 13621,
    "RedStorm": 17375,    "unknown`s": 33848, "그냥하자": 10318,
    "기드온스": 27811,    "다이아몬스터": 39783,     "데빌베어스": 19135,
    "라이노즈": 41236,    "미파스": 19757,    "분당스타즈": 34402,
    "블루레이커즈": 22924,    "성시야구선교단": 29105,    "와사비": 14207,
}

# st.write('You selected TEAM:', option)

ALB_URL_SCHD = "http://alb.or.kr/s/schedule/schedule_team_2019.php?id=schedule_team&sc=2&team=%B7%B9%BE%CB%B7%E7%C5%B0%C1%EE&gyear=2024"

def load_data(team_name, team_id):
    urls = {
        'hitter': f"http://www.gameone.kr/club/info/ranking/hitter?club_idx={team_id}",
        'pitcher': f"http://www.gameone.kr/club/info/ranking/pitcher?club_idx={team_id}"
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

# 병렬로 데이터 로딩
hitters = []
pitchers = []
with ThreadPoolExecutor(max_workers=10) as executor:
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

# 데이터프레임 df에 적용할 자료형 매핑
hitter_data_types = {
    '성명': 'str', '배번': 'str', '타율': 'float', '경기': 'int', '타석': 'int', '타수': 'int',
    '득점': 'int', '총안타': 'int', '1루타': 'int', '2루타': 'int', '3루타': 'int', '홈런': 'int',
    '루타': 'int', '타점': 'int', '도루': 'int', '도실(도루자)': 'int', '희타': 'int', '희비': 'int',
    '볼넷': 'int', '고의4구': 'int', '사구': 'int', '삼진': 'int', '병살': 'int', '장타율': 'float',
    '출루율': 'float', '도루성공률': 'float', '멀티히트': 'int', 'OPS': 'float', 'BB/K': 'float',
    '장타/안타': 'float', '팀': 'str'
}
# 데이터프레임 df의 컬럼 자료형 설정
df_hitter = final_hitters_data.astype(hitter_data_types)
# 타자 데이터프레임 컬럼명 영어로
df_hitter.columns = ['Name', 'No', 'BA', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'SH', 'SF', 
                     'BB', 'IBB', 'HBP', 'SO', 'DP', 'SLG', 'OBP',  'SB%', 'MHit', 'OPS', 'BB/K', 'XBH/H', 'Team']

# 투수 데이터프레임 df_pitcher에 적용할 자료형 매핑
pitcher_data_types = {
    '성명': 'str', '배번': 'str', '방어율': 'float', '경기수': 'int', '승': 'int', '패': 'int', '세': 'int',
    '홀드': 'int', '승률': 'float', '타자': 'int', '타수': 'int', '투구수': 'int', '이닝': 'float',
    '피안타': 'int', '피홈런': 'int', '희타': 'int', '희비': 'int', '볼넷': 'int', '고의4구': 'int',
    '사구': 'int', '탈삼진': 'int', '폭투': 'int', '보크': 'int', '실점': 'int', '자책점': 'int',
    'WHIP': 'float', '피안타율': 'float', '탈삼진율': 'float', '팀': 'str'
}

final_pitchers_data.loc[final_pitchers_data.방어율 == '-', '방어율'] = np.nan
# 투수 데이터프레임 df_pitcher의 컬럼 자료형 설정
df_pitcher = final_pitchers_data.astype(pitcher_data_types)
# 투수 데이터프레임 컬럼명 영어로
df_pitcher.columns = ['Name', 'No', 'ERA', 'GS', 'W', 'L', 'SV', 'HLD', 'WPCT', 'BF', 'AB', 'P', 'IP', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'WP', 'BK', 
                      'R', 'ER', 'WHIP', 'BAA', 'K9', 'Team']
# IP 컬럼을 올바른 소수 형태로 변환
df_pitcher['IP'] = df_pitcher['IP'].apply(lambda x: int(x) + (x % 1) * 10 / 3).round(2)

## 탭 설정 ]]]]]]]]]
tab_sn_players, tab_sn_teamwise, tab_sn_viz, tab_sn_terms, tab_sn_void = st.tabs(["성남:전체선수", "성남:팀별선수", "성남:시각화", "용어", "void"])

with tab_sn_players:
    tab_sn_players_1, tab_sn_players_2 = st.tabs(["성남:전체타자", "성남:전체투수"])
    with tab_sn_players_1:
        st.subheader('성남 : 전체타자 [{}명]'.format(df_hitter.shape[0]))
        st.dataframe(df_hitter)
        st.subheader('팀별 기록')
        hitter_sumcols = ['PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'SO', 'DP', 'MHit']
        hitter_grpby = df_hitter[hitter_sumcols + ['Team']].groupby('Team').sum().reset_index()

        # 타율(BA), 출루율(OBP), 장타율(SLG), OPS 계산 & 반올림
        hitter_grpby['BA'] = (hitter_grpby['H'] / hitter_grpby['AB']).round(3)
        hitter_grpby['OBP'] = ((hitter_grpby['H'] + hitter_grpby['BB'] + hitter_grpby['HBP']) / (hitter_grpby['AB'] + hitter_grpby['BB'] + hitter_grpby['HBP'] + hitter_grpby['SF'])).round(3)
        hitter_grpby['SLG'] = (hitter_grpby['TB'] / hitter_grpby['AB']).round(3)
        hitter_grpby['OPS'] = (hitter_grpby['OBP'] + hitter_grpby['SLG']).round(3)
        
        # 'Team' 컬럼 바로 다음에 계산된 컬럼들 삽입
        for col in ['OPS', 'SLG', 'OBP', 'BA']:
            team_idx = hitter_grpby.columns.get_loc('Team') + 1
            hitter_grpby.insert(team_idx, col, hitter_grpby.pop(col))
            
        st.dataframe(hitter_grpby)
        hitter_grpby_rank = (pd.concat([hitter_grpby.Team, hitter_grpby.rank(method = 'min', ascending=False).drop('Team', axis= 1)], axis = 1))
        st.write('Ranking')
        st.dataframe(hitter_grpby_rank)        

    with tab_sn_players_2:
        st.subheader('성남 : 전체투수 [{}명]'.format(df_pitcher.shape[0]))
        st.dataframe(df_pitcher)
        st.subheader('팀별 기록')
        # 팀별로 그룹화하고 정수형 변수들의 합계 계산
        pitcher_sumcols = df_pitcher.select_dtypes(include=['int64']).columns.tolist() + ['IP'] # Sum 컬럼 선택
        pitcher_grpby = df_pitcher.groupby('Team')[pitcher_sumcols].sum().reset_index()  # 팀별 합계
        # 파생 변수 추가
        # 방어율(ERA) 계산: (자책점 / 이닝) * 9 (예제로 자책점과 이닝 컬럼 필요)
        if 'ER' in df_pitcher.columns and 'IP' in df_pitcher.columns:
            pitcher_grpby['ERA'] = ((pitcher_grpby['ER'] / pitcher_grpby['IP']) * 9).round(3)
        
        # 이닝당 삼진/볼넷/피안타 계산 (예제로 삼진(K), 볼넷(BB), 피안타(HA) 컬럼 필요)
        if 'SO' in df_pitcher.columns and 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
            pitcher_grpby['SO/IP'] = (pitcher_grpby['SO'] / pitcher_grpby['IP']).round(2)
            pitcher_grpby['BB/IP'] = (pitcher_grpby['BB'] / pitcher_grpby['IP']).round(2)
            pitcher_grpby['H/IP'] = (pitcher_grpby['HA'] / pitcher_grpby['IP']).round(2)
        
        # WHIP 계산: (볼넷 + 피안타) / 이닝
        if 'BB' in df_pitcher.columns and 'HA' in df_pitcher.columns:
            pitcher_grpby['WHIP'] = ((pitcher_grpby['BB'] + pitcher_grpby['HA']) / pitcher_grpby['IP']).round(3)
        
        # 'Team' 컬럼 바로 다음에 계산된 컬럼들 삽입
        new_cols = ['K/IP', 'BB/IP', 'H/IP', 'WHIP', 'ERA']
        for col in new_cols:
            if col in pitcher_grpby.columns:
                team_idx = pitcher_grpby.columns.get_loc('Team') + 1
                pitcher_grpby.insert(team_idx, col, pitcher_grpby.pop(col))

        # 결과 확인
        # rank_by_ascending, rank_by_descending columns 
        rank_by_ascending_cols_p = ['ERA', 'WHIP', 'H/IP', 'BB/IP', 'BF', 'AB', 'P', 'HA', 'HR', 
                                    'SH', 'SF', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'R', 'ER'] # 낮을수록 좋은 지표들
        rank_by_descending_cols_p = ['IP', 'GS', 'W', 'L', 'SV', 'HLD', 'SO', 'SO/IP'] # 높을수록 좋은 지표들
        # 출력시 열 순서 변경
        rank_by_cols_p_sorted = ['Team', 'ERA', 'WHIP', 'H/IP', 'BB/IP', 'SO/IP', 'IP', 'GS', 'W', 'L', 'SV', 'HLD', 'SO', 'BF', 'AB', 'P', 'HA', 'HR', 'SH', 'SF', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'R', 'ER'] 
        st.write(pitcher_grpby.loc[:, rank_by_cols_p_sorted])
        pitcher_grpby_rank = pd.concat([
                                        pitcher_grpby.Team, 
                                        pitcher_grpby[rank_by_descending_cols_p].rank(method = 'min', ascending=False),
                                        pitcher_grpby[rank_by_ascending_cols_p].rank(method = 'min', ascending=True)
                                    ], axis = 1)
        st.write('Ranking')
        st.dataframe(pitcher_grpby_rank.loc[:, rank_by_cols_p_sorted])

with tab_sn_teamwise:
    tab_sn_teamwise_1, tab_sn_teamwise_2 = st.tabs(["성남:팀별타자", "성남:팀별투수"])
    with tab_sn_teamwise_1:
        team_name_B = st.selectbox('팀 선택', (team_id_dict.keys()), key = 'selbox_team_b')
        # 팀명을 기준으로 데이터 프레임 필터링
        team_id = team_id_dict[team_name_B]
        DATA_URL_B = "http://www.gameone.kr/club/info/ranking/hitter?club_idx={}".format(team_id)
        
        df_hitter_team = df_hitter.loc[df_hitter.Team == team_name_B].reset_index(drop=True).drop('Team', axis = 1)
        st.subheader('타자 : {} [{}명]'.format(team_name_B, df_hitter_team.shape[0]))
        st.dataframe(df_hitter_team)
        st.write(DATA_URL_B)
        st.write(hitter_grpby.loc[hitter_grpby.Team == team_name_B])
        st.write(hitter_grpby_rank.loc[hitter_grpby_rank.Team == team_name_B])
        
    with tab_sn_teamwise_2:
        team_name_P = st.selectbox('팀 선택', (team_id_dict.keys()), key = 'selbox_team_p')   
        # 팀명을 기준으로 데이터 프레임 필터링
        team_id = team_id_dict[team_name_P]
        DATA_URL_P = "http://www.gameone.kr/club/info/ranking/pitcher?club_idx={}".format(team_id)

        df_pitcher_team = df_pitcher.loc[df_pitcher.Team == team_name_P].reset_index(drop=True).drop('Team', axis = 1)
        st.subheader('투수 : {} [{}명]'.format(team_name_P, df_pitcher_team.shape[0]))
        st.dataframe(df_pitcher_team) 
        st.write(DATA_URL_P) 
        st.write(pitcher_grpby.loc[pitcher_grpby.Team == team_name_P])
        st.write(pitcher_grpby_rank.loc[pitcher_grpby_rank.Team == team_name_P])        

with tab_sn_viz:
    tab_sn_viz_1, tab_sn_viz_2 = st.tabs(["선수별기록분포", "팀별비교"])
    with tab_sn_viz_1: # 개인 선수별 기록 분포 시각화
        st.subheader('선수별 기록 분포 시각화')    
        df_plot = df_hitter
        tab_sn_viz_col1, tab_sn_viz_col2, tab_sn_viz_col3 = st.columns(3)
        with tab_sn_viz_col1:        # 데이터셋 선택을 위한 토글 버튼
            dataset_choice = st.radio('데이터셋 선택', ('타자', '투수'), key = 'dataset_choice')
        with tab_sn_viz_col2:         # 그래프 유형 선택을 위한 토글 버튼
            graph_type = st.radio('그래프 유형', ('히스토그램', '박스플롯'), key = 'graph_type')
        with tab_sn_viz_col3:
            colsNo = st.selectbox('한 줄에 몇개의 그래프를 표시할까요? (1~4열):', options=[1, 2, 3, 4], index=2)

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
    # plotly - Plotly의 기본 템플릿.
    # plotly_white - 배경이 하얀색인 깔끔한 템플릿.
    # plotly_dark - 배경이 어두운색인 템플릿.
    # ggplot2 - R의 ggplot2 스타일을 모방한 템플릿.
    # seaborn - Python의 seaborn 라이브러리 스타일을 모방한 템플릿.
    # simple_white - 매우 단순하고 깨끗한 템플릿.
    # none - 최소한의 스타일로, 사용자가 자신만의 스타일을 쉽게 추가할 수 있게 해줍니다

    # [0:"Team"1:"BA"2:"OBP"3:"SLG"4:"OPS"5:"PA"6:"AB"7:"R"8:"H"9:"1B"10:"2B"11:"3B"12:"HR"13:"TB"
    # 14:"RBI"15:"SB"16:"CS"17:"SH"18:"SF"19:"BB"20:"IBB"21:"HBP"22:"SO"23:"DP"24:"MHit"]

    # [0:"Team"1:"ERA"2:"WHIP"3:"H/IP"4:"BB/IP"5:"GS"6:"W"7:"L"8:"SV"9:"HLD"10:"BF"11:"AB"12:"P"13:"HA"
    # 14:"HR"15:"SH"16:"SF"17:"BB"18:"IBB"19:"HBP"20:"SO"21:"WP"22:"BK"23:"R"24:"ER"25:"IP"26:"SO/IP"]
    with tab_sn_viz_2: # tab_sn_vs
        template_input = 'plotly_white'    
        st.subheader('팀 간 전력 비교')      
        tab_sn_vs_col1, tab_sn_vs_col2, tab_sn_vs_col3 = st.columns(3)
        with tab_sn_vs_col1:        # 데이터셋 선택을 위한 토글 버튼
            team_selection_rader = st.radio('팀 선택', ('전체', 'VS'), key = 'team_selection_rader')        
        teams = hitter_grpby['Team'].unique()
        with tab_sn_vs_col2:         # # 스트림릿 셀렉트박스로 팀 선택
            if team_selection_rader == 'VS':            # 스트림릿 셀렉트박스로 팀 선택
                team1 = st.selectbox('Select Team 1:', options = teams, index=14)
        with tab_sn_vs_col3:  
            if team_selection_rader == 'VS':            # 스트림릿 셀렉트박스로 팀 선택              
                team2 = st.selectbox('Select Team 2:', options = teams, index=12)

        # "Plotting" 버튼 추가
        if st.button('Plotting', key = 'vs_rader_btn'):
            # 선택된 데이터셋에 따라 데이터 프레임 설정
            selected_cols_h = ['Team', 'BA', 'OBP', 'OPS', 'BB', 'SO', 'SB']
            selected_cols_p = ['Team', 'ERA', 'WHIP', 'H/IP', 'BB/IP', 'SO/IP']        

            if team_selection_rader == '전체':
                filtered_data_h = hitter_grpby.copy()
                radar_data_h = filtered_data_h[selected_cols_h].melt(id_vars=['Team'], var_name='Stat', value_name='Value')
                fig_h = px.line_polar(radar_data_h, r='Value', theta='Stat', color='Team', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'공격력')   

                filtered_data_p = pitcher_grpby.copy()
                radar_data_p = filtered_data_p[selected_cols_p].melt(id_vars=['Team'], var_name='Stat', value_name='Value')
                fig_p = px.line_polar(radar_data_p, r='Value', theta='Stat', color='Team', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'수비력')  

            else: # team_selection_rader == 'VS' : 2개팀을 비교할 경우
                # 선택된 팀 데이터 필터링
                filtered_data_h = hitter_grpby[hitter_grpby['Team'].isin([team1, team2])].copy()
                # st.write(filtered_data_h)
                # 레이더 차트 데이터 준비
                radar_data_h = filtered_data_h[selected_cols_h].melt(id_vars=['Team'], var_name='Stat', value_name='Value')
                # 레이더 차트 생성
                fig_h = px.line_polar(radar_data_h, r='Value', theta='Stat', color='Team', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'공격력 : {team1} vs {team2}')
                # 선택된 팀 데이터 필터링
                filtered_data_p = pitcher_grpby[pitcher_grpby['Team'].isin([team1, team2])].copy()
                # st.write(filtered_data_p)
                # 레이더 차트 데이터 준비
                radar_data_p = filtered_data_p[selected_cols_p].melt(id_vars=['Team'], var_name='Stat', value_name='Value')
                # 레이더 차트 생성
                fig_p = px.line_polar(radar_data_p, r='Value', theta='Stat', color='Team', line_close=True,
                                    color_discrete_sequence=px.colors.qualitative.D3, #px.colors.sequential.Plasma_r,
                                    template=template_input, title=f'수비력 : {team1} vs {team2}')
            tab_sn_vs_col2_1, tab_sn_vs_col2_2 = st.columns(2)   
            with tab_sn_vs_col2_1:            # 차트 보기 [Hitter]
                if team_selection_rader == 'VS':        
                    st.dataframe(pd.concat([filtered_data_h.loc[filtered_data_h.Team == team1, selected_cols_h], 
                                        filtered_data_h.loc[filtered_data_h.Team == team2, selected_cols_h]], axis = 0))        
                else :
                    st.dataframe(filtered_data_h[selected_cols_h])
                st.plotly_chart(fig_h, use_container_width=True)
            with tab_sn_vs_col2_2:             # 차트 보기 [Pitcher]
                if team_selection_rader == 'VS':                
                    st.dataframe(pd.concat([filtered_data_p.loc[filtered_data_p.Team == team1, selected_cols_p], 
                                        filtered_data_p.loc[filtered_data_p.Team == team2, selected_cols_p]], axis = 0))     
                else :
                    st.dataframe(filtered_data_p[selected_cols_p])
                st.plotly_chart(fig_p, use_container_width=True)

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
        | BA           | 타율        | Batting average                |
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


with tab_sn_void:
    st.subheader('빈 칸')    
