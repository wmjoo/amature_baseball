import lxml
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
        page_title="Baseball Data",
)
st.title('성남(토요)리그 DATA')

# DATE_COLUMN = 'date/time'
team_id_dict = {
    "코메츠 호시탐탐": 7984, "Big Hits": 36636,    "FA Members": 13621,
    "RedStorm": 17375,    "unknown`s": 33848, "그냥하자": 10318,
    "기드온스": 27811,    "다이아몬스터": 39783,     "데빌베어스(Devil Bears)": 19135,
    "라이노즈": 41236,    "미파스": 19757,    "분당스타즈": 34402,
    "블루레이커즈": 22924,    "성시야구선교단": 29105,    "와사비": 14207,
}

team_name = st.selectbox(
    '팀 선택',
    (team_id_dict.keys()))

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

# 팀명을 기준으로 데이터 프레임 필터링
team_id = team_id_dict[team_name]
DATA_URL_B = "http://www.gameone.kr/club/info/ranking/hitter?club_idx={}".format(team_id)
DATA_URL_P = "http://www.gameone.kr/club/info/ranking/pitcher?club_idx={}".format(team_id)

# 최종 데이터 프레임 출력
#print("Hitters Data:")
#display(df_hitter)
#print("Pitchers Data:")
#display(df_pitcher)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["성남:팀별타자", "성남:팀별투수",
                                                                            "성남:전체타자", "성남:전체투수", "투수:규정이상", "투수:규정미달", "안양_일정"])

with tab1:
    st.subheader('타자 : {}'.format(team_name))
    st.dataframe(df_hitter.loc[df_hitter.팀 == team_name].reset_index(drop=True).drop('팀', axis = 1)) #)
    st.write(DATA_URL_B)

with tab2:
    st.subheader('투수 : {}'.format(team_name))
    st.dataframe(df_pitcher.loc[df_pitcher.팀 == team_name].reset_index(drop=True).drop('팀', axis = 1)) 
    st.write(DATA_URL_P)

with tab3:
   st.subheader('성남 : 전체타자')
   st.dataframe(df_hitter)

with tab4:
   st.subheader('성남 : 전체투수')
   st.dataframe(df_pitcher)

with tab5:
    st.subheader('규정이닝 이상 : {}'.format(team_name))
    #st.write(data_p[0])   

with tab6:
    st.subheader('규정이닝 미달 : {}'.format(team_name))
    #st.write(data_p[1])       

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
        
