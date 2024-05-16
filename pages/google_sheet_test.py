import streamlit as st
from streamlit_gsheets import GSheetsConnection

load_yn = st.button('Load Data')

@st.cache
if load_yn :
    # Create a connection object.
    conn = st.connection("gsheets", type=GSheetsConnection)
    df_hitter = conn.read(worksheet="hitter")    
    df_pitcher = conn.read(worksheet="pitcher")
    df = df_hitter.copy()
    df = df.loc[~df.isna().all(1), ~df.isna().all(0)].copy()
    st.subheader('Google Sheets')
    st.dataframe(df, use_container_width = True, 
                 hide_index = True)
    
    st.subheader('df_pitcher')
    st.dataframe(df_pitcher, use_container_width = True, 
                 hide_index = True)