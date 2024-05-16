import streamlit as st
from streamlit_gsheets import GSheetsConnection

load_yn = st.button('Load Data')

if load_yn :
    # Create a connection object.
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read()
    df = df.loc[~df.isna().all(1), ~df.isna().all(0)].copy()
    st.subheader('Google Sheets')
    st.dataframe(df, use_container_width = True, hide_index = True)

    st.write(df.isna().all(0))
    st.write(df.isna().all(1))