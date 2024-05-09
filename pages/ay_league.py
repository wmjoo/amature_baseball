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

st.write('🚧 Under Construction ... 🚧')
ALB_URL_SCHD = "http://alb.or.kr/s/schedule/schedule_team_2019.php?id=schedule_team&sc=2&team=%B7%B9%BE%CB%B7%E7%C5%B0%C1%EE&gyear=2024"
