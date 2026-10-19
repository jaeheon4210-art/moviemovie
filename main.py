import math
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 제목 및 아이콘 설정
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")

# 2. 데이터 로드
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL)


raw_df = load_data()

# 3. 데이터 전처리 및 파생 속성 생성
req_cols = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
    "movieNm",
]
df = raw_df.dropna(subset=req_cols).copy()
df = df[
    (df["first_week_audi"] > 0) & (df["first_scrn"] > 0) & (df["total_audi"] > 0)
]

df["first_scrn_log"] = df["first_scrn"].apply(math.log10)
df["total_audi_log"] = df["total_audi"].apply(math.log10)
df["long_run_index"] = df["total_audi"] / df["first_week_audi"]

st.write(f"전체 영화: **{len(raw_df)}**편 | 분석 대상 영화: **{len(df)}**편")

feature_dict = {
    "스크린 수 (상용로그)": "first_scrn_log",
    "누적 관객 (상용로그)": "total_audi_log",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 4. 군집화 설정 (속성 선택 및 묶음 수 k 조절)
st.markdown("---")
st.subheader("⚙️ 군집화 설정")

col_feat, col_k = st.columns([3, 1])

with col_feat:
  selected_feature_names = st.multiselect(
      "군집화에 사용할 속성을 선택하세요 (2개 이상 선택):",
      options=list(feature_dict.keys()),
      default=list(feature_dict.keys()),
  )

with col_k:
  n_clusters = st.slider(
      "묶음 수 (k) 선택:", min_value=2, max_value=7, value=3, step=1
  )

if len(selected_feature_names) < 2:
  st.warning("⚠️ 군집화를 위해 최소 2개 이상의 속성을 선택해 주세요.")
  st.stop()

selected_feature_cols = [feature_dict[name] for name in selected_feature_names]

# 5. 데이터 표준화 및 K-평균 군집화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_feature_cols])

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

circle_labels = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

cluster_means = df.groupby("cluster_raw")["total_audi"].mean()
sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()

label_map = {
    raw_id: label
    for raw_id, label in zip(sorted_clusters, circle_labels[:n_clusters])
}
df["묶음"] = df["cluster_raw"].map(label_map)

active_labels = circle_labels[:n_clusters]

# 6. 실루엣 점수 표시
sil_score = silhouette_score(X_scaled, df["cluster_raw"])
st.info(
    f"💡 **현재 고른 묶음 수({n_clusters})의 실루엣 점수:** **{sil_score:.3f}**"
    " (점수는 -1에서 1 사이이고 1에 가까울수록 묶음이 뚜렷하다는 뜻입니다.)"
)

# 7. 2차원 산점도
st.markdown("---")
st.subheader("📊 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
  x_2d = st.selectbox(
      "2D 가로축 속성", options=list(feature_dict.keys()), index=0, key="x_2d"
  )
with col2:
  y_2d = st.selectbox(
      "2D 세로축 속성", options=list(feature_dict.keys()), index=1, key="y_2d"
  )

labels_2d_dict = {feature_dict[x_2d]: x_2d, feature_dict[y_2d]: y_2d}

fig_2d = px.scatter(
    df,
    x=feature_dict[x_2d],
    y=feature_dict[y_2d],
    color="묶음",
    hover_name="movieNm",
    labels=labels_2d_dict,
    category_orders={"묶음": active_labels},
    title=f"2D 산점도 ({x_2d} vs {y_2d})",
)
st.plotly_chart(fig_2d, use_container_width=True)

# 8. 3차원 산점도
st.markdown("---")
st.subheader("🎲 3차원 산점도")
c1, c2, c3 = st.columns(3)
with c1:
  x_3d = st.selectbox(
      "3D X축 속성", options=list(feature_dict.keys()), index=0, key="x_3d"
  )
with c2:
  y_3d = st.selectbox(
      "3D Y축 속성", options=list(feature_dict.keys()), index=1, key="y_3d"
  )
with c3:
  z_3d = st.selectbox(
      "3D Z축 속성", options=list(feature_dict.keys()), index=2, key="z_3d"
  )

labels_3d_dict = {
    feature_dict[x_3d]: x_3d,
    feature_dict[y_3d]: y_3d,
    feature_dict[z_3d]: z_3d,
}

fig_3d = px.scatter_3d(
    df,
    x=feature_dict[x_3d],
    y=feature_dict[y_3d],
    z=feature_dict[z_3d],
    color="묶음",
    hover_name="movieNm",
    labels=labels_3d_dict,
    category_orders={"묶음": active_labels},
    title=f"3D 산점도 ({x_3d} x {y_3d} x {z_3d})",
)
st.plotly_chart(fig_3d, use_container_width=True)

# 9. 묶음
