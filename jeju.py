import streamlit as st
import pandas as pd
import folium
from haversine import haversine
from streamlit_folium import st_folium
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="제주 어때? - 호텔 & 맛집 추천", layout="wide")
st.title("🍊제주 어때?🍊 - 데이터가 추천하는 유형별 호텔 & 맛집")

# 데이터 불러오기
df1 = pd.read_csv("JEJU/공모전_군집분석_최종완성끝끝끝!!.csv", encoding='cp949')
tourism_df = pd.read_csv("JEJU/맵자료정리.csv", encoding='cp949')
restaurant_df = pd.read_csv("JEJU/제주도 식당.csv", encoding='cp949')

cluster_col = "군집"

cluster_names = {
    0: "여유로운 중가 리조트 스팟",
    1: "공항 초근접 가성비 호텔존",
    2: "밸런스 끝판왕! 실속 여행자용",
    3: "로드트립에 딱! 무난한 스테이",
    4: "프라이빗 풀옵션 럭셔리 스테이"
}

if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = None

st.header("🏨 여행자 유형에 맞는 숙소 선택")

cols = st.columns(5)
with cols[0]:
    if st.button("🏞 여유로운 중가 리조트 스팟"):
        st.session_state.selected_cluster = 0
with cols[1]:
    if st.button("✈️ 공항 초근접 가성비 호텔존"):
        st.session_state.selected_cluster = 1
with cols[2]:
    if st.button("✨ 밸런스 끝판왕! 실속 여행자용"):
        st.session_state.selected_cluster = 2
with cols[3]:
    if st.button("🧳 로드트립에 딱! 무난한 스테이"):
        st.session_state.selected_cluster = 3
with cols[4]:
    if st.button("🏖 프라이빗 풀옵션 럭셔리 스테이"):
        st.session_state.selected_cluster = 4

# 클러스터 선택 후
if st.session_state.selected_cluster is not None:
    selected_cluster = st.session_state.selected_cluster
    st.markdown(f"### 📌 '{cluster_names[selected_cluster]}' 유형의 호텔을 선택하세요")

    cluster_df = df1[df1[cluster_col] == selected_cluster]
    hotel = st.selectbox("호텔을 선택하세요", cluster_df["숙박업명"].unique())
    st.success(f"✅ 선택한 호텔: {hotel}")





    hotel_row = df1[df1["숙박업명"] == hotel].iloc[0]  # 먼저 정의돼야 함
    hotel_loc = (hotel_row["위도"], hotel_row["경도"])

    # 네이버 지도 검색 링크 (제주 키워드 포함)
    search_query = f"제주 {hotel_row['숙박업명']}"
    naver_map_link = f"https://map.naver.com/v5/search/{search_query}"


    # 주요 편의시설 리스트
    amenities = [
    ("🍽 식당", "식당존재여부"),
    ("🅿️ 주차장", "주차장존재여부"),
    ("☕ 카페", "카페보유여부"),
    ("💆 스파", "스파여부"),
    ("🍷 바", "바여부"),
    ("🔥 사우나", "사우나여부"),
    ("🏊 야외수영장", "야외수영장여부"),
    ("💼 비즈니스센터", "비즈니스센터여부"),
    ("🎉 연회장", "연회장여부"),
    ("🏖 해변 접근성", "해변여부"),
    ("🏋 피트니스", "피트니스센터여부"),
    ]

    # 호텔별 보유 시설 추출
    available_amenities = [label for label, col in amenities if hotel_row.get(col) == 'Y']
    amenities_text = " / ".join(available_amenities) if available_amenities else "없음"

    # 컨테이너에 출력
    with st.container():
        st.markdown(
            f"""
            ### 🏨 선택한 호텔 정보
        
            **이름:** {hotel_row['숙박업명']}  
            **위치:** {hotel_row['구군명']}  
            **가격대:** {hotel_row['가격대']}  
            **객실 수:** {hotel_row['객실수_구간']}  
            **공항 거리:** {hotel_row['공항과의 거리']}  
            **보유 편의시설:** {amenities_text}  
            **주변 관광지:** {hotel_row['주변 관광지']}
            """,
            unsafe_allow_html=True
        )

        # 🔗 네이버 지도 버튼
        st.link_button("📍 제주에서 이 호텔 검색하기 (네이버 지도)", url=naver_map_link)













    # 필터 설정
    with st.expander("🗺️ 지도 필터 설정", expanded=True):
        show_tourism = st.checkbox("추천 관광지 보기", value=True)
        show_restaurant = st.checkbox("추천 식당 보기", value=True)

    # 선택 호텔 위치
    hotel_row = df1[df1["숙박업명"] == hotel].iloc[0]
    hotel_loc = (hotel_row["위도"], hotel_row["경도"])

    # 지도 생성: 호텔 중심!
    m = folium.Map(location=hotel_loc, zoom_start=11)

    # 호텔 마커
    folium.Marker(
        location=hotel_loc,
        popup=folium.Popup(f"🏨 {hotel}", max_width=250),
        tooltip="호텔",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)

    # 추천 관광지
    if show_tourism:
        tourism_df["거리"] = tourism_df.apply(
            lambda row: haversine(hotel_loc, (row["위도"], row["경도"])), axis=1)
        close_tourism = tourism_df.sort_values("거리").groupby("관광지분류").head(3)

        for _, row in close_tourism.iterrows():
            pos = (row["위도"], row["경도"])
            popup_html = f"""
            <b>📍가까운 추천 관광지📍</b><br>
            <b>{row['관광지명']}</b><br>
            분류: {row['관광지분류']}<br>
            주소: {row['주소']}
            """
            folium.Marker(
                location=pos,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row["관광지명"],
                icon=folium.Icon(color="green", icon="info-sign")
            ).add_to(m)
            folium.PolyLine([hotel_loc, pos], color="green", weight=2).add_to(m)

    # 추천 식당
    if show_restaurant:
        restaurant_df["거리"] = restaurant_df.apply(
            lambda row: haversine(hotel_loc, (row["식당위도"], row["식당경도"])), axis=1)
        close_restaurants = restaurant_df.sort_values("거리").head(10)

        for _, row in close_restaurants.iterrows():
            pos = (row["식당위도"], row["식당경도"])
            popup_html = f"""
            <b>🍽가까운 추천 식당🍽</b><br>
            <b>{row['식당명']}</b><br>
            주소: {row['식당주소']}
            """
            folium.Marker(
                location=pos,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row["식당명"],
                icon=folium.Icon(color="red", icon="cutlery")
            ).add_to(m)
            folium.PolyLine([hotel_loc, pos], color="red", weight=2).add_to(m)

    # 지도 출력
    st.markdown("## 📍 지도에서 추천 위치 확인")
    st_folium(m, use_container_width=True, height=720)


st.header("🔎 당신을 위한 맞춤 식당 추천")

user_input = st.text_input("🍽 어떤 분위기나 음식을 원하시나요? (예: 한적한 브런치 카페, 바다가 보이는 식당 등)")

if user_input:

    st.markdown(f"👉 '{user_input}'에 어울리는 식당을 추천드릴게요!")

        # 모델 & 데이터 불러오기
    with open("JEJU/restaurant_data.pkl", "rb") as f:
        restaurant_df = pickle.load(f)

    with open("JEJU/basic_tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open("JEJU/basic_tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    # 입력값 벡터화
    user_vec = tfidf_vectorizer.transform([user_input])

    # 유사도 계산
    sim_scores = cosine_similarity(user_vec, tfidf_matrix)[0]
    top_indices = np.argsort(sim_scores)[::-1][:5]  # 상위 5개

    top_recs = restaurant_df.iloc[top_indices].copy()

    # 지도 생성 (호텔 중심 기준)
    m = folium.Map(location=hotel_loc, zoom_start=12)

    # 호텔 마커
    folium.Marker(
        location=hotel_loc,
        popup="🏨 선택한 호텔",
        tooltip="호텔",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)

    # 추천 식당 마커
    for i, row in top_recs.iterrows():
        pos = (row["식당위도"], row["식당경도"])
        search_query = f"제주 {row['식당명']}"
        naver_link = f"https://map.naver.com/v5/search/{search_query}"
        popup_html = f"""
        <b>🍽 가까운 추천 식당</b><br>
        <b>{row['식당명']}</b><br>
        주소: {row['식당주소']}<br>
        <a href="{naver_link}" target="_blank">🔗 네이버에서 보기</a>
        """
        folium.Marker(
            location=pos,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row["식당명"],
            icon=folium.Icon(color="purple", icon="star")
        ).add_to(m)
        folium.PolyLine([hotel_loc, pos], color="purple", weight=2).add_to(m)

    # 지도 출력
    st.markdown("## 🌟 맞춤 추천 식당 위치")
    st_folium(m, use_container_width=True, height=720)

    top_scores = sim_scores[top_indices] * 100  # 0~1 → 퍼센트

    with st.expander("📋 추천 식당 목록 보기"):
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            name = restaurant_df.iloc[idx]["식당명"]
            addr = restaurant_df.iloc[idx]["식당주소"]
            st.markdown(
                f"""- <b>{name}</b> ({score:.1f}%)<br>주소: {addr}""",
                unsafe_allow_html=True
            )