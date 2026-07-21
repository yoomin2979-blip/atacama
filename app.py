import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="SCM Project",
    page_icon="📊",
    layout="wide"
)

# 메인 화면 타이틀 및 소개
st.title("📑데이터 통합 관리 및 예측 시스템")
st.markdown("""
### 원하시는 메뉴를 왼쪽 사이드바에서 선택해주세요.
* **Sales Analysis**: 과거 판매 데이터 분석 및 시각화 대시보드
* **Sales Forecasting**: AI 모델 기반의 향후 판매량 예측 및 성능 지표 확인
""")
