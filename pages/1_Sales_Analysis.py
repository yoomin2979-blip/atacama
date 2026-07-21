import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# 1. 모델 로드 (가져오신 분석 전용 페이지로 변경되었으므로 메인 app.py의 배포를 방해하지 않게 경로 확인 필요)
model = joblib.load("models/xgb_model.pkl")

# 2. 타이틀 설정
st.title("📊 Walmart Sales Analysis")


# 3. 불필요한 menu 조건문을 없애고 바로 데이터 로드 및 시각화 진행
train = pd.read_csv("data/train.csv")
stores = pd.read_csv("data/stores.csv")
features = pd.read_csv("data/features.csv")

st.subheader("Train Data")
st.write(train.shape)
st.write(train.tail())

st.subheader("Stores Data")
st.dataframe(stores)

st.subheader("Features Data")
st.dataframe(features)

st.subheader("Weekly Sales Trend")

sales = (
    train.groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    sales,
    x="Date",
    y="Weekly_Sales",
    title="Total Weekly Sales"
)

st.plotly_chart(fig, use_container_width=True)

train["Date"] = pd.to_datetime(train["Date"])
train["Month"] = train["Date"].dt.to_period("M").astype(str)

monthly = (
    train.groupby("Month")["Weekly_Sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly,
    x="Month",
    y="Weekly_Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig, use_container_width=True)

train["Year"] = train["Date"].dt.year

yearly = (
    train.groupby("Year")["Weekly_Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    yearly,
    x="Year",
    y="Weekly_Sales",
    text_auto=".2s",
    title="Annual Sales"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", len(train))
col2.metric("Stores", train["Store"].nunique())
col3.metric("Departments", train["Dept"].nunique())

store = st.selectbox(
    "Select Store",
    sorted(train["Store"].unique())
)

filtered = train[train["Store"] == store]
st.dataframe(filtered.head())

st.subheader("Store Sales")

store_sales = (
    train.groupby("Store")["Weekly_Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    store_sales,
    x="Store",
    y="Weekly_Sales"
)

fig.update_traces(
    marker_color="#3B82F6",
    texttemplate="$%{text:,.0f}",
    textposition="outside"
)

fig.update_layout(
    template="simple_white",
    height=500,
    showlegend=False,
    xaxis_title="Total Sales ($)",
    yaxis_title="Store",
    margin=dict(l=20, r=30, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# Top 10 & Bottom 10 Stores
# ==============================

col1, col2 = st.columns(2)

# ------------------------------
# Top 10
# ------------------------------
with col1:
    st.subheader("Top 10 Stores")

    top10 = (
        train.groupby("Store")["Weekly_Sales"]
        .sum()
        .reset_index()
        .nlargest(10, "Weekly_Sales")
        .sort_values("Weekly_Sales", ascending=True)
    )

    fig = px.bar(
        top10,
        x="Weekly_Sales",
        y="Store",
        orientation="h",
        text="Weekly_Sales"
    )

    fig.update_traces(
        marker_color="#2563EB",      # 파란색
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        template="simple_white",
        title="Top 10 Stores",
        title_x=0.5,
        height=450,
        showlegend=False,
        xaxis_title="Total Sales ($)",
        yaxis_title="Store",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Bottom 10
# ------------------------------
with col2:
    st.subheader("Bottom 10 Stores")

    bottom10 = (
        train.groupby("Store")["Weekly_Sales"]
        .sum()
        .reset_index()
        .nsmallest(10, "Weekly_Sales")
        .sort_values("Weekly_Sales", ascending=False)
    )

    fig = px.bar(
        bottom10,
        x="Weekly_Sales",
        y="Store",
        orientation="h",
        text="Weekly_Sales"
    )

    fig.update_traces(
        marker_color="#DC2626",      # 빨간색
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        template="simple_white",
        title="Bottom 10 Stores",
        title_x=0.5,
        height=450,
        showlegend=False,
        xaxis_title="Total Sales ($)",
        yaxis_title="Store",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Holiday Sales")

holiday_sales = (
    train.groupby("IsHoliday")["Weekly_Sales"]
    .mean()
    .reset_index()
)

holiday_sales["IsHoliday"] = holiday_sales["IsHoliday"].replace({
    False: "Non-Holiday",
    True: "Holiday"
})

fig = px.bar(
    holiday_sales,
    x="IsHoliday",
    y="Weekly_Sales",
    color="IsHoliday",
    text_auto=".2s",
    title="Average Weekly Sales"
)
fig.update_traces(width=0.4)

fig.update_layout(
    template="plotly_white",
    showlegend=False,
    height=300,
    xaxis_title="",
    yaxis_title="Average Sales"
)

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# ==============================
# Season Sales
# ==============================

st.subheader("Season Sales")

train["MonthNum"] = train["Date"].dt.month

def season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

train["Season"] = train["MonthNum"].apply(season)

season_sales = (
    train.groupby("Season", as_index=False)["Weekly_Sales"]
    .sum()
)

season_sales["Season"] = pd.Categorical(
    season_sales["Season"],
    categories=["Spring", "Summer", "Fall", "Winter"],
    ordered=True
)
season_sales = season_sales.sort_values("Season")

fig = px.bar(
    season_sales,
    x="Season",
    y="Weekly_Sales",
    color="Season",
    color_discrete_map={
        "Spring": "#0D8D56",
        "Summer": "#D62F2F",
        "Fall": "#F59E0B",
        "Winter": "#2765CA"
    },
    text_auto=".2s",
    title="Seasonal Sales"
)

fig.update_traces(width=0.45)
fig.update_layout(
    template="simple_white",
    height=380,
    showlegend=False,
    bargap=0.45
)
fig.update_yaxes(rangemode="tozero")

st.plotly_chart(fig, use_container_width=True)
