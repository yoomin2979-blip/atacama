import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np



st.markdown(
    """
    <style>
    h2 {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("📈 Walmart Sales Forecast")


model = joblib.load("models/xgb_model.pkl")


train_file = st.file_uploader("Upload train.csv", type=["csv"])

stores_file = st.file_uploader("Upload stores.csv", type=["csv"])

features_file = st.file_uploader("Upload features.csv", type=["csv"])


if train_file and stores_file and features_file:

    train = pd.read_csv(train_file)
    stores = pd.read_csv(stores_file)
    features_df = pd.read_csv(features_file)

    train["Date"] = pd.to_datetime(train["Date"])
    features_df["Date"] = pd.to_datetime(features_df["Date"])

    df = train.merge(stores, on="Store", how="left")

    df = df.merge(
        features_df,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    df["Type"] = df["Type"].map({
        "A": 0,
        "B": 1,
        "C": 2
    })

    actual_sales = df["Weekly_Sales"]

    features = [
        "Store",
        "Dept",
        "IsHoliday",
        "Type",
        "Size",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "Month",
        "Year",
        "Week",
        "Quarter"
    ]

    X_predict = df[features]

    prediction = model.predict(X_predict)

    df["Predicted_Sales"] = prediction

    st.sidebar.header("🔍 Filter")

    store_list = ["All"] + sorted(df["Store"].unique().tolist())
    selected_store = st.sidebar.selectbox("Select Store", store_list)

    dept_list = ["All"] + sorted(df["Dept"].unique().tolist())
    selected_dept = st.sidebar.selectbox("Select Department", dept_list)

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df["Date"].min(), df["Date"].max())
    )


    filtered_df = df.copy()

    if selected_store != "All":
        filtered_df = filtered_df[
            filtered_df["Store"] == selected_store
        ]

    if selected_dept != "All":
        filtered_df = filtered_df[
            filtered_df["Dept"] == selected_dept
        ]

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        filtered_df = filtered_df[
            (filtered_df["Date"] >= start_date) &
            (filtered_df["Date"] <= end_date)
        ]

    # -----------------------------
    # KPI Dashboard
    # -----------------------------

    total_sales = filtered_df["Weekly_Sales"].sum()

    avg_weekly_sales = filtered_df.groupby("Date")["Weekly_Sales"].sum().mean()

    best_store = (
        filtered_df.groupby("Store")["Weekly_Sales"]
        .sum()
        .idxmax()
    )

    best_store_sales = (
        filtered_df.groupby("Store")["Weekly_Sales"]
        .sum()
        .max()
    )

    forecast_sales = filtered_df["Predicted_Sales"].sum()


    st.subheader("📊 Sales Overview")


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Actual Sales",
        f"${total_sales:,.0f}"
    )


    col2.metric(
        "Avg Weekly Sales",
        f"${avg_weekly_sales:,.0f}"
    )


    col3.metric(
        "Best Store",
        f"Store {best_store}"
    )


    col4.metric(
        "Forecast Sales",
        f"${forecast_sales:,.0f}"
    )

    compare = (
        filtered_df.groupby("Date")[["Weekly_Sales", "Predicted_Sales"]]
        .sum()
        .reset_index()
    )

    st.subheader("01.Actual Sales vs Predicted Sales")
    fig = px.line(
        compare,
        x="Date",
        y=["Weekly_Sales", "Predicted_Sales"],
        labels={
            "value": "Sales",
            "Date": "Date",
            "variable": "Type",
        },
        color_discrete_map={
         "Weekly_Sales": "#1f77b4",
         "Predicted_Sales": "#ff7f0e"
        }
    )

    fig.update_layout(
        template="simple_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)


    st.subheader("02.Prediction Result")

    st.dataframe(
        filtered_df[
            [
                "Store",
                "Dept",
                "Date",
                "Weekly_Sales",
                "Predicted_Sales"
            ]
        ]
    )


    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Result",
        csv,
        "prediction_result.csv",
        "text/csv"
    )


    compare = (
        filtered_df.groupby("Date")[["Weekly_Sales", "Predicted_Sales"]]
        .sum()
        .reset_index()
    )


    st.subheader("03.Actual vs Predicted Sales by Store")

    store_compare = (
        filtered_df.groupby("Store")[["Weekly_Sales", "Predicted_Sales"]]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        store_compare,
        x="Store",
        y=["Weekly_Sales", "Predicted_Sales"],
        barmode="group",
        title="Actual vs Predicted Sales by Store",
        color_discrete_map={
        "Weekly_Sales": "#1f77b4",
        "Predicted_Sales": "#F97316"
    }
    )

    st.plotly_chart(fig, use_container_width=True)


    st.subheader("04.Predicted Sales by Department")

    dept_compare = (
        filtered_df.groupby("Dept")[["Weekly_Sales", "Predicted_Sales"]]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        dept_compare,
        x="Dept",
        y=["Weekly_Sales", "Predicted_Sales"],
        barmode="group",
        title="Actual vs Predicted Sales by Department",
        color_discrete_map={
        "Weekly_Sales": "#1f77b4",
        "Predicted_Sales": "#F97316"
    }
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Forecast Sales Heatmap
    # -----------------------------

    st.subheader("05.Forecast Sales Heatmap (Store × Department)")


    heatmap_df = (
        filtered_df
        .groupby(["Store", "Dept"])["Predicted_Sales"]
        .sum()
        .reset_index()
    )


    heatmap_data = heatmap_df.pivot(
        index="Store",
        columns="Dept",
        values="Predicted_Sales"
    )


    fig = px.imshow(
        heatmap_data,
        color_continuous_scale=[
        "#eaf2f8",
        "#9ecae1",
        "#4292c6",
        "#2171b5",
        "#1f77b4"
    ],
        labels={
            "x": "Department",
            "y": "Store",
            "color": "Predicted Sales"
        }
    )


    fig.update_layout(
        height=500,
        template="simple_white",
        margin=dict(
            l=50,
            r=50,
            t=30,
            b=50
        )
    )


    st.plotly_chart(fig, use_container_width=True)

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("06.Feature Importance")

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color_discrete_sequence=["#1f77b4"]
    )

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Model Accuracy
    # -----------------------------

    mae = mean_absolute_error(
        filtered_df["Weekly_Sales"],
        filtered_df["Predicted_Sales"]
    )


    rmse = np.sqrt(
        mean_squared_error(
            filtered_df["Weekly_Sales"],
            filtered_df["Predicted_Sales"]
        )
    )


    # WMAPE 계산

    wmape = (
        np.sum(
            np.abs(
                filtered_df["Weekly_Sales"]
                -
                filtered_df["Predicted_Sales"]
            )
        )
        /
        np.sum(
            np.abs(filtered_df["Weekly_Sales"])
        )
    ) * 100


    accuracy = 100 - wmape


    st.subheader("07.Model Performance")


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Accuracy",
        f"{accuracy:.2f}%"
    )


    col2.metric(
        "WMAPE",
        f"{wmape:.2f}%"
    )


    col3.metric(
        "MAE",
        f"{mae:,.0f}"
    )


    col4.metric(
        "RMSE",
        f"{rmse:,.0f}"
    )