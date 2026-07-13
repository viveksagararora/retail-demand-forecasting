
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")
    weekly_sales = pd.read_csv("weekly_sales.csv")
    monthly_sales = pd.read_csv("monthly_sales.csv")
    cluster_df = pd.read_csv("cluster_data.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    weekly_sales["Order Date"] = pd.to_datetime(weekly_sales["Order Date"])
    monthly_sales["Order Date"] = pd.to_datetime(monthly_sales["Order Date"])
    iso_anomalies = pd.read_csv("iso_anomalies.csv")
    zscore_anomalies = pd.read_csv("zscore_anomalies.csv")

    iso_anomalies["Order Date"] = pd.to_datetime(iso_anomalies["Order Date"])
    zscore_anomalies["Order Date"] = pd.to_datetime(zscore_anomalies["Order Date"]) 

    return (
    df,
    weekly_sales,
    monthly_sales,
    cluster_df,
    iso_anomalies,
    zscore_anomalies
)


(
    df,
    weekly_sales,
    monthly_sales,
    cluster_df,
    iso_anomalies,
    zscore_anomalies
) = load_data()
# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
model_xgb = joblib.load("xgboost_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.title("📊 Retail Demand Forecasting")

page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# ==========================================================
# PAGE 1 : SALES OVERVIEW DASHBOARD
# ==========================================================

if page == "Sales Overview":

    st.title("📊 Sales Overview Dashboard")
    st.write("Interactive dashboard for exploring retail sales trends.")

    st.divider()

    # -----------------------------
    # KPI CARDS
    # -----------------------------

    total_sales = df["Sales"].sum()
    total_orders = len(df)
    total_customers = df["Customer ID"].nunique()
    avg_order = df["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "🛒 Orders",
        f"{total_orders:,}"
    )

    c3.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )

    c4.metric(
        "📦 Avg Order Value",
        f"${avg_order:.2f}"
    )

    st.divider()

    # -----------------------------
    # TOTAL SALES BY YEAR
    # -----------------------------

    yearly_sales = (
        df.groupby(df["Order Date"].dt.year)["Sales"]
        .sum()
        .reset_index()
    )

    yearly_sales.columns = ["Year", "Sales"]

    st.subheader("📈 Total Sales by Year")

    fig1 = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        color="Sales",
        text_auto=".2s",
        color_continuous_scale="Blues"
    )

    fig1.update_layout(
        height=500,
        xaxis_title="Year",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig1,
        width="stretch"
    )

    st.divider()

    # -----------------------------
    # MONTHLY SALES TREND
    # -----------------------------

    st.subheader("📅 Monthly Sales Trend")

    fig2 = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True
    )

    fig2.update_traces(line=dict(width=3))

    fig2.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

    st.divider()

    # -----------------------------
    # REGION & CATEGORY FILTER
    # -----------------------------

    st.subheader("🌍 Sales by Region & Category")

    col1, col2 = st.columns(2)

    region = col1.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    category = col2.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered_df = df[
        (df["Region"] == region) &
        (df["Category"] == category)
    ]

    monthly_filtered = (
        filtered_df
        .groupby(filtered_df["Order Date"].dt.to_period("M"))["Sales"]
        .sum()
        .reset_index()
    )

    monthly_filtered["Order Date"] = monthly_filtered["Order Date"].astype(str)

    fig3 = px.bar(
        monthly_filtered,
        x="Order Date",
        y="Sales",
        color="Sales",
        color_continuous_scale="Viridis"
    )

    fig3.update_layout(
        height=500,
        xaxis_title="Month",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig3,
        width="stretch"
    )

    st.success(
        f"Showing {len(filtered_df)} records for **{region}** region and **{category}** category."
    )

    st.subheader("📋 Filtered Dataset")

    st.dataframe(
        filtered_df,
        width="stretch"
    )
# ==========================================================
# PAGE 2 : FORECAST EXPLORER
# ==========================================================

elif page == "Forecast Explorer":

    st.title("📈 Forecast Explorer")
    st.write("Generate demand forecasts using the trained XGBoost model.")

    st.divider()

    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    if forecast_type == "Category":

        selected = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )

        data = df[df["Category"] == selected]

    else:

        selected = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

        data = df[df["Region"] == selected]

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    # ----------------------------------------
    # Monthly Aggregation
    # ----------------------------------------

    monthly = (
        data
        .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )

    monthly.columns = ["Date", "Sales"]

    # ----------------------------------------
    # Feature Engineering
    # ----------------------------------------

    monthly["Lag_1"] = monthly["Sales"].shift(1)
    monthly["Lag_2"] = monthly["Sales"].shift(2)
    monthly["Lag_3"] = monthly["Sales"].shift(3)

    monthly["Rolling_Mean_3"] = (
        monthly["Sales"]
        .rolling(3)
        .mean()
    )

    monthly["Month"] = monthly["Date"].dt.month
    monthly["Quarter"] = monthly["Date"].dt.quarter

    def get_season(month):

        if month in [12,1,2]:
            return 1

        elif month in [3,4,5]:
            return 2

        elif month in [6,7,8]:
            return 3

        else:
            return 4

    monthly["Season"] = monthly["Month"].apply(get_season)

    monthly = monthly.dropna().reset_index(drop=True)

    # ----------------------------------------
    # Recursive Forecast
    # ----------------------------------------

    history = monthly.copy()

    predictions = []

    future_dates = []

    last_date = history["Date"].iloc[-1]

    for i in range(horizon):

        lag1 = history["Sales"].iloc[-1]
        lag2 = history["Sales"].iloc[-2]
        lag3 = history["Sales"].iloc[-3]

        rolling = history["Sales"].iloc[-3:].mean()

        next_date = last_date + pd.DateOffset(months=1)

        month = next_date.month
        quarter = next_date.quarter
        season = get_season(month)

        X = pd.DataFrame({

            "Lag_1":[lag1],
            "Lag_2":[lag2],
            "Lag_3":[lag3],
            "Rolling_Mean_3":[rolling],
            "Month":[month],
            "Quarter":[quarter],
            "Season":[season]

        })

        pred = model_xgb.predict(X)[0]

        predictions.append(pred)

        future_dates.append(next_date)

        history.loc[len(history)] = [

            next_date,
            pred,
            lag1,
            lag2,
            lag3,
            rolling,
            month,
            quarter,
            season

        ]

        last_date = next_date

    forecast_df = pd.DataFrame({

        "Forecast Date":future_dates,
        "Forecast Sales":predictions

    })
      
    st.divider()

    # ----------------------------------------
    # Forecast Chart
    # ----------------------------------------

    st.subheader("📈 Forecast Results")

    fig = go.Figure()

    # Historical Sales
    fig.add_trace(
        go.Scatter(
            x=monthly["Date"],
            y=monthly["Sales"],
            mode="lines+markers",
            name="Historical Sales"
        )
    )

    # Forecast Sales
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Forecast Date"],
            y=forecast_df["Forecast Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="red", width=3)
        )
    )

    fig.update_layout(

        title=f"Sales Forecast for {selected}",

        xaxis_title="Date",

        yaxis_title="Sales",

        template="plotly_white",

        height=550

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.divider()

    # ----------------------------------------
    # Forecast Table
    # ----------------------------------------

    st.subheader("📋 Forecast Values")

    forecast_display = forecast_df.copy()

    forecast_display["Forecast Date"] = (
        forecast_display["Forecast Date"]
        .dt.strftime("%b %Y")
    )

    forecast_display["Forecast Sales"] = (
        forecast_display["Forecast Sales"]
        .round(2)
    )

    st.dataframe(
        forecast_display,
        width="stretch"
    )

    st.divider()

    # ----------------------------------------
    # Model Performance
    # ----------------------------------------

    st.subheader("📊 Model Performance")

    # Values obtained from Task 3
    mae = 14763.81
    rmse = 18337.41
    mape = 14.48

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    c2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    c3.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

    st.info(
        """
        **Best Model:** XGBoost

        This model was selected because it achieved the lowest MAE,
        RMSE and MAPE among SARIMA, Prophet and XGBoost during
        model evaluation.
        """
    )
# ==========================================================
# PAGE 3 : ANOMALY REPORT
# ==========================================================

elif page == "Anomaly Report":

    st.title("🚨 Sales Anomaly Report")
    st.write("Identify unusual sales patterns detected using Isolation Forest and Z-Score.")

    st.divider()

    # -----------------------------
    # Weekly Sales Chart
    # -----------------------------

    st.subheader("Weekly Sales with Detected Anomalies")

    fig = go.Figure()

    # Weekly Sales
    fig.add_trace(
        go.Scatter(
            x=weekly_sales["Order Date"],
            y=weekly_sales["Sales"],
            mode="lines",
            name="Weekly Sales",
            line=dict(width=2)
        )
    )

    # Isolation Forest Anomalies
    fig.add_trace(
        go.Scatter(
            x=iso_anomalies["Order Date"],
            y=iso_anomalies["Sales"],
            mode="markers",
            name="Isolation Forest",
            marker=dict(
                color="red",
                size=10,
                symbol="circle"
            )
        )
    )

    # Z-Score Anomalies
    fig.add_trace(
        go.Scatter(
            x=zscore_anomalies["Order Date"],
            y=zscore_anomalies["Sales"],
            mode="markers",
            name="Z-Score",
            marker=dict(
                color="blue",
                size=10,
                symbol="diamond"
            )
        )
    )

    fig.update_layout(

        title="Weekly Sales Anomaly Detection",

        xaxis_title="Date",

        yaxis_title="Sales",

        template="plotly_white",

        height=550

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.divider()

    # -----------------------------
    # Isolation Forest Table
    # -----------------------------

    st.subheader("Isolation Forest Anomalies")

    iso_display = iso_anomalies[["Order Date", "Sales"]].copy()

    iso_display["Order Date"] = (
        pd.to_datetime(iso_display["Order Date"])
        .dt.strftime("%d-%b-%Y")
    )

    iso_display["Sales"] = iso_display["Sales"].round(2)

    st.dataframe(
        iso_display,
        width="stretch"
    )

    st.divider()

    # -----------------------------
    # Z Score Table
    # -----------------------------

    st.subheader("Z-Score Anomalies")

    z_display = zscore_anomalies[["Order Date", "Sales"]].copy()

    z_display["Order Date"] = (
        pd.to_datetime(z_display["Order Date"])
        .dt.strftime("%d-%b-%Y")
    )

    z_display["Sales"] = z_display["Sales"].round(2)

    st.dataframe(
        z_display,
        width="stretch"
    )

    st.divider()

    # -----------------------------
    # KPI Cards
    # -----------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Isolation Forest",
        len(iso_anomalies)
    )

    c2.metric(
        "Z-Score",
        len(zscore_anomalies)
    )

    common = pd.merge(
        iso_anomalies,
        zscore_anomalies,
        on="Order Date"
    )

    c3.metric(
        "Common Anomalies",
        len(common)
    )

    st.info(
        """
        Isolation Forest identifies anomalies using an unsupervised
        machine learning approach, while Z-Score flags observations
        that deviate significantly from the statistical mean.
        Common anomalies detected by both methods are considered
        the strongest indicators of unusual sales behavior.
        """
    )
    # ==========================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# ==========================================================

elif page == "Demand Segments":

    st.title("📦 Product Demand Segments")
    st.write("Visualize product clusters and their demand characteristics.")

    st.divider()

    # -----------------------------
    # Cluster Scatter Plot
    # -----------------------------

    st.subheader("Demand Segmentation using K-Means Clustering")
    # -----------------------------
    # Generate PCA Coordinates
    # -----------------------------

    features = cluster_df[[
        "Total Sales",
        "Growth Rate",
        "Volatility",
        "Average Order Value"
    ]]

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(scaled_features)

    cluster_df["PC1"] = pca_result[:, 0]
    cluster_df["PC2"] = pca_result[:, 1]

    fig = px.scatter(

        cluster_df,

        x="PC1",

        y="PC2",

        color="Demand Segment",

        hover_name="Sub-Category",

        size="Total Sales",

        title="Product Demand Clusters"

    )

    fig.update_layout(

        template="plotly_white",

        height=600

    )

    st.plotly_chart(

        fig,

        width="stretch"

    )

    st.divider()

    # -----------------------------
    # Cluster Table
    # -----------------------------

    st.subheader("Product Demand Segments")

    display_df = cluster_df[[

        "Sub-Category",

        "Demand Segment",

        "Cluster",

        "Total Sales",

        "Growth Rate",

        "Volatility",

        "Average Order Value"

    ]]

    st.dataframe(

        display_df,

        width="stretch"

    )

    st.divider()

    # -----------------------------
    # Cluster Summary
    # -----------------------------

    st.subheader("Products in Each Demand Segment")

    summary = (

        cluster_df

        .groupby("Demand Segment")["Sub-Category"]

        .apply(list)

        .reset_index()

    )

    for i in range(len(summary)):

        st.markdown(f"### {summary.iloc[i]['Demand Segment']}")

        st.write(", ".join(summary.iloc[i]["Sub-Category"]))

    st.divider()

    # -----------------------------
    # Stocking Strategy
    # -----------------------------

    st.subheader("Recommended Stocking Strategy")

    st.success("""

✅ **High Value Products**

Maintain adequate inventory while avoiding excess stock due to high unit cost.

---

✅ **Stable & Growing Demand**

Keep regular inventory with scheduled replenishment to meet consistent customer demand.

---

✅ **High Volume, High Volatility**

Maintain higher safety stock and monitor demand frequently to avoid stockouts.

---

✅ **Rapidly Growing Niche Products**

Increase inventory gradually and closely monitor future demand trends before scaling further.

    """)
