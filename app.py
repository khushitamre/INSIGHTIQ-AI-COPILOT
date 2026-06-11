import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet

import io
from datetime import datetime

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="INSIGHT IQ AI COPILOT PRO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# SESSION STATE
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv("business_data.csv")

    df.columns = df.columns.str.strip()

    if "Order_Date" in df.columns:

        df["Order_Date"] = pd.to_datetime(
            df["Order_Date"],
            errors="coerce"
        )

        df["Year"] = df["Order_Date"].dt.year
        df["Month"] = df["Order_Date"].dt.month
        df["Quarter"] = "Q" + df["Order_Date"].dt.quarter.astype(str)
        df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)

    return df


df = load_data()

# ==================================================
# COLORS
# ==================================================

PRIMARY = "#7C3AED"
SECONDARY = "#00FFA3"
PINK = "#EC4899"
BLUE = "#38BDF8"
ORANGE = "#FB923C"

PALETTE = [
    PRIMARY,
    SECONDARY,
    PINK,
    BLUE,
    ORANGE
]

# ==================================================
# CHART THEME
# ==================================================

def apply_theme(fig, height=350):

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(
            family="Arial",
            color="white"
        ),
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        )
    )

    return fig

# ==================================================
# HEADER
# ==================================================

st.markdown("""
# 🚀 INSIGHT IQ AI COPILOT PRO

### Next Generation Business Intelligence Platform
""")

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("⚙️ Control Center")

    st.markdown("---")

    years = sorted(df["Year"].dropna().unique())

    selected_years = st.multiselect(
        "Select Year",
        years,
        default=years
    )

    regions = sorted(df["Region"].dropna().unique())

    selected_regions = st.multiselect(
        "Select Region",
        regions,
        default=regions
    )

    segments = sorted(df["Segment"].dropna().unique())

    selected_segments = st.multiselect(
        "Select Segment",
        segments,
        default=segments
    )

    categories = sorted(df["Category"].dropna().unique())

    selected_categories = st.multiselect(
        "Select Category",
        categories,
        default=categories
    )

    st.markdown("---")

    st.success("AI System Online ✅")

    st.markdown(
        """
        ### 👩‍💻 Developer
        
        **Khushi Tamre**
        
        AI & BI Engineer
        
        """
    )

# ==================================================
# FILTER DATA
# ==================================================

filtered_df = df[
    (df["Year"].isin(selected_years))
    &
    (df["Region"].isin(selected_regions))
    &
    (df["Segment"].isin(selected_segments))
    &
    (df["Category"].isin(selected_categories))
]

# ==================================================
# KPI CALCULATIONS
# ==================================================

total_sales = filtered_df["Sales"].sum()

total_profit = filtered_df["Profit"].sum()

total_orders = filtered_df["Order_ID"].nunique()

avg_order_value = (
    total_sales / total_orders
    if total_orders > 0
    else 0
)

profit_margin = (
    total_profit / total_sales * 100
    if total_sales > 0
    else 0
)

# ==================================================
# TOP REGION
# ==================================================

region_sales = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
)

top_region = region_sales.idxmax()

top_region_sales = region_sales.max()

# ==================================================
# TOP CATEGORY
# ==================================================

category_sales = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
)

top_category = category_sales.idxmax()

# ==================================================
# TOP PRODUCT
# ==================================================

product_sales = (
    filtered_df
    .groupby("Product")["Sales"]
    .sum()
)

top_product = product_sales.idxmax()

# ==================================================
# HEALTH SCORE
# ==================================================

if profit_margin >= 20:
    health_score = 95

elif profit_margin >= 15:
    health_score = 85

elif profit_margin >= 10:
    health_score = 75

else:
    health_score = 60

# ==================================================
# RISK LEVEL
# ==================================================

if health_score >= 90:

    risk_level = "Low"

elif health_score >= 75:

    risk_level = "Medium"

else:

    risk_level = "High"

# ==================================================
# AI CONFIDENCE
# ==================================================

if health_score >= 90:

    confidence = "95%"

elif health_score >= 75:

    confidence = "85%"

else:

    confidence = "70%"

# ==================================================
# EXECUTIVE SUMMARY
# ==================================================

business_summary = {
    "Revenue": total_sales,
    "Profit": total_profit,
    "Orders": total_orders,
    "Top Region": top_region,
    "Top Category": top_category,
    "Top Product": top_product,
    "Health Score": health_score,
    "Risk": risk_level
}

# ==================================================
# EXECUTIVE DASHBOARD
# ==================================================

st.markdown("---")

st.subheader("📊 Executive Dashboard")

# ==================================================
# KPI CARDS
# ==================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "💰 Revenue",
        f"₹{total_sales:,.0f}"
    )

with col2:
    st.metric(
        "📈 Profit",
        f"₹{total_profit:,.0f}"
    )

with col3:
    st.metric(
        "🛒 Orders",
        f"{total_orders:,}"
    )

with col4:
    st.metric(
        "🎯 Margin",
        f"{profit_margin:.1f}%"
    )

with col5:
    st.metric(
        "💎 Avg Order",
        f"₹{avg_order_value:,.0f}"
    )

# ==================================================
# STRATEGIC KPIs
# ==================================================

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "🌎 Top Region",
        top_region
    )

with s2:
    st.metric(
        "📦 Top Category",
        top_category
    )

with s3:
    st.metric(
        "🧠 Health Score",
        f"{health_score}/100"
    )

with s4:
    st.metric(
        "⚠️ Risk Level",
        risk_level
    )

# ==================================================
# EXECUTIVE INSIGHT
# ==================================================

st.markdown("### 🤖 AI Executive Insight")

st.info(
    f"""
Top performing region is **{top_region}**.

Best revenue generating category is **{top_category}**.

Current profit margin is **{profit_margin:.1f}%**.

Business health score stands at **{health_score}/100**.

Current risk profile is **{risk_level} Risk**.

Recommended action:
Scale operations in **{top_region}**
and increase investment in **{top_category}**.
"""
)

# ==================================================
# BUSINESS STATUS
# ==================================================

if health_score >= 90:

    st.success(
        "🚀 Excellent Business Performance Detected"
    )

elif health_score >= 75:

    st.warning(
        "⚡ Business Stable But Growth Opportunities Exist"
    )

else:

    st.error(
        "⚠️ High Business Risk Detected"
    )

# ==================================================
# PERFORMANCE LEADERBOARD
# ==================================================

st.markdown("## 🏆 Performance Leaderboard")

top_regions = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

st.markdown("### 🌎 Top Regions")

st.dataframe(
    top_regions,
    use_container_width=True
)

top_products = (
    filtered_df
    .groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

st.markdown("### 📦 Top Products")

st.dataframe(
    top_products,
    use_container_width=True
)

category_performance = (
    filtered_df
    .groupby("Category")
    [["Sales", "Profit"]]
    .sum()
    .reset_index()
)

st.markdown("### 📊 Category Performance")

st.dataframe(
    category_performance,
    use_container_width=True
)

# ==================================================
# REVENUE ANALYTICS
# ==================================================

st.markdown("---")
st.subheader("📈 Revenue Analytics")

monthly_sales = (
    filtered_df
    .groupby("YearMonth")["Sales"]
    .sum()
    .reset_index()
)

fig_trend = px.line(
    monthly_sales,
    x="YearMonth",
    y="Sales",
    title="Monthly Revenue Trend",
    markers=True
)

fig_trend.update_layout(
    height=450
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:

    category_chart = (
        filtered_df
        .groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig_donut = px.pie(
        category_chart,
        names="Category",
        values="Sales",
        hole=0.55,
        title="Revenue by Category"
    )

    fig_donut.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_donut,
        use_container_width=True
    )

with col2:

    region_chart = (
        filtered_df
        .groupby("Region")["Sales"]
        .sum()
        .reset_index()
        .sort_values(
            by="Sales",
            ascending=True
        )
    )

    fig_region = px.bar(
        region_chart,
        x="Sales",
        y="Region",
        orientation="h",
        title="Revenue by Region"
    )

    fig_region.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_region,
        use_container_width=True
    )

st.markdown("---")
st.subheader("🎯 Profitability Analysis")

sample_df = filtered_df.sample(
    min(
        1000,
        len(filtered_df)
    )
)

fig_scatter = px.scatter(
    sample_df,
    x="Sales",
    y="Profit",
    color="Category",
    size="Quantity",
    hover_data=["Product"],
    title="Sales vs Profit Analysis"
)

fig_scatter.update_layout(
    height=550
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

st.markdown("---")
st.subheader("📊 Quarterly Analysis")

quarterly_sales = (
    filtered_df
    .groupby(
        ["Year", "Quarter"]
    )["Sales"]
    .sum()
    .reset_index()
)

quarterly_sales["Period"] = (
    quarterly_sales["Year"].astype(str)
    + " "
    + quarterly_sales["Quarter"]
)

fig_quarter = px.bar(
    quarterly_sales,
    x="Period",
    y="Sales",
    color="Quarter",
    title="Quarterly Revenue"
)

fig_quarter.update_layout(
    height=450
)

st.plotly_chart(
    fig_quarter,
    use_container_width=True
)

st.markdown("---")
st.subheader("🔥 Top Products")

top_products_chart = (
    filtered_df
    .groupby("Product")["Sales"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
    .reset_index()
)

fig_products = px.bar(
    top_products_chart,
    x="Sales",
    y="Product",
    orientation="h",
    title="Top 10 Products"
)

fig_products.update_layout(
    height=500
)

st.plotly_chart(
    fig_products,
    use_container_width=True
)

st.markdown("---")
st.subheader("💰 Category Profit Analysis")

profit_chart = (
    filtered_df
    .groupby("Category")["Profit"]
    .sum()
    .reset_index()
)

fig_profit = px.bar(
    profit_chart,
    x="Category",
    y="Profit",
    color="Category",
    title="Profit by Category"
)

fig_profit.update_layout(
    height=450
)

st.plotly_chart(
    fig_profit,
    use_container_width=True
)

# ==================================================
# AI FORECASTING ENGINE
# ==================================================

st.markdown("---")
st.subheader("🔮 AI Revenue Forecasting")

forecast_df = (
    filtered_df
    .groupby("YearMonth")["Sales"]
    .sum()
    .reset_index()
)

forecast_df["Month_No"] = range(
    1,
    len(forecast_df) + 1
)

model = LinearRegression()

model.fit(
    forecast_df[["Month_No"]],
    forecast_df["Sales"]
)

future_months = np.arange(
    len(forecast_df) + 1,
    len(forecast_df) + 7
).reshape(-1,1)

future_predictions = model.predict(
    future_months
)

next_month_prediction = future_predictions[0]

history_x = list(
    forecast_df["Month_No"]
)

history_y = list(
    forecast_df["Sales"]
)

forecast_x = list(
    future_months.flatten()
)

forecast_y = list(
    future_predictions
)

fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=history_x,
        y=history_y,
        mode="lines+markers",
        name="Historical Revenue"
    )
)

fig_forecast.add_trace(
    go.Scatter(
        x=forecast_x,
        y=forecast_y,
        mode="lines+markers",
        name="AI Forecast"
    )
)

fig_forecast.update_layout(
    title="6 Month Revenue Forecast",
    height=500
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

st.markdown("### 🤖 Forecast Summary")

avg_monthly_sales = forecast_df["Sales"].mean()

growth_percent = (
    (
        next_month_prediction
        -
        avg_monthly_sales
    )
    /
    avg_monthly_sales
) * 100

f1, f2, f3 = st.columns(3)

with f1:

    st.metric(
        "Next Month Forecast",
        f"₹{next_month_prediction:,.0f}"
    )

with f2:

    st.metric(
        "Expected Growth",
        f"{growth_percent:.1f}%"
    )

with f3:

    st.metric(
        "Confidence",
        confidence
    )

st.markdown("---")
st.subheader("🧠 Business Health Gauge")

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={
            "text":"Health Score"
        },
        gauge={
            "axis":{
                "range":[0,100]
            }
        }
    )
)

gauge.update_layout(
    height=450
)

st.plotly_chart(
    gauge,
    use_container_width=True
)

st.markdown("---")
st.subheader("🚨 AI Smart Alerts")

if profit_margin >= 20:

    st.success(
        "✅ Excellent profit margin detected."
    )

elif profit_margin >= 10:

    st.warning(
        "⚠️ Profit margin can be improved."
    )

else:

    st.error(
        "🚨 Low profitability risk detected."
    )

if next_month_prediction > avg_monthly_sales:

    st.info(
        "📈 AI predicts growth next month."
    )

else:

    st.warning(
        "📉 AI predicts slower growth."
    )

st.info(
    f"🌎 Top region remains {top_region}"
)

st.markdown("---")
st.subheader("🤖 AI Recommendations")

recommendation = f"""

1. Expand operations in {top_region}

2. Increase investment in {top_category}

3. Promote top product:
{top_product}

4. Forecast indicates
₹{next_month_prediction:,.0f}
potential revenue next month.

5. Current business health:
{health_score}/100

"""

st.success(
    recommendation
)

# ==================================================
# AI BUSINESS ASSISTANT
# ==================================================

st.markdown("---")
st.subheader("🤖 AI Business Assistant")

question = st.text_input(
    "Ask a business question",
    placeholder="Example: What is the top region?"
)

if question:

    q = question.lower()

    if "region" in q:

        answer = (
            f"Top Region: {top_region}"
        )

    elif "category" in q:

        answer = (
            f"Top Category: {top_category}"
        )

    elif "product" in q:

        answer = (
            f"Top Product: {top_product}"
        )

    elif "profit" in q:

        answer = (
            f"Total Profit: ₹{total_profit:,.0f}"
        )

    elif "revenue" in q or "sales" in q:

        answer = (
            f"Total Revenue: ₹{total_sales:,.0f}"
        )

    elif "forecast" in q:

        answer = (
            f"Next Month Forecast: ₹{next_month_prediction:,.0f}"
        )

    elif "health" in q:

        answer = (
            f"Business Health Score: {health_score}/100"
        )

    elif "risk" in q:

        answer = (
            f"Current Risk Level: {risk_level}"
        )

    else:

        answer = (
            "Try asking about revenue, profit, forecast, product, category, health score or region."
        )

    st.success(answer)

st.markdown("---")
st.subheader("📄 Executive Report")

def create_pdf():

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "INSIGHT IQ AI COPILOT REPORT",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1,20)
    )

    content.append(
        Paragraph(
            f"Total Revenue: ₹{total_sales:,.0f}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Total Profit: ₹{total_profit:,.0f}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Profit Margin: {profit_margin:.1f}%",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Top Region: {top_region}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Top Category: {top_category}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Health Score: {health_score}/100",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Risk Level: {risk_level}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Forecast Revenue: ₹{next_month_prediction:,.0f}",
            styles["Normal"]
        )
    )

    doc.build(content)

    buffer.seek(0)

    return buffer

pdf_file = create_pdf()

st.download_button(
    label="⬇ Download Executive Report",
    data=pdf_file,
    file_name="Insight_IQ_Report.pdf",
    mime="application/pdf"
)

st.markdown("---")
st.subheader("📋 Executive Summary")

st.info(
    f"""
Revenue : ₹{total_sales:,.0f}

Profit : ₹{total_profit:,.0f}

Top Region : {top_region}

Top Category : {top_category}

Top Product : {top_product}

Forecast : ₹{next_month_prediction:,.0f}

Health Score : {health_score}/100

Risk Level : {risk_level}
"""
)

st.markdown("---")

st.markdown(
    """
    ### 🚀 INSIGHT IQ AI COPILOT

    Built By Khushi Tamre

    Python • Streamlit • Plotly • Machine Learning

    Final Year AI & BI Project
    """
)

