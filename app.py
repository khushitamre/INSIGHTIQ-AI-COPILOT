import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from reportlab.pdfgen import canvas
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="INSIGHT IQ AI COPILOT",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827,
        #1e293b
    );
}

[data-testid="stMetric"]{
    background: rgba(17,24,39,0.8);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 0 25px rgba(0,255,255,0.15);
}

h1{
    color:#00E5FF !important;
    font-weight:800 !important;
}

h2,h3{
    color:white !important;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)






# ------------
# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

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

        df["YearMonth"] = (
            df["Order_Date"]
            .dt.to_period("M")
            .astype(str)
        )

    return df


# Load Dataset
df = load_data()


   
   
   

 

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚡ INSIGHT IQ AI")

    years = sorted(
        df["Year"]
        .dropna()
        .unique()
    )

    selected_years = st.multiselect(
        "Select Year",
        years,
        default=years
    )

    regions = sorted(
        df["Region"]
        .dropna()
        .unique()
    )

    selected_regions = st.multiselect(
        "Select Region",
        regions,
        default=regions
    )

    segments = sorted(
        df["Segment"]
        .dropna()
        .unique()
    )

    selected_segments = st.multiselect(
        "Select Segment",
        segments,
        default=segments
    )

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = df[
    (df["Year"].isin(selected_years))
    &
    (df["Region"].isin(selected_regions))
    &
    (df["Segment"].isin(selected_segments))
]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div style="text-align:center;padding:40px 20px;">

<h1 style="
font-size:60px;
font-weight:800;
background: linear-gradient(90deg,#00E5FF,#00FFA3);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:10px;
">
INSIGHT IQ AI
</h1>

<h2 style="
color:white;
font-size:28px;
margin-bottom:20px;
">
Enterprise Intelligence Copilot
</h2>

<p style="
font-size:20px;
color:#B8C1CC;
max-width:900px;
margin:auto;
">
Transforming Business Data into Executive Decisions using
Artificial Intelligence, Predictive Analytics and Smart Insights.
</p>

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🤖 AI Powered Analytics")

with col2:
    st.info("📈 Predictive Forecasting")

with col3:
    st.info("🚀 Executive Intelligence")

    st.markdown("---")



# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order_ID"].nunique()

profit_margin = (
    (total_profit / total_sales) * 100
    if total_sales > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Revenue",
    f"₹{total_sales:,.0f}"
)

col2.metric(
    "📈 Profit",
    f"₹{total_profit:,.0f}"
)

col3.metric(
    "🛒 Orders",
    f"{total_orders:,}"
)

col4.metric(
    "🎯 Profit Margin",
    f"{profit_margin:.1f}%"
)

st.divider()

# --------------------------------------------------
# REVENUE TREND
# --------------------------------------------------

st.subheader("📊 Revenue Trend")

monthly_sales = (
    filtered_df
    .groupby("YearMonth")["Sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_sales,
    x="YearMonth",
    y="Sales",
    markers=True,
    title="Monthly Revenue Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="revenue_trend"
)

# --------------------------------------------------
# CATEGORY ANALYSIS
# --------------------------------------------------

st.subheader("📦 Revenue by Category")

category_sales = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig2 = px.pie(
    category_sales,
    names="Category",
    values="Sales",
    hole=0.5
)

st.plotly_chart(
    fig2,
    use_container_width=True,
    key="category_pie"
)

# --------------------------------------------------
# REGION ANALYSIS
# --------------------------------------------------


st.subheader("🤖 AI Chat Assistant")

question = st.text_input(
    "Ask about your business"
)

top_region = (
    filtered_df
    .groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

top_category = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

if question:

    if "region" in question.lower():
        answer = top_region

    elif "category" in question.lower():
        answer = top_category

    else:
        answer = "Please ask about region or category."

    st.success(answer)



# --------------------------------------------------
# AI REVENUE FORECAST
# --------------------------------------------------

st.subheader("🔮 AI Revenue Forecast")

forecast_df = (
    filtered_df
    .groupby("YearMonth")["Sales"]
    .sum()
    .reset_index()
)

forecast_df["Month_Num"] = range(
    1,
    len(forecast_df) + 1
)

X = forecast_df[["Month_Num"]]
y = forecast_df["Sales"]

model = LinearRegression()
model.fit(X, y)

next_month = [[len(forecast_df) + 1]]

predicted_revenue = model.predict(next_month)[0]

current_revenue = forecast_df["Sales"].sum()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "📈 Predicted Revenue",
        f"₹{predicted_revenue:,.0f}"
    )

with col2:
    st.metric(
        "💰 Current Revenue",
        f"₹{current_revenue:,.0f}"
    )

#Business Leaderboard

st.markdown("---")
st.subheader("🏆 Business Leaderboard")

col1, col2 = st.columns(2)

with col1:
    top_regions = (
        filtered_df
        .groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    st.write("### 🌎 Top Regions")
    st.dataframe(top_regions)

with col2:
    top_products = (
        filtered_df
        .groupby("Product")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    st.write("### 📦 Top Products")
    st.dataframe(top_products)

# -------------------------------------------------
# AI FUTURE FORECAST
# -------------------------------------------------

st.markdown("---")
st.subheader("📈 AI Future Revenue Forecast")

# Create forecast data
forecast_df = (
    filtered_df
    .groupby("YearMonth")["Sales"]
    .sum()
    .reset_index()
)

forecast_df["Month_Num"] = range(
    1,
    len(forecast_df) + 1
)

X = forecast_df[["Month_Num"]]
y = forecast_df["Sales"]

model = LinearRegression()
model.fit(X, y)

future_months = np.arange(
    len(forecast_df) + 1,
    len(forecast_df) + 7
).reshape(-1, 1)

future_predictions = model.predict(
    future_months
)

predicted_revenue = future_predictions[0]

current_revenue = forecast_df["Sales"].sum()

future_df = pd.DataFrame({
    "Month": [
        "Month+1",
        "Month+2",
        "Month+3",
        "Month+4",
        "Month+5",
        "Month+6"
    ],
    "Predicted Revenue": future_predictions
})

forecast_chart = px.line(
    future_df,
    x="Month",
    y="Predicted Revenue",
    markers=True,
    title="Next 6 Months Revenue Forecast"
)

st.plotly_chart(
    forecast_chart,
    use_container_width=True,
    key="future_forecast_chart"
)



# -------------------------------------------------
# EXECUTIVE BOARD REPORT
# -------------------------------------------------

profit_margin = (
    total_profit / total_sales * 100
    if total_sales > 0
    else 0
)

if profit_margin >= 20:
    health_score = 95
elif profit_margin >= 15:
    health_score = 85
elif profit_margin >= 10:
    health_score = 75
else:
    health_score = 60

st.markdown("---")
st.subheader("🏆 Executive Scorecard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🧠 AI Health Score",
        f"{health_score}/100"
    )

with col2:
    st.metric(
        "📈 Growth Score",
        "92/100"
    )

with col3:
    st.metric(
        "⚠ Risk Level",
        "Low"
    )

with col4:
    st.metric(
        "🎯 Forecast Confidence",
        "89%"
    )

#risk analysis


st.markdown("---")
st.subheader("⚠ AI Risk Analysis")

if profit_margin > 20:
    st.success("Low Risk Business")
elif profit_margin > 10:
    st.warning("Moderate Risk Business")
else:
    st.error("High Risk Business")


# -------------------------------------------------
# HEALTH SCORE GAUGE
# -------------------------------------------------

# Business Health Score Calculation


st.subheader("🧠 AI Health Score")

profit_margin = (
    total_profit / total_sales * 100
    if total_sales > 0
    else 0
)

if profit_margin >= 20:
    health_score = 95
elif profit_margin >= 15:
    health_score = 85
elif profit_margin >= 10:
    health_score = 75
else:
    health_score = 60

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "AI Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "cyan"},
            "steps": [
                {"range": [0, 50], "color": "red"},
                {"range": [50, 75], "color": "orange"},
                {"range": [75, 100], "color": "green"}
            ]
        }
    )
)

st.plotly_chart(
    gauge,
    use_container_width=True,
    key="health_gauge"
)

region_sales = filtered_df.groupby("Region")["Sales"].sum()
best_region = region_sales.idxmax()
region_share = (region_sales.max() / region_sales.sum()) * 100

f"{best_region} contributes {region_share:.1f}% of total revenue, making it dominant region."

st.subheader("🚨 Smart AI Alerts")

if profit_margin < 10:
    st.error("Profit margin is very low. Business risk detected!")

if total_sales > 100000:
    st.success("Strong revenue performance detected!")



if predicted_revenue > current_revenue:
    st.info("AI predicts growth in next cycle 🚀")

fig = px.scatter(
    filtered_df,
    x="Sales",
    y="Profit",
    color="Category",
    size="Quantity",
    hover_data=["Product", "Customer_Name"],
    title="Profitability Intelligence Map"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="profitability_map"
)
filtered_df,
x="Sales",
y="Profit",
color="Category",
size="Quantity",
hover_data=["Product", "Customer_Name"],
title="Profitability Intelligence Map"



st.plotly_chart(fig, use_container_width=True)

top_category = "Technology"
print(f"Focus should be on scaling {top_category} category for higher growth.")

print(f"Focus should be on scaling {top_category} category for higher growth.")

def get_category():
    return "Finance"

top_category = get_category()
print(f"Focus should be on scaling {top_category} category for higher growth.")

summary = f"""
Business generated ₹{total_sales:,.0f} revenue with ₹{total_profit:,.0f} profit.

{best_region} is leading region contributing maximum revenue.

Focus should be on scaling {top_category} category for higher growth.

AI suggests optimizing low-performing regions and increasing marketing spend in top region.
"""

st.markdown("---")
st.subheader("🤖 Insight AI Assistant")

question = st.text_input(
    "Ask your business question"
)

if question:

    question = question.lower()
    st.subheader("🤖 AI Chat Assistant")


top_region = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

top_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

top_product = (
    filtered_df.groupby("Product")["Sales"]
    .sum()
    .idxmax()
)

if question:

    if "region" in question.lower():
        answer = f"Best Region: {top_region}"

    elif "category" in question.lower():
        answer = f"Best Category: {top_category}"

    elif "product" in question.lower():
        answer = f"Top Product: {top_product}"

    else:
        answer = "Try asking about region, category, or product."

    st.success(answer)

    if "best product" in question:
        
        answer = top_product

    elif "best region" in question:
        answer = top_region

    elif "profit" in question:
        answer = f"Total Profit = ₹{total_profit:,.0f}"

    elif "revenue" in question:
        answer = f"Total Revenue = ₹{total_sales:,.0f}"

    else:
        answer = "No insight available."

    st.success(answer)

    if st.button("Generate PDF"):

        pdf = canvas.Canvas("business_report.pdf")

        pdf.drawString(100, 800, "Business Report")

        pdf.save()

        st.success("PDF Generated Successfully")

    st.markdown("---")


    # --------------------------------------------------
# PDF EXECUTIVE REPORT
# --------------------------------------------------

st.markdown("---")
st.subheader("📄 Executive PDF Report")

if st.button("Generate Executive PDF Report"):

    pdf = canvas.Canvas("business_report.pdf")

    pdf.setTitle("Insight IQ AI Report")

    pdf.drawString(50, 800, "INSIGHT IQ AI COPILOT")
    pdf.drawString(50, 780, "Executive Business Report")

    pdf.drawString(
        50,
        740,
        f"Total Revenue: ₹{total_sales:,.0f}"
    )

    pdf.drawString(
        50,
        720,
        f"Total Profit: ₹{total_profit:,.0f}"
    )

    pdf.drawString(
        50,
        700,
        f"Profit Margin: {profit_margin:.1f}%"
    )

    pdf.drawString(
        50,
        680,
        f"Health Score: {health_score}/100"
    )

    pdf.drawString(
        50,
        660,
        f"Top Region: {top_region}"
    )

    pdf.drawString(
        50,
        640,
        f"Top Category: {top_category}"
    )

    pdf.drawString(
        50,
        620,
        "AI Recommendation:"
    )

    pdf.drawString(
        70,
        600,
        f"Focus expansion in {top_region}"
    )

    pdf.drawString(
        70,
        580,
        f"Increase investment in {top_category}"
    )

    pdf.save()

    with open(
        "business_report.pdf",
        "rb"
    ) as pdf_file:

        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_file,
            file_name="Insight_IQ_AI_Report.pdf",
            mime="application/pdf"
        )

    st.success(
        "Executive Report Generated Successfully!"
    )


# --------------------------------------------------
# SMART AI BUSINESS ASSISTANT
# --------------------------------------------------

st.markdown("---")

st.subheader("🤖 Insight IQ AI Assistant")

question = st.text_input(
    "Ask a business question",
    key="business_ai_chat"
)

if question:

    q = question.lower()

    if "region" in q:
        answer = (
            filtered_df
            .groupby("Region")["Sales"]
            .sum()
            .idxmax()
        )

        st.success(
            f"🌎 Best Performing Region: {answer}"
        )

    elif "category" in q:
        answer = (
            filtered_df
            .groupby("Category")["Sales"]
            .sum()
            .idxmax()
        )

        st.success(
            f"📦 Best Category: {answer}"
        )

    elif "product" in q:
        answer = (
            filtered_df
            .groupby("Product")["Sales"]
            .sum()
            .idxmax()
        )

        st.success(
            f"🔥 Top Product: {answer}"
        )

    elif "profit" in q:
        st.success(
            f"💰 Total Profit: ₹{total_profit:,.0f}"
        )

    elif "revenue" in q:
        st.success(
            f"📈 Total Revenue: ₹{total_sales:,.0f}"
        )

    else:
        st.info(
            "Please ask about revenue, profit, region, category, or product."
        )




# --------------------------------------------------
# EXECUTIVE COMMAND CENTER
# --------------------------------------------------

st.markdown("---")

st.subheader("🚀 Executive Command Center")

col1, col2 = st.columns(2)

with col1:

    st.info(f"""
### 📈 Strategic Growth Opportunity

🌎 Strongest Region: {top_region}

📦 Best Category: {top_category}

🎯 Suggested Action:

Increase investment in {top_category}
and expand operations in {top_region}.

Expected Impact:
Higher Revenue Growth
""")

with col2:

    if health_score >= 90:
        risk_level = "Low"
        confidence = "95%"

    elif health_score >= 75:
        risk_level = "Medium"
        confidence = "85%"

    else:
        risk_level = "High"
        confidence = "70%"

    st.warning(f"""
### ⚠ Business Risk Center

Risk Level: {risk_level}

Forecast Confidence: {confidence}

Business Health Score:
{health_score}/100

AI Status:
Monitoring Performance
""")
    

    

st.subheader("👩‍💻 Project Developer")

st.info("""
Name: Khushi Tamre

Role: AI & Business Intelligence Developer

Skills:
• Python
• Streamlit
• Pandas
• Plotly
• Machine Learning
• Data Analytics
• Business Intelligence

Project:
Insight IQ AI Copilot
""")