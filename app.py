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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import io
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="INSIGHT IQ AI COPILOT PRO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== ULTRA PREMIUM CSS ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #000428 100%);
        color: white;
    }
    .title {
        font-size: 4.2rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #7C3AED, #00F5FF, #FF00CC, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-shadow: 0 0 40px rgba(0, 245, 255, 0.6);
    }
    .subtitle {
        text-align: center;
        font-size: 1.5rem;
        color: #00F5FF;
        letter-spacing: 5px;
        margin-top: -20px;
    }
    .glass {
        background: rgba(255,255,255,0.07);
        border-radius: 22px;
        padding: 22px;
        border: 1px solid rgba(0, 255, 255, 0.25);
        backdrop-filter: blur(20px);
    }
    .neon { text-shadow: 0 0 25px #00F5FF; }
    .metric-value { font-size: 2.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ====================== DATA LOAD ======================
@st.cache_data
def load_data():
    df = pd.read_csv("business_data.csv")
    df.columns = df.columns.str.strip()
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df["Year"] = df["Order_Date"].dt.year
        df["Month"] = df["Order_Date"].dt.month
        df["Quarter"] = "Q" + df["Order_Date"].dt.quarter.astype(str)
        df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    return df

df = load_data()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⚡ INSIGHT IQ")
    st.markdown("### AI Business Copilot Pro")
    st.markdown("---")
    
    years = sorted(df["Year"].dropna().unique())
    selected_years = st.multiselect("Select Year", years, default=years)
    
    regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.multiselect("Select Region", regions, default=regions)
    
    segments = sorted(df["Segment"].dropna().unique())
    selected_segments = st.multiselect("Select Segment", segments, default=segments)
    
    categories = sorted(df["Category"].dropna().unique())
    selected_categories = st.multiselect("Select Category", categories, default=categories)
    
    st.markdown("---")
    st.success("🚀 AI System Fully Operational")
    st.markdown("### 👩‍💻 **Khushi Tamre**  \nAI & BI Engineer")

# ====================== FILTERED DATA ======================
filtered_df = df[
    (df["Year"].isin(selected_years)) &
    (df["Region"].isin(selected_regions)) &
    (df["Segment"].isin(selected_segments)) &
    (df["Category"].isin(selected_categories))
]

# ====================== KPIs ======================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order_ID"].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
avg_order_value = total_sales / total_orders if total_orders > 0 else 0

top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()

health_score = 95 if profit_margin >= 20 else 85 if profit_margin >= 15 else 75 if profit_margin >= 10 else 60
risk_level = "Low" if health_score >= 90 else "Medium" if health_score >= 75 else "High"

# ====================== HEADER ======================
st.markdown('<h1 class="title">INSIGHT IQ</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle neon">AI POWERED BUSINESS INTELLIGENCE PLATFORM</p>', unsafe_allow_html=True)
st.markdown("---")

# ====================== KPI CARDS ======================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">TOTAL REVENUE</p><p class="metric-value" style="color:#00F5FF;">₹{total_sales:,.0f}</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">TOTAL PROFIT</p><p class="metric-value" style="color:#00FFAA;">₹{total_profit:,.0f}</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">TOTAL ORDERS</p><p class="metric-value" style="color:#FF00CC;">{total_orders:,}</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">PROFIT MARGIN</p><p class="metric-value" style="color:#FFD700;">{profit_margin:.1f}%</p></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">HEALTH SCORE</p><p class="metric-value" style="color:#00FFAA;">{health_score}/100</p></div>', unsafe_allow_html=True)

# Strategic KPIs
st.markdown("---")
colA, colB, colC, colD = st.columns(4)
with colA: st.metric("🌍 Top Region", top_region)
with colB: st.metric("📦 Top Category", top_category)
with colC: st.metric("🧠 Health Score", f"{health_score}/100")
with colD: st.metric("⚠️ Risk Level", risk_level)

# AI Executive Insight
st.info(f"""
**🤖 AI Insight**: Top performing region is **{top_region}**. Best category is **{top_category}**. 
Profit margin **{profit_margin:.1f}%**. Business health **{health_score}/100** ({risk_level} Risk).
**Recommendation**: Scale operations in **{top_region}** and invest heavily in **{top_category}**.
""")

# ====================== CHARTS SECTION ======================
st.markdown("---")
st.subheader("📈 Revenue Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Category", "Region", "Profitability"])

with tab1:
    monthly = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
    fig = px.line(monthly, x="YearMonth", y="Sales", title="Monthly Revenue Trend", markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    cat_df = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    fig_pie = px.pie(cat_df, names="Category", values="Sales", hole=0.6, title="Revenue by Category", template="plotly_dark")
    st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    reg_df = filtered_df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
    fig_bar = px.bar(reg_df, x="Region", y="Sales", title="Revenue by Region", template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    sample = filtered_df.sample(min(800, len(filtered_df)))
    fig_scatter = px.scatter(sample, x="Sales", y="Profit", color="Category", size="Quantity", 
                           hover_data=["Product"], title="Sales vs Profit Analysis", template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ====================== AI FORECAST + GAUGE ======================
st.markdown("---")
colF1, colF2 = st.columns([3,2])

with colF1:
    st.subheader("🔮 AI Revenue Forecasting")
    forecast_df = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
    forecast_df["Month_No"] = range(1, len(forecast_df)+1)
    model = LinearRegression().fit(forecast_df[["Month_No"]], forecast_df["Sales"])
    future = np.arange(len(forecast_df)+1, len(forecast_df)+7).reshape(-1,1)
    pred = model.predict(future)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast_df["Month_No"], y=forecast_df["Sales"], name="Actual", line=dict(color="#00F5FF")))
    fig_forecast.add_trace(go.Scatter(x=future.flatten(), y=pred, name="Forecast", line=dict(dash="dash", color="#FF00CC")))
    fig_forecast.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_forecast, use_container_width=True)

with colF2:
    st.subheader("🧠 Business Health")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        title={"text": "Health Score"},
        gauge={"axis": {"range": [0,100]}, "bar": {"color": "#00FFAA"}}
    ))
    gauge.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(gauge, use_container_width=True)

# ====================== PREMIUM AI INSIGHTS (PART 7) ======================
st.markdown("---")
st.subheader("🧠 Advanced AI Business Intelligence Center")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:15px;color:white;"><h3>🚀 Growth Opportunity Score</h3></div>', unsafe_allow_html=True)
    growth_score = min(100, round(((profit_margin + health_score) / 2), 1))
    st.metric("Growth Opportunity", f"{growth_score}%")
    if growth_score >= 80:
        st.success("Excellent growth potential detected!")
    elif growth_score >= 60:
        st.warning("Moderate growth opportunities available.")
    else:
        st.error("Growth strategy optimization recommended.")

with col2:
    st.markdown('<div style="background:linear-gradient(135deg,#11998e,#38ef7d);padding:20px;border-radius:15px;color:white;"><h3>⚠️ Business Risk Meter</h3></div>', unsafe_allow_html=True)
    risk_score = round(100 - growth_score, 1)
    st.metric("Risk Score", f"{risk_score}%")
    if risk_score < 20:
        st.success("Low Risk")
    elif risk_score < 40:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

# Smart Recommendations
st.subheader("🎯 AI Strategic Recommendations")
recs = []
if profit_margin < 15: recs.append("Improve profit margins through pricing & cost optimization.")
if health_score < 75: recs.append("Focus on operational efficiency and cost control.")
if len(recs) == 0: recs.append("Business is performing excellently. Focus on aggressive scaling.")
for r in recs:
    st.info(r)

# PDF Report
st.markdown("---")
if st.button("📥 Download Full Executive PDF Report"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("INSIGHT IQ AI COPILOT - EXECUTIVE REPORT", styles['Title'])]
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y')}", styles['Normal']))
    story.append(Paragraph(f"Total Revenue: ₹{total_sales:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Total Profit: ₹{total_profit:,.0f}", styles['Normal']))
    story.append(Paragraph(f"Profit Margin: {profit_margin:.1f}%", styles['Normal']))
    story.append(Paragraph(f"Top Region: {top_region}", styles['Normal']))
    story.append(Paragraph(f"Health Score: {health_score}/100", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    st.download_button("⬇️ Download PDF", buffer, "InsightIQ_Executive_Report.pdf", "application/pdf")

st.success("🚀 **Ultimate Version** by Khushi Tamre | Final Year AI & BI Masterpiece")
