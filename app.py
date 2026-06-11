import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
from datetime import datetime

# ====================== FUTURISTIC UI ======================
st.set_page_config(page_title="INSIGHT IQ AI COPILOT PRO", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    .main {background: linear-gradient(135deg, #0a0a0a, #1a0033, #000428);}
    .title {
        font-size: 4.5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #7C3AED, #00F5FF, #FF00CC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: glow 2.5s infinite alternate;
    }
    @keyframes glow { from {text-shadow: 0 0 20px #00F5FF;} to {text-shadow: 0 0 50px #FF00CC;} }
    .glass {background: rgba(255,255,255,0.07); border-radius: 24px; padding: 24px; 
            border: 1px solid rgba(0,255,255,0.3); backdrop-filter: blur(20px);}
    .neon {text-shadow: 0 0 30px #00F5FF;}
</style>
""", unsafe_allow_html=True)

# ====================== DATA ======================
@st.cache_data
def load_data():
    df = pd.read_csv("business_data.csv")
    df.columns = df.columns.str.strip()
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df["Year"] = df["Order_Date"].dt.year
        df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    return df

df = load_data()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⚡ INSIGHT IQ")
    st.markdown("### AI Business Intelligence")
    st.markdown("---")
    
    years = sorted(df["Year"].dropna().unique())
    selected_years = st.multiselect("Year", years, default=years)
    
    regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.multiselect("Region", regions, default=regions)
    
    segments = sorted(df["Segment"].dropna().unique())
    selected_segments = st.multiselect("Segment", segments, default=segments)
    
    categories = sorted(df["Category"].dropna().unique())
    selected_categories = st.multiselect("Category", categories, default=categories)

# ====================== FILTER ======================
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

top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()

health_score = 95 if profit_margin >= 20 else 85 if profit_margin >= 15 else 75 if profit_margin >= 10 else 60

# ====================== HEADER ======================
st.markdown('<h1 class="title">INSIGHT IQ</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:1.6rem; color:#00F5FF;" class="neon">AI POWERED • NEXT GEN BUSINESS INTELLIGENCE</p>', unsafe_allow_html=True)
st.markdown("---")

# KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("💰 Revenue", f"₹{total_sales:,.0f}")
with c2: st.metric("📈 Profit", f"₹{total_profit:,.0f}")
with c3: st.metric("🛒 Orders", f"{total_orders:,}")
with c4: st.metric("🎯 Margin", f"{profit_margin:.1f}%")
with c5: st.metric("🧠 Health", f"{health_score}/100")

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Analytics", "🔮 Forecasting", "🧠 AI Insights", "📄 Reports"])

with tab1:
    st.subheader("🎯 KMeans Customer & Product Segmentation")
    if len(filtered_df) > 10:
        features = filtered_df[["Sales", "Profit", "Quantity"]].fillna(0)
        scaled = StandardScaler().fit_transform(features)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clustered_df = filtered_df.copy()
        clustered_df["Cluster"] = kmeans.labels_
        
        fig = px.scatter(clustered_df, x="Sales", y="Profit", color="Cluster",
                        size="Quantity", hover_data=["Product", "Region"],
                        title="Advanced KMeans Clustering", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for clustering")

with tab2:
    colA, colB = st.columns(2)
    with colA:
        monthly = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
        st.plotly_chart(px.line(monthly, x="YearMonth", y="Sales", title="Revenue Trend", markers=True, template="plotly_dark"), use_container_width=True)
    with colB:
        st.plotly_chart(px.pie(filtered_df.groupby("Category")["Sales"].sum().reset_index(), 
                             names="Category", values="Sales", hole=0.6), use_container_width=True)

with tab3:
    st.subheader("🔮 AI Revenue Forecasting")
    forecast_df = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
    forecast_df["Month_No"] = range(1, len(forecast_df)+1)
    model = LinearRegression().fit(forecast_df[["Month_No"]], forecast_df["Sales"])
    future = np.arange(len(forecast_df)+1, len(forecast_df)+7).reshape(-1,1)
    pred = model.predict(future)
    
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=forecast_df["Month_No"], y=forecast_df["Sales"], name="Actual"))
    fig_f.add_trace(go.Scatter(x=future.flatten(), y=pred, name="Forecast", line=dict(dash="dash")))
    st.plotly_chart(fig_f, use_container_width=True)

with tab4:
    st.subheader("🧠 Advanced AI Insights")
    growth_score = min(100, round((profit_margin + health_score)/2, 1))
    colX, colY = st.columns(2)
    with colX: st.metric("🚀 Growth Score", f"{growth_score}%")
    with colY: st.metric("⚠️ Risk Score", f"{100 - growth_score}%")
    st.info(f"**Top Recommendation**: Scale in **{top_region}** and focus on **{top_category}**")

with tab5:
    st.subheader("📄 Executive Reports")
    if st.button("📥 Generate Executive PDF"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("INSIGHT IQ AI COPILOT - EXECUTIVE REPORT", styles['Title'])]
        story.append(Spacer(1,20))
        story.append(Paragraph(f"Revenue: ₹{total_sales:,.0f}", styles['Normal']))
        story.append(Paragraph(f"Profit: ₹{total_profit:,.0f}", styles['Normal']))
        story.append(Paragraph(f"Health Score: {health_score}/100", styles['Normal']))
        story.append(Paragraph(f"Top Region: {top_region}", styles['Normal']))
        doc.build(story)
        buffer.seek(0)
        st.download_button("⬇️ Download PDF", buffer, "InsightIQ_Report.pdf", "application/pdf")

st.success("🚀 **Dangerous Premium Version** by Khushi Tamre")
st.balloons()
