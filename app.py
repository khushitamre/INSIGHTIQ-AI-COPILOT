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

st.set_page_config(
    page_title="INSIGHT IQ AI COPILOT PRO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);}
    .stMetric {background: rgba(255,255,255,0.1); border-radius: 15px; padding: 10px;}
    h1 {font-family: 'Arial'; color: #7C3AED; text-shadow: 0 0 20px #7C3AED;}
    .glass {background: rgba(255,255,255,0.08); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px;}
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
    st.title("⚙️ Control Center")
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
    st.success("🚀 AI System Online")

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
avg_order = total_sales / total_orders if total_orders > 0 else 0

top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()

health_score = 95 if profit_margin >= 20 else 85 if profit_margin >= 15 else 75 if profit_margin >= 10 else 60

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Executive Dashboard", "📊 Analytics", "🔮 AI Forecast", "🧠 Insights", "📄 Reports"])

with tab1:
    st.markdown("# 🚀 INSIGHT IQ AI COPILOT PRO")
    st.markdown("### *Next-Gen Business Intelligence Platform*")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("💰 Revenue", f"₹{total_sales:,.0f}")
    with col2: st.metric("📈 Profit", f"₹{total_profit:,.0f}")
    with col3: st.metric("🛒 Orders", f"{total_orders:,}")
    with col4: st.metric("🎯 Margin", f"{profit_margin:.1f}%")
    with col5: st.metric("💎 Avg Order", f"₹{avg_order:,.0f}")

    st.markdown("---")
    colA, colB, colC = st.columns(3)
    with colA: st.metric("🌎 Top Region", top_region)
    with colB: st.metric("📦 Top Category", top_category)
    with colC: st.metric("🧠 Health Score", f"{health_score}/100")

with tab2:
    # All your charts here (I kept the best ones)
    st.plotly_chart(px.line(filtered_df.groupby("YearMonth")["Sales"].sum().reset_index(), 
                           x="YearMonth", y="Sales", title="Revenue Trend"), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(filtered_df.groupby("Category")["Sales"].sum().reset_index(), 
                              names="Category", values="Sales", hole=0.6), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(filtered_df.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=True), 
                              x="Sales", y="Region", orientation='h'), use_container_width=True)

with tab3:
    st.subheader("🔮 AI Revenue Forecasting")
    # Linear Regression Forecast (kept + improved)
    forecast_df = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
    forecast_df["Month_No"] = range(1, len(forecast_df)+1)
    model = LinearRegression().fit(forecast_df[["Month_No"]], forecast_df["Sales"])
    
    future = np.arange(len(forecast_df)+1, len(forecast_df)+7).reshape(-1,1)
    pred = model.predict(future)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["Month_No"], y=forecast_df["Sales"], name="Actual"))
    fig.add_trace(go.Scatter(x=future.flatten(), y=pred, name="Forecast", line=dict(dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("🧠 Advanced AI Insights")
    # KMeans Clustering
    if len(filtered_df) > 10:
        features = filtered_df[["Sales", "Profit", "Quantity"]].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
        filtered_df["Cluster"] = kmeans.labels_
        
        st.plotly_chart(px.scatter(filtered_df, x="Sales", y="Profit", color="Cluster", 
                                  hover_data=["Product"], title="Customer/Product Segmentation"), use_container_width=True)

with tab5:
    st.subheader("📄 Generate Executive Report")
    if st.button("📥 Download PDF Report"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("INSIGHT IQ AI COPILOT - Executive Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y')}", styles['Normal']))
        # Add more content...
        doc.build(story)
        buffer.seek(0)
        st.download_button("Download Report", buffer, "Executive_Report.pdf", "application/pdf")

st.sidebar.success("Made with ❤️ by Khushi Tamre")
