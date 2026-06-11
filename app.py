import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="INSIGHT IQ • AI COPILOT", page_icon="⚡", layout="wide")

# ====================== DANGEROUS CSS ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #000000 100%);
    }
    .title {
        font-size: 4rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #7C3AED, #00F5FF, #FF00CC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #00F5FF;
        font-size: 1.4rem;
        letter-spacing: 4px;
        margin-top: -15px;
    }
    .glass {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(0, 255, 255, 0.15);
        backdrop-filter: blur(15px);
    }
    .neon { text-shadow: 0 0 20px #00F5FF; }
    .metric-value { font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ====================== DATA ======================
@st.cache_data
def load_data():
    df = pd.read_csv("business_data.csv")
    df.columns = df.columns.str.strip()
    if "Order_Date" in df.columns:
        df = pd.to_datetime(df , errors="coerce")
        df = df .dt.year
        df = df .dt.month
        df = df .dt.to_period("M").astype(str)
    return df

df = load_data()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("# ⚡ INSIGHT IQ")
    st.markdown("### AI Business Copilot")
    st.markdown("---")
    
    years = sorted(df .dropna().unique())
    selected_years = st.multiselect("Select Year", years, default=years)
    
    regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.multiselect("Select Region", regions, default=regions)
    
    categories = sorted(df .dropna().unique())
    selected_categories = st.multiselect("Select Category", categories, default=categories)

# ====================== FILTER ======================
filtered_df = df .isin(selected_years)) &
    (df .isin(selected_regions)) &
    (df .isin(selected_categories))
]

# ====================== KPIs ======================
total_sales = filtered_df .sum()
total_profit = filtered_df .sum()
total_orders = filtered_df["Order_ID"].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
avg_order = total_sales / total_orders if total_orders > 0 else 0

top_region = filtered_df.groupby("Region") .sum().idxmax()
top_category = filtered_df.groupby("Category") .sum().idxmax()
top_product = filtered_df.groupby("Product") .sum().idxmax()

health_score = 95 if profit_margin >= 20 else 85 if profit_margin >= 15 else 72

# ====================== HEADER ======================
st.markdown('<h1 class="title">INSIGHT IQ</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle neon">AI POWERED BUSINESS INTELLIGENCE PLATFORM</p>', unsafe_allow_html=True)
st.markdown("---")

# ====================== KPI SECTION ======================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">TOTAL REVENUE</p><p class="metric-value" style="color:#00F5FF;">₹{total_sales:,.0f}</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">TOTAL PROFIT</p><p class="metric-value" style="color:#00FFAA;">₹{total_profit:,.0f}</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">ORDERS</p><p class="metric-value" style="color:#FF00CC;">{total_orders:,}</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">PROFIT MARGIN</p><p class="metric-value" style="color:#FFD700;">{profit_margin:.1f}%</p></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="glass"><p style="margin:0;color:#aaa;">HEALTH SCORE</p><p class="metric-value" style="color:#00FFAA;">{health_score}/100</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ====================== CHARTS ======================
colA, colB = st.columns([2,1])

with colA:
    monthly = filtered_df.groupby("YearMonth") .sum().reset_index()
    fig1 = px.line(monthly, x="YearMonth", y="Sales", title="📈 Revenue Trend", markers=True,
                   color_discrete_sequence= )
    fig1.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    pie = filtered_df.groupby("Category") .sum().reset_index()
    fig2 = px.pie(pie, names="Category", values="Sales", hole=0.65, title="Revenue by Category")
    fig2.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

colC, colD = st.columns(2)
with colC:
    fig3 = px.bar(filtered_df.groupby("Region") .sum().reset_index(), 
                  x="Region", y="Sales", title="Revenue by Region", color_discrete_sequence= )
    st.plotly_chart(fig3, use_container_width=True)

with colD:
    fig4 = px.scatter(filtered_df.sample(min(800, len(filtered_df))), x="Sales", y="Profit", 
                      color="Category", size="Quantity", title="Sales vs Profit Analysis")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.success("🚀 Project Built by **Khushi Tamre** • Final Year AI & BI Project")
