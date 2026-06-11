import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from prophet import Prophet
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="INSIGHT IQ QUANTUM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM THEME - DANGEROUS BEAUTIFUL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: #000000;
    background-image:
        radial-gradient(circle at 10% 20%, rgba(255,0,128,0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(0,255,255,0.15) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(120,40,255,0.1) 0%, transparent 50%);
    animation: bgPulse 8s ease-in-out infinite;
}

@keyframes bgPulse {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.2); }
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1600px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0d1a 0%, #0d0f1f 100%);
    border-right: 1px solid rgba(255,0,128,0.2);
}
[data-testid="stSidebar"].stMarkdown h3 {
    color: #ff0080!important;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.5rem;
}

[data-testid="stMultiSelect"] > div {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,0,128,0.3);
    border-radius: 10px;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,0,128,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem!important;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s;
}
[data-testid="stMetric"]:hover { border-color: rgba(255,0,128,0.6); }
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #ff0080, #00ffff, #7828ff);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8!important;
    font-size: 0.78rem!important;
    font-weight: 500!important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9!important;
    font-size: 1.8rem!important;
    font-weight: 800!important;
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04)!important;
    border: 1px solid rgba(255,0,128,0.3)!important;
    border-radius: 10px!important;
    color: #f1f5f9!important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #ff0080!important;
    box-shadow: 0 0 0 3px rgba(255,0,128,0.15)!important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff0080, #7828ff)!important;
    color: white!important;
    border: none!important;
    border-radius: 10px!important;
    font-weight: 700!important;
    font-size: 0.88rem!important;
    letter-spacing: 0.04em;
    padding: 0.6rem 1.8rem!important;
    transition: all 0.25s!important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff1a8c, #8b3eff)!important;
    transform: translateY(-1px)!important;
    box-shadow: 0 8px 24px rgba(255,0,128,0.35)!important;
}

.stSuccess,.stInfo,.stWarning,.stError {
    border-radius: 12px!important;
    border: none!important;
    font-weight: 500!important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,0,128,0.2);
    border-radius: 12px;
    overflow: hidden;
}

hr { border-color: rgba(255,0,128,0.15)!important; }
h1, h2, h3, h4 { color: #f1f5f9!important; }
h2 { font-size: 1.3rem!important; font-weight: 700!important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,0,128,0.4); border-radius: 3px; }

.iq-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,0,128,0.25);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
}
.iq-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #ff0080, #00ffff, #7828ff);
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #ff0080;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,0,128,0.25);
}
.ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,0,128,0.15);
    border: 1px solid rgba(255,0,128,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: #ff6b9d;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pulse {
    width: 6px; height: 6px;
    background: #00ffff;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,255,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,255,255,0); }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("business_data.csv")
        st.success("✅ Loaded business_data.csv")
    except FileNotFoundError:
        st.info("📝 Using sample data (20 records)")
        # FIXED: All arrays exactly 20 length
        df = pd.DataFrame({
            'Order_ID': list(range(1001, 1021)),
            'Order_Date': pd.date_range('2024-01-01', periods=20, freq='2D'),
            'Region': ['North', 'South', 'East', 'West', 'North'] * 4,
            'Category': ['Technology', 'Furniture', 'Office Supplies', 'Technology'] * 5,
            'Product': ['Laptop', 'Chair', 'Printer', 'Monitor', 'Table'] * 4,
            'Customer_Name': [f'Customer{i}' for i in range(20)],
            'Sales': np.random.randint(2000, 60000, 20),
            'Profit': np.random.randint(500, 15000, 20),
            'Quantity': np.random.randint(1, 10, 20),
            'Segment': ['Consumer', 'Corporate', 'Home Office', 'Consumer', 'Corporate'] * 4
        })

    df.columns = df.columns.str.strip()
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df["Year"] = df["Order_Date"].dt.year
        df["Month"] = df["Order_Date"].dt.month
        df["Quarter"] = "Q" + df["Order_Date"].dt.quarter.astype(str)
        df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
        df["Week"] = df["Order_Date"].dt.isocalendar().week
        df["Day"] = df["Order_Date"].dt.day
        df["DayOfWeek"] = df["Order_Date"].dt.dayofweek
    df["Discount"] = (1 - (df["Profit"] / df["Sales"])).clip(0, 0.5)
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#ff0080,#00ffff);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">⚡</div>
            <span style="font-size:1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.02em;">INSIGHT IQ</span>
        </div>
        <div style="font-size:0.68rem;color:#64748b;font-weight:500;letter-spacing:0.1em;
                    text-transform:uppercase;padding-left:2px;">Quantum Edition</div>
    </div>
    <hr style="border-color:rgba(255,0,128,0.2);margin:0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    st.markdown("<h3>📅 Time Period</h3>", unsafe_allow_html=True)
    years = sorted(df["Year"].dropna().unique())
    selected_years = st.multiselect("Year", years, default=years, label_visibility="collapsed")

    st.markdown("<h3>🌎 Geography</h3>", unsafe_allow_html=True)
    regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.multiselect("Region", regions, default=regions, label_visibility="collapsed")

    st.markdown("<h3>👥 Segment</h3>", unsafe_allow_html=True)
    segments = sorted(df["Segment"].dropna().unique())
    selected_segments = st.multiselect("Segment", segments, default=segments, label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(255,0,128,0.2);margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(255,0,128,0.08);border:1px solid rgba(255,0,128,0.2);
                border-radius:10px;padding:0.9rem;margin-top:0.5rem;">
        <div style="font-size:0.68rem;font-weight:700;color:#ff6b9d;letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.5rem;">Developer</div>
        <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">Khushi Tamre</div>
        <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem;">AI &amp; BI Engineer</div>
        <div style="margin-top:0.5rem;">
            <span style="background:rgba(0,255,255,0.1);border:1px solid rgba(0,255,255,0.3);
                         border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;
                         font-weight:600;color:#00ffff;margin:0.15rem;">Python</span>
            <span style="background:rgba(255,0,128,0.1);border:1px solid rgba(255,0,128,0.3);
                         border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;
                         font-weight:600;color:#ff6b9d;margin:0.15rem;">XGBoost</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────
filtered_df = df[
    df["Year"].isin(selected_years) &
    df["Region"].isin(selected_regions) &
    df["Segment"].isin(selected_segments)
]

# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE HEALTH & GROWTH SCORES
# ─────────────────────────────────────────────────────────────────────────────
def calculate_health_score(df):
    profit_margin = (df['Profit'].sum() / df['Sales'].sum() * 100) if df['Sales'].sum() > 0 else 0
    order_freq = len(df) / df['Order_Date'].nunique() if df['Order_Date'].nunique() > 0 else 0
    avg_order = df['Sales'].mean()
    health = (profit_margin * 0.4) + (min(order_freq * 10, 30)) + (min(avg_order / 1000, 30))
    return round(min(health, 100), 1)

def calculate_growth_score(df):
    if len(df) < 10:
        return 50.0
    df_sorted = df.sort_values('Order_Date')
    mid = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid]['Sales'].sum()
    second_half = df_sorted.iloc[mid:]['Sales'].sum()
    growth = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
    return round(max(min(growth + 50, 100), 0), 1)

health_score = calculate_health_score(filtered_df)
growth_score = calculate_growth_score(filtered_df)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order_ID"].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
avg_order_val = total_sales / total_orders if total_orders > 0 else 0

top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax()
region_share = (filtered_df.groupby("Region")["Sales"].sum().max() / total_sales * 100) if total_sales > 0 else 0
risk_level = "Low" if health_score >= 90 else "Medium" if health_score >= 75 else "High"
confidence = "95%" if health_score >= 90 else "85%" if health_score >= 75 else "70%"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:2rem;flex-wrap:wrap;gap:1rem;">
    <div>
        <div class="ai-badge"><span class="pulse"></span> Live Intelligence Active</div>
        <h1 style="
            font-size:clamp(2rem,4vw,3rem);
            font-weight:900;
            letter-spacing:-0.04em;
            margin:0.5rem 0 0.3rem;
            background:linear-gradient(90deg,#f1f5f9 30%,#ff0080 65%,#00ffff 100%);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            line-height:1.1;
        ">INSIGHT IQ QUANTUM</h1>
        <p style="font-size:0.95rem;color:#64748b;font-weight:400;margin:0;letter-spacing:0.01em;">
            Enterprise Intelligence Platform &nbsp;·&nbsp; Predictive Analytics &nbsp;·&nbsp; Executive Decisions
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">✦ &nbsp;Core Metrics</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Total Revenue", f"₹{total_sales:,.0f}")
c2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
c3.metric("🛒 Orders", f"{total_orders:,}")
c4.metric("🎯 Profit Margin", f"{profit_margin:.1f}%")
c5.metric("💎 Avg. Order Value", f"₹{avg_order_val:,.0f}")
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Revenue Trend + Category Donut
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📈 &nbsp;Revenue Analysis</div>', unsafe_allow_html=True)
col_left, col_right = st.columns([2, 1])

with col_left:
    monthly = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index().rename(columns={"Sales": "Revenue"})
    fig_trend = px.area(
        monthly, x="YearMonth", y="Revenue",
        title="Monthly Revenue Trend",
        color_discrete_sequence=["#ff0080"]
    )
    fig_trend.update_traces(
        line=dict(width=2.5, color="#ff0080"),
        fillcolor="rgba(255,0,128,0.12)",
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>"
    )
    fig_trend.update_layout(title=dict(font=dict(color="#94a3b8", size=13), x=0))
    st.plotly_chart(apply_theme(fig_trend, 300), use_container_width=True, key="trend")

with col_right:
    cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    fig_donut = px.pie(
        cat_sales, names="Category", values="Sales",
        hole=0.65, color_discrete_sequence=["#ff0080", "#00ffff", "#7828ff", "#38bdf8", "#fb923c"],
        title="Revenue by Category"
    )
    fig_donut.update_traces(
        textfont=dict(color="white", size=12),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>"
    )
    fig_donut.update_layout(
        title=dict(font=dict(color="#94a3b8", size=13), x=0),
        legend=dict(orientation="h", y=-0.12, font=dict(color="#94a3b8", size=11))
    )
    st.plotly_chart(apply_theme(fig_donut, 300), use_container_width=True, key="donut")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Region Bar + Scatter
# ─────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-label">🌎 &nbsp;Regional Performance</div>', unsafe_allow_html=True)
    reg = filtered_df.groupby("Region").agg(Revenue=("Sales","sum"), Profit=("Profit","sum")).reset_index().sort_values("Revenue", ascending=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=reg["Region"], x=reg["Revenue"],
        orientation="h", name="Revenue",
        marker=dict(
            color=reg["Revenue"],
            colorscale=[[0,"#3b0764"],[0.5,"#ff0080"],[1,"#ff6b9d"]],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>"
    ))
    fig_bar.update_layout(title=dict(text="Revenue by Region", font=dict(color="#94a3b8", size=13), x=0), showlegend=False)
    st.plotly_chart(apply_theme(fig_bar, 280), use_container_width=True, key="region_bar")

with col_b:
    st.markdown('<div class="section-label">🎯 &nbsp;Profitability Intelligence Map</div>', unsafe_allow_html=True)
    sample = filtered_df.sample(min(1200, len(filtered_df)))
    fig_scatter = px.scatter(
        sample, x="Sales", y="Profit",
        color="Category", size="Quantity",
        color_discrete_sequence=["#ff0080", "#00ffff", "#7828ff", "#38bdf8", "#fb923c"],
        hover_data=["Product"],
        title="Sales vs Profit"
    )
    fig_scatter.update_traces(
        marker=dict(opacity=0.7, line=dict(width=0)),
        hovertemplate="<b>%{customdata[0]}</b><br>Sales: ₹%{x:,.0f}<br>Profit: ₹%{y:,.0f}<extra></extra>"
    )
    fig_scatter.update_layout(title=dict(font=dict(color="#94a3b8", size=13), x=0))
    st.plotly_chart(apply_theme(fig_scatter, 280), use_container_width=True, key="scatter")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — Quarter Bar + Sub-Category Bar
# ─────────────────────────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<div class="section-label">📊 &nbsp;Quarterly Revenue</div>', unsafe_allow_html=True)
    qtr = filtered_df.groupby(["Year","Quarter"])["Sales"].sum().reset_index().sort_values(["Year","Quarter"])
    qtr["Label"] = qtr["Year"].astype(str) + " " + qtr["Quarter"]
    fig_qtr = px.bar(
        qtr, x="Label", y="Sales",
        color="Quarter", color_discrete_sequence=["#ff0080", "#00ffff", "#7828ff", "#38bdf8"],
        title="Revenue by Quarter",
        barmode="group"
    )
    fig_qtr.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>"
    )
    fig_qtr.update_layout(title=dict(font=dict(color="#94a3b8", size=13), x=0), showlegend=True)
    st.plotly_chart(apply_theme(fig_qtr, 280), use_container_width=True, key="quarterly")

with col_d:
    st.markdown('<div class="section-label">📦 &nbsp;Sub-Category Performance</div>', unsafe_allow_html=True)
    sub_col = "Sub-Category" if "Sub-Category" in filtered_df.columns else "Category"
    sub = filtered_df.groupby(sub_col)["Profit"].sum().reset_index().sort_values("Profit", ascending=True).tail(10)
    sub["Color"] = sub["Profit"].apply(lambda v: "#00ffff" if v >= 0 else "#f87171")
    fig_sub = go.Figure()
    fig_sub.add_trace(go.Bar(
        y=sub[sub_col], x=sub["Profit"],
        orientation="h",
        marker=dict(color=sub["Color"], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Profit: ₹%{x:,.0f}<extra></extra>"
    ))
    fig_sub.update_layout(
        title=dict(text="Profit by Sub-Category", font=dict(color="#94a3b8", size=13), x=0),
        showlegend=False
    )
    st.plotly_chart(apply_theme(fig_sub, 280), use_container_width=True, key="subcategory")

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST MODEL (NO RANDOM FOREST)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🚀 &nbsp;XGBoost Gradient Boosting Engine</div>', unsafe_allow_html=True)

if len(filtered_df) >= 15:
    features = ['Quantity', 'Discount', 'Month', 'DayOfWeek', 'Year', 'Week', 'Day']
    X = filtered_df[features].fillna(0)
    y = filtered_df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # XGBOOST MODEL
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="iq-card">
            <div style="display:inline-flex;align-items:center;gap:0.5rem;background:linear-gradient(135deg,#ff6b35,#f7931e);
                        border-radius:20px;padding:0.5rem 1.2rem;font-size:0.75rem;font-weight:700;
                        color:white;box-shadow:0 4px 20px rgba(255,107,53,0.4);margin-bottom:1rem;">
                🔥 XGBOOST EXTREME
            </div>
            <h3 style="color:#ff6b35;margin:0 0 1rem;font-size:1.3rem;">Gradient Boosting Model</h3>
            <div style="margin:1.5rem 0;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;">R² SCORE</div>
                <div style="font-size:3rem;font-weight:900;color:#ff6b35;">{r2*100:.2f}%</div>
            </div>
            <div style="margin:1.5rem 0;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;">BOOSTING ROUNDS</div>
                <div style="font-size:2rem;font-weight:700;color:#f1f5f9;">200</div>
            </div>
            <div style="margin:1.5rem 0;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;">MEAN ERROR</div>
                <div style="font-size:2rem;font-weight:700;color:#f1f5f9;">₹{mae:,.0f}</div>
            </div>
            <div style="margin:1.5rem 0;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;">RMSE</div>
                <div style="font-size:2rem;font-weight:700;color:#f1f5f9;">₹{rmse:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fi_df = pd.DataFrame({
            'Feature': features,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=fi_df['Feature'],
            x=fi_df['Importance'],
            orientation='h',
            marker=dict(
                color=fi_df['Importance'],
                colorscale=[[0, '#0a0a0f'], [0.5, '#ff6b35'], [1, '#f7931e']],
                line=dict(width=0)
            )
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#94a3b8', family='Space Grotesk'),
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
            title=dict(text='XGBoost Feature Importance', font=dict(size=16, color='#94a3b8')),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig, use_container_width=True, key="xgb_importance")

# ─────────────────────────────────────────────────────────────────────────────
# PROPHET FORECAST
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🧠 &nbsp;Prophet Neural Forecast Engine</div>', unsafe_allow_html=True)

if len(filtered_df) >= 10:
    ts_data = filtered_df.groupby('Order_Date')['Sales'].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(ts_data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            name='Neural Forecast',
            line=dict(color='#00ffff', width=3),
            fill='tozeroy',
            fillcolor='rgba(0,255,255,0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            name='Upper Bound',
            line=dict(color='rgba(0,255,255,0.3)', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            name='Lower Bound',
            line=dict(color='rgba(0,255,255,0.3)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0,255,255,0.05)'
        ))
        fig.add_trace(go.Scatter(
            x=ts_data['ds'], y=ts_data['y'],
            name='Actual',
            mode='markers',
            marker=dict(color='#ff0080', size=8, line=dict(width=2, color='#ffffff'))
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#94a3b8', family='Space Grotesk'),
            height=450,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, title='Revenue (₹)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True, key="prophet_forecast")

    with col2:
        next_30 = forecast.tail(30)['yhat'].sum()
        st.markdown(f"""
        <div class="iq-card" style="height:450px;display:flex;flex-direction:column;justify-content:center;">
            <div style="display:inline-flex;align-items:center;gap:0.5rem;background:linear-gradient(135deg,#00ffff,#0080ff);
                        border-radius:20px;padding:0.5rem 1.2rem;font-size:0.75rem;font-weight:700;
                        color:#000;box-shadow:0 4px 20px rgba(0,255,255,0.4);margin-bottom:1rem;">
                🧠 PROPHET FORECAST
            </div>
            <div style="margin:2rem 0;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;">30-DAY PREDICTION</div>
                <div style="font-size:2.5rem;font-weight:900;color:#00ffff;">₹{next_30:,.0f}</div>
            </div>
            <div style="margin:1.5rem 0;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;">MODEL</div>
                <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">Prophet Neural</div>
            </div>
            <div style="margin:1.5rem 0;">
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;">SEASONALITY</div>
                <div style="font
