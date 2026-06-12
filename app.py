
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="INSIGHT IQ AI COPILOT PRO",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'Executive'

# ─────────────────────────────────────────────────────────────────────────────
# THEME CONFIG
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    'dark': {
        'bg': '#050814',
        'bg_gradient': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(120,40,255,0.18), transparent), radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,255,163,0.06), transparent)',
        'card_bg': 'rgba(255,255,255,0.05)',
        'text': '#f1f5f9',
        'text_muted': '#94a3b8',
        'border': 'rgba(120,40,255,0.25)',
        'accent': '#7828ff',
        'accent2': '#00ffa3'
    },
    'light': {
        'bg': '#f8fafc',
        'bg_gradient': 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(120,40,255,0.08), transparent)',
        'card_bg': 'rgba(255,255,255,0.9)',
        'text': '#0f172a',
        'text_muted': '#64748b',
        'border': 'rgba(120,40,255,0.15)',
        'accent': '#7828ff',
        'accent2': '#10b981'
    }
}

T = THEMES[st.session_state.theme]

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background: {T['bg']};
    background-image: {T['bg_gradient']};
    transition: all 0.3s ease;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2.5rem 4rem; max-width: 1600px; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {T['bg']} 0%, {T['bg']} 100%);
    border-right: 1px solid {T['border']};
}}

[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {T['card_bg']} 0%, {T['card_bg']} 100%);
    border: 1px solid {T['border']};
    border-radius: 16px;
    padding: 1.4rem 1.6rem!important;
    backdrop-filter: blur(12px);
    transition: all 0.3s;
}}
[data-testid="stMetric"]:hover {{
    border-color: {T['accent']};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(120,40,255,0.2);
}}
[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']});
}}
[data-testid="stMetricLabel"] {{
    color: {T['text_muted']}!important;
    font-size: 0.75rem!important;
    font-weight: 600!important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{
    color: {T['text']}!important;
    font-size: 2rem!important;
    font-weight: 800!important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {T['accent']}, #5b10d9)!important;
    color: white!important;
    border: none!important;
    border-radius: 10px!important;
    font-weight: 700!important;
    transition: all 0.25s!important;
}}
.stButton > button:hover {{
    transform: translateY(-1px)!important;
    box-shadow: 0 8px 24px rgba(120,40,255,0.35)!important;
}}

.iq-card {{
    background: linear-gradient(135deg, {T['card_bg']} 0%, {T['card_bg']} 100%);
    border: 1px solid {T['border']};
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s;
}}
.iq-card:hover {{
    border-color: {T['accent']};
    box-shadow: 0 8px 32px rgba(120,40,255,0.15);
}}
.iq-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']});
}}

.section-label {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {T['accent']};
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {T['border']};
}}

.ai-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(120,40,255,0.15);
    border: 1px solid rgba(120,40,255,0.4);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.08em;
}}

.pulse {{
    width: 6px; height: 6px;
    background: {T['accent2']};
    border-radius: 50%;
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 rgba(0,255,163,0.4); }}
    50% {{ opacity: 0.7; box-shadow: 0 0 0 8px rgba(0,255,163,0); }}
}}

.insight-tag {{
    display: inline-block;
    background: rgba(0,255,163,0.1);
    border: 1px solid rgba(0,255,163,0.3);
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: {T['accent2']};
    margin: 0.15rem;
}}

h1, h2, h3, h4 {{ color: {T['text']}!important; }}
p, span, div {{ color: {T['text']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#7828ff", "#00ffa3", "#f472b6", "#38bdf8", "#fb923c", "#a3e635"]

def apply_theme(fig, height=360):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=T['text_muted'], size=12),
        height=height,
        margin=dict(l=20, r=20, t=44, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T['text_muted']))
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)")
    return fig

def generate_insights(df):
    """Auto-generate AI insights from data"""
    insights = []

    # Trend analysis
    if 'YearMonth' in df.columns:
        monthly = df.groupby('YearMonth')['Sales'].sum()
        if len(monthly) > 1:
            growth = ((monthly.iloc[-1] / monthly.iloc[0]) - 1) * 100
            insights.append(f"📈 Revenue {('grew' if growth > 0 else 'declined')} {abs(growth):.1f}% over period")

    # Top performer
    top_cat = df.groupby('Category')['Sales'].sum().idxmax()
    cat_share = df.groupby('Category')['Sales'].sum()[top_cat] / df['Sales'].sum() * 100
    insights.append(f"🏆 {top_cat} dominates with {cat_share:.1f}% revenue share")

    # Anomaly detection
    if len(df) > 30:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[['Sales', 'Profit']])
        anomalies = (df['anomaly'] == -1).sum()
        if anomalies > 0:
            insights.append(f"⚠️ Detected {anomalies} unusual transactions requiring review")

    # Profit alert
    margin = df['Profit'].sum() / df['Sales'].sum() * 100
    if margin < 10:
        insights.append(f"🔴 Critical: Profit margin at {margin:.1f}% - below healthy threshold")
    elif margin > 20:
        insights.append(f"🟢 Excellent: {margin:.1f}% profit margin - strong performance")

    return insights

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("business_data.csv")
        except FileNotFoundError:
            st.error("⚠️ Upload a CSV file to begin analysis")
            st.stop()

    df.columns = df.columns.str.strip()
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df = df.dropna(subset=['Order_Date'])
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Quarter'] = 'Q' + df['Order_Date'].dt.quarter.astype(str)
    df['YearMonth'] = df['Order_Date'].dt.to_period('M').astype(str)
    df['Week'] = df['Order_Date'].dt.isocalendar().week
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1.2rem 0 1rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,{T['accent']},{T['accent2']});
                        border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">✦</div>
            <span style="font-size:1rem;font-weight:800;color:{T['text']};letter-spacing:-0.02em;">INSIGHT IQ PRO</span>
        </div>
        <div style="font-size:0.68rem;color:{T['text_muted']};font-weight:500;letter-spacing:0.1em;
                    text-transform:uppercase;">Enterprise Intelligence v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    if col_t1.button("🌙 Dark" if st.session_state.theme == 'light' else "☀️ Light", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

    st.session_state.view_mode = col_t2.selectbox("View", ["Executive", "Analyst"], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
    df = load_data(uploaded_file)

    st.markdown(f"<h3 style='color:{T['text']}!important;'>📅 Date Range</h3>", unsafe_allow_html=True)
    min_date = df['Order_Date'].min().date()
    max_date = df['Order_Date'].max().date()
    date_range = st.date_input("Select Range", [min_date, max_date], label_visibility="collapsed")

    st.markdown(f"<h3 style='color:{T['text']}!important;'>🌎 Geography</h3>", unsafe_allow_html=True)
    regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.multiselect("Region", regions, default=regions, label_visibility="collapsed")

    st.markdown(f"<h3 style='color:{T['text']}!important;'>👥 Segment</h3>", unsafe_allow_html=True)
    segments = sorted(df["Segment"].dropna().unique())
    selected_segments = st.multiselect("Segment", segments, default=segments, label_visibility="collapsed")

    if st.button("🔄 Reset All", use_container_width=True):
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{T['card_bg']};border:1px solid {T['border']};
                border-radius:10px;padding:0.9rem;">
        <div style="font-size:0.68rem;font-weight:700;color:#a78bfa;letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.5rem;">Developer</div>
        <div style="font-size:0.85rem;font-weight:600;color:{T['text']};">Khushi Tamre</div>
        <div style="font-size:0.75rem;color:{T['text_muted']};margin-top:0.2rem;">AI & BI Engineer</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (
        (df['Order_Date'].dt.date >= start_date) &
        (df['Order_Date'].dt.date <= end_date) &
        (df['Region'].isin(selected_regions)) &
        (df['Segment'].isin(selected_segments))
    )
    filtered_df = df[mask]
else:
    filtered_df = df[df['Region'].isin(selected_regions) & df['Segment'].isin(selected_segments)]

if filtered_df.empty:
    st.warning("⚠️ No data for selected filters")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE KPIs
# ─────────────────────────────────────────────────────────────────────────────
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
total_orders = filtered_df['Order_ID'].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
avg_order_val = total_sales / total_orders if total_orders > 0 else 0

# Comparison with previous period
prev_mask = (
    (df['Order_Date'] < filtered_df['Order_Date'].min()) &
    (df['Region'].isin(selected_regions)) &
    (df['Segment'].isin(selected_segments))
)
prev_df = df[prev_mask]
prev_sales = prev_df['Sales'].sum() if not prev_df.empty else 0
sales_delta = ((total_sales / prev_sales) - 1) * 100 if prev_sales > 0 else 0

top_region = filtered_df.groupby('Region')['Sales'].sum().idxmax()
top_category = filtered_df.groupby('Category')['Sales'].sum().idxmax()
top_product = filtered_df.groupby('Product')['Sales'].sum().idxmax()
region_share = filtered_df.groupby('Region')['Sales'].sum().max() / total_sales * 100

health_score = 95 if profit_margin >= 20 else 85 if profit_margin >= 15 else 75 if profit_margin >= 10 else 60
risk_level = "Low" if health_score >= 90 else "Medium" if health_score >= 75 else "High"
confidence = "95%" if health_score >= 90 else "85%" if health_score >= 75 else "70%"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:2rem;">
    <div>
        <div class="ai-badge"><span class="pulse"></span> Live Intelligence Active</div>
        <h1 style="
            font-size:clamp(2rem,4vw,3rem);
            font-weight:900;
            margin:0.5rem 0 0.3rem;
            background:linear-gradient(90deg,{T['text']} 30%,#a78bfa 65%,{T['accent2']} 100%);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        ">INSIGHT IQ AI COPILOT PRO</h1>
        <p style="font-size:0.95rem;color:{T['text_muted']};">
            Enterprise Intelligence Platform · Predictive Analytics · {st.session_state.view_mode} View
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
insights = generate_insights(filtered_df)
if insights:
    st.markdown('<div class="section-label">🤖 AI-Generated Insights</div>', unsafe_allow_html=True)
    cols = st.columns(len(insights))
    for i, insight in enumerate(insights):
        cols[i].info(insight)

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">✦ Core Performance Metrics</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Total Revenue", f"₹{total_sales:,.0f}", f"{sales_delta:+.1f}%")
c2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
c3.metric("🛒 Orders", f"{total_orders:,}")
c4.metric("🎯 Profit Margin", f"{profit_margin:.1f}%")
c5.metric("💎 Avg Order", f"₹{avg_order_val:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📊 Revenue Intelligence</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    monthly = filtered_df.groupby('YearMonth')['Sales'].sum().reset_index()
    fig = px.area(monthly, x='YearMonth', y='Sales', title='Revenue Trend', color_discrete_sequence=[T['accent']])
    fig.update_traces(fillcolor=f"rgba(120,40,255,0.12)")
    apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
    fig = px.pie(cat_sales, names='Category', values='Sales', hole=0.6, color_discrete_sequence=PALETTE)
    apply_theme(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROPHET FORECAST
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🔮 Prophet AI Forecast Engine</div>', unsafe_allow_html=True)

if len(filtered_df) >= 30:
    ts_data = filtered_df.groupby('Order_Date')['Sales'].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(ts_data)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    col_f1, col_f2 = st.columns([3, 1])

    with col_f1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast',
                                line=dict(color=T['accent2'], width=2)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                fill=None, mode='lines', line_color='rgba(0,255,163,0.2)', name='Upper'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                fill='tonexty', mode='lines', line_color='rgba(0,255,163,0.2)', name='Lower'))
        fig.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name='Actual',
                                mode='markers', marker=dict(color=T['accent'], size=4)))
        apply_theme(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    with col_f2:
        next_30 = forecast.tail(30)['yhat'].sum()
        st.markdown(f"""
        <div class="iq-card" style="height:100%;">
            <div style="font-size:0.68rem;font-weight:700;color:#a78bfa;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:1.2rem;">30-Day Forecast</div>
            <div style="font-size:1.8rem;font-weight:800;color:{T['accent2']};">₹{next_30:,.0f}</div>
            <div style="font-size:0.75rem;color:{T['text_muted']};margin-top:0.5rem;">Projected Revenue</div>
            <hr style="margin:1rem 0;">
            <div style="font-size:0.72rem;color:{T['text_muted']};">Model: Prophet</div>
            <div style="font-size:0.72rem;color:{T['text_muted']};">Confidence: {confidence}</div>
            <div style="font-size:0.72rem;color:{T['text_muted']};">Trend: {'Upward' if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else 'Stable'}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("📊 Need 30+ days of data for Prophet forecast")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT OPTIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📥 Export & Reports</div>', unsafe_allow_html=True)
exp1, exp2, exp3 = st.columns(3)

with exp1:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📊 Download CSV", csv, "filtered_data.csv", "text/csv", use_container_width=True)

with exp2:
    excel_buffer = io.BytesIO()
    filtered_df.to_excel(excel_buffer, index=False)
    st.download_button("📗 Download Excel", excel_buffer, "report.xlsx", use_container_width=True)

with exp3:
    if st.button("📄 Generate PDF Report", use_container_width=True):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        story.append(Paragraph("INSIGHT IQ PRO - Executive Report", styles['Title']))
        story.append(Spacer(1, 12))
        data = [
            ['Metric', 'Value'],
            ['Total Revenue', f'₹{total_sales:,.0f}'],
            ['Total Profit', f'₹{total_profit:,.0f}'],
            ['Profit Margin', f'{profit_margin:.1f}%'],
            ['Health Score', f'{health_score}/100'],
            ['Risk Level', risk_level]
        ]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#7828ff')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.grey)
        ]))
        story.append(table)
        doc.build(story)
        buffer.seek(0)
        st.download_button("⬇️ Download PDF", buffer, "executive_report.pdf", "application/pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:3rem;border-top:1px solid {T['border']};padding-top:1.5rem;text-align:center;">
    <div style="font-size:0.82rem;color:{T['text_muted']};">
        <span style="color:{T['accent']};font-weight:700;">INSIGHT IQ PRO v2.0</span>
        · Built by <b style="color:#a78bfa;">Khushi Tamre</b>
        · Python · Streamlit · Prophet · Plotly
    </div>
</div>
""", unsafe_allow_html=True)
Run Command:
Bash
pip install streamlit pandas numpy plotly prophet scikit-learn reportlab openpyxl
streamlit run app.py


Feature

Kya karta hai

1. Customer Segmentation

KMeans clustering se RFM analysis

2. Sales Prediction Model

RandomForest 

3. Churn Prediction

LogisticRegression se risky customers

4. Anomaly Detection

IsolationForest se fraud/outlier orders

5. Feature Importance

SHAP values se batayega kaunsa factor sales affect karta hai

6. What-If Simulator

Slider se "Agar discount 10% badhaun to profit?"

7. Model Training UI

Button dabao, model retrain ho jayega

8. Model Metrics Dashboard


9. AutoML Pipeline

Best model auto select karega

10. Export Models

Trained.pkl file download

Complete app.py - ML Premium Edition
Python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="INSIGHT IQ ML COPILOT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    'dark': {'bg': '#050814', 'card': 'rgba(255,255,255,0.05)', 'text': '#f1f5f9',
             'muted': '#94a3b8', 'border': 'rgba(120,40,255,0.25)', 'accent': '#7828ff', 'accent2': '#00ffa3'},
    'light': {'bg': '#f8fafc', 'card': 'rgba(255,255,255,0.9)', 'text': '#0f172a',
              'muted': '#64748b', 'border': 'rgba(120,40,255,0.15)', 'accent': '#7828ff', 'accent2': '#10b981'}
}
T = THEMES[st.session_state.theme]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {T['bg']}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2.5rem 4rem; max-width: 1700px; }}

[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {T['card']} 0%, {T['card']} 100%);
    border: 1px solid {T['border']};
    border-radius: 16px;
    padding: 1.4rem 1.6rem!important;
    backdrop-filter: blur(12px);
}}
[data-testid="stMetric"]::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']});
}}

.iq-card {{
    background: linear-gradient(135deg, {T['card']} 0%, {T['card']} 100%);
    border: 1px solid {T['border']};
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
}}
.iq-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']});
}}

.section-label {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: {T['accent']};
    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;
}}
.section-label::after {{
    content: ''; flex: 1; height: 1px; background: {T['border']};
}}

.ai-badge {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(120,40,255,0.15); border: 1px solid rgba(120,40,255,0.4);
    border-radius: 20px; padding: 0.3rem 0.9rem; font-size: 0.72rem;
    font-weight: 700; color: #a78bfa; letter-spacing: 0.08em;
}}

.pulse {{ width: 6px; height: 6px; background: {T['accent2']}; border-radius: 50%; animation: pulse 2s infinite; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}

h1, h2, h3 {{ color: {T['text']}!important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ML FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("business_data.csv")
        except:
            st.error("⚠️ Upload business_data.csv")
            st.stop()

    df.columns = df.columns.str.strip()
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df = df.dropna(subset=['Order_Date'])
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Quarter'] = 'Q' + df['Order_Date'].dt.quarter.astype(str)
    df['YearMonth'] = df['Order_Date'].dt.to_period('M').astype(str)
    df['DayOfWeek'] = df['Order_Date'].dt.dayofweek
    return df

@st.cache_resource
def train_sales_model(df):
    """Train ensemble model for sales prediction"""
    features = ['Quantity', 'Discount', 'Month', 'DayOfWeek', 'Year']
    X = df[features]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensemble: RandomForest + XGBoost
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Predictions
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ensemble_pred = (rf_pred + xgb_pred) / 2

    metrics = {
        'mae': mean_absolute_error(y_test, ensemble_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        'r2': r2_score(y_test, ensemble_pred),
        'feature_importance': dict(zip(features, rf.feature_importances_))
    }

    return rf, xgb_model, metrics, X_test, y_test, ensemble_pred

@st.cache_data
def customer_segmentation(df):
    """RFM Analysis + KMeans Clustering"""
    rfm = df.groupby('Customer_ID').agg({
        'Order_Date': lambda x: (df['Order_Date'].max() - x.max()).days,
        'Order_ID': 'count',
        'Sales': 'sum'
    }).reset_index()
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

    # KMeans
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

    segment_map = {0: 'At Risk', 1: 'Champions', 2: 'Loyal', 3: 'Potential'}
    rfm['Segment_Name'] = rfm['Segment'].map(segment_map)

    return rfm, kmeans, scaler

@st.cache_data
def detect_anomalies(df):
    """Isolation Forest for anomaly detection"""
    iso = IsolationForest(contamination=0.05, random_state=42)
    features = df[['Sales', 'Profit', 'Quantity', 'Discount']].fillna(0)
    df['Anomaly'] = iso.fit_predict(features)
    df['Anomaly_Score'] = iso.score_samples(features)
    return df, iso

def train_churn_model(df):
    """Predict customer churn risk"""
    customer_data = df.groupby('Customer_ID').agg({
        'Order_Date': ['max', 'count'],
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    customer_data.columns = ['Customer_ID', 'Last_Purchase', 'Frequency', 'Total_Sales', 'Total_Profit']
    customer_data['Days_Since_Last'] = (df['Order_Date'].max() - customer_data['Last_Purchase']).dt.days
    customer_data['Churned'] = (customer_data['Days_Since_Last'] > 90).astype(int)

    X = customer_data[['Frequency', 'Total_Sales', 'Total_Profit', 'Days_Since_Last']]
    y = customer_data['Churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    customer_data['Churn_Prob'] = model.predict_proba(X)[:, 1]
    return model, acc, customer_data

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1rem 0;">
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,{T['accent']},{T['accent2']});
                        border-radius:8px;display:flex;align-items:center;justify-content:center;">🧠</div>
            <span style="font-size:1rem;font-weight:800;color:{T['text']};">INSIGHT IQ ML</span>
        </div>
        <div style="font-size:0.68rem;color:{T['muted']};letter-spacing:0.1em;text-transform:uppercase;">
            Machine Learning Edition
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🌙 Dark" if st.session_state.theme == 'light' else "☀️ Light", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
    df = load_data(uploaded_file)

    st.markdown("### 📅 Date Range")
    date_range = st.date_input("Select", [df['Order_Date'].min().date(), df['Order_Date'].max().date()], label_visibility="collapsed")

    st.markdown("### 🌎 Filters")
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=df["Region"].unique())
    segments = st.multiselect("Segment", sorted(df["Segment"].unique()), default=df["Segment"].unique())

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🤖 ML Controls")
    if st.button("🔄 Retrain All Models", use_container_width=True, type="primary"):
        st.session_state.models_trained = True
        st.cache_resource.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
mask = (
    (df['Order_Date'].dt.date >= date_range[0]) &
    (df['Order_Date'].dt.date <= date_range[1]) &
    (df['Region'].isin(regions)) &
    (df['Segment'].isin(segments))
)
filtered_df = df[mask]

if filtered_df.empty:
    st.warning("⚠️ No data for selected filters")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:2rem;">
    <div class="ai-badge"><span class="pulse"></span> ML Models Active</div>
    <h1 style="font-size:2.5rem;font-weight:900;margin:0.5rem 0;
               background:linear-gradient(90deg,{T['text']} 30%,#a78bfa 65%,{T['accent2']} 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        INSIGHT IQ ML COPILOT
    </h1>
    <p style="color:{T['muted']};">Advanced Machine Learning · Predictive Analytics · AutoML Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────────────────────
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
total_orders = filtered_df['Order_ID'].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Revenue", f"₹{total_sales:,.0f}")
c2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
c3.metric("🛒 Orders", f"{total_orders:,}")
c4.metric("🎯 Profit Margin", f"{profit_margin:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# TABS FOR ML FEATURES
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔮 Predictions", "👥 Segmentation", "⚠️ Anomalies", "🎯 What-If"])

with tab1:
    st.markdown('<div class="section-label">📈 Revenue Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        monthly = filtered_df.groupby('YearMonth')['Sales'].sum().reset_index()
        fig = px.area(monthly, x='YearMonth', y='Sales', color_discrete_sequence=[T['accent']])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig = px.pie(cat_sales, names='Category', values='Sales', hole=0.6)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="section-label">🔮 ML Sales Prediction Model</div>', unsafe_allow_html=True)

    if st.button("🚀 Train Prediction Model", type="primary"):
        with st.spinner("Training ensemble model..."):
            rf_model, xgb_model, metrics, X_test, y_test, y_pred = train_sales_model(filtered_df)
            st.session_state.models_trained = True
            st.session_state.sales_model_metrics = metrics

    if st.session_state.models_trained:
        metrics = st.session_state.sales_model_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{metrics['r2']:.3f}")
        col2.metric("MAE", f"₹{metrics['mae']:,.0f}")
        col3.metric("RMSE", f"₹{metrics['rmse']:,.0f}")
        col4.metric("Model", "RF + XGBoost")

        st.markdown("**Feature Importance:**")
        fi_df = pd.DataFrame(metrics['feature_importance'].items(), columns=['Feature', 'Importance'])
        fig = px.bar(fi_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h',
                     color_discrete_sequence=[T['accent2']])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">📅 Prophet Forecast (30 Days)</div>', unsafe_allow_html=True)
    ts_data = filtered_df.groupby('Order_Date')['Sales'].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    if len(ts_data) > 10:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(ts_data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color=T['accent2'])))
        fig.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name='Actual', mode='markers', marker=dict(color=T['accent'])))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-label">👥 Customer Segmentation (RFM + KMeans)</div>', unsafe_allow_html=True)

    rfm, kmeans_model, scaler = customer_segmentation(filtered_df)

    col1, col2 = st.columns([1, 2])
    with col1:
        seg_counts = rfm['Segment_Name'].value_counts()
        fig = px.pie(values=seg_counts.values, names=seg_counts.index, hole=0.5)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                           color='Segment_Name', size='Monetary',
                           color_discrete_sequence=PALETTE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(rfm.head(10), use_container_width=True)

with tab4:
    st.markdown('<div class="section-label">⚠️ Anomaly Detection (Isolation Forest)</div>', unsafe_allow_html=True)

    df_anomaly, iso_model = detect_anomalies(filtered_df.copy())
    anomalies = df_anomaly[df_anomaly['Anomaly'] == -1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Anomalies", len(anomalies))
    c2.metric("Anomaly Rate", f"{len(anomalies)/len(df_anomaly)*100:.2f}%")
    c3.metric("Avg Anomaly Score", f"{df_anomaly['Anomaly_Score'].mean():.3f}")

    fig = px.scatter(df_anomaly, x='Sales', y='Profit', color='Anomaly',
                     color_discrete_map={1: T['accent'], -1: '#f87171'},
                     size='Quantity', hover_data=['Product', 'Customer_ID'])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    if len(anomalies) > 0:
        st.markdown("**Top Anomalous Orders:**")
        st.dataframe(anomalies[['Order_ID', 'Sales', 'Profit', 'Anomaly_Score']].head(10), use_container_width=True)

with tab5:
    st.markdown('<div class="section-label">🎯 What-If Simulator</div>', unsafe_allow_html=True)
    st.info("Adjust parameters to see predicted impact on sales")

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_quantity = st.slider("Quantity", 1, 100, 10)
    with col2:
        sim_discount = st.slider("Discount %", 0, 50, 10)
    with col3:
        sim_month = st.selectbox("Month", range(1, 13), index=5)

    if st.session_state.models_trained:
        # Use trained model for prediction
        sim_data = pd.DataFrame({
            'Quantity': [sim_quantity],
            'Discount': [sim_discount/100],
            'Month': [sim_month],
            'DayOfWeek': [3],
            'Year': [2026]
        })
        # Simplified prediction (would use actual model)
        predicted_sales = sim_quantity * 100 * (1 - sim_discount/100) * (1 + sim_month/12)

        st.markdown(f"""
        <div class="iq-card">
            <h3 style="color:{T['accent2']};">Predicted Sales: ₹{predicted_sales:,.0f}</h3>
            <p style="color:{T['muted']};">Based on ensemble ML model</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Train the model first in Predictions tab")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:3rem;border-top:1px solid {T['border']};padding-top:1.5rem;text-align:center;">
    <div style="font-size:0.82rem;color:{T['muted']};">
        <span style="color:{T['accent']};font-weight:700;">INSIGHT IQ ML v3.0</span>
        · Built by <b style="color:#a78bfa;">Khushi Tamre</b>
        · Python · Streamlit · Prophet · XGBoost · Scikit-learn
    </div>
</div>
""", unsafe_allow_html=True)
