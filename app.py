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
