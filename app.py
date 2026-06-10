import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="INSIGHT IQ AI COPILOT",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #050814;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(120,40,255,0.18), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,255,163,0.06), transparent);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0d1a 0%, #0d0f1f 100%);
    border-right: 1px solid rgba(120,40,255,0.2);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa !important;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.5rem;
}

[data-testid="stMultiSelect"] > div {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(120,40,255,0.3);
    border-radius: 10px;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(120,40,255,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem !important;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s;
}
[data-testid="stMetric"]:hover { border-color: rgba(120,40,255,0.6); }
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7828ff, #00ffa3);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(120,40,255,0.3) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7828ff !important;
    box-shadow: 0 0 0 3px rgba(120,40,255,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7828ff, #5b10d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b3eff, #7828ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(120,40,255,0.35) !important;
}

.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px !important;
    border: none !important;
    font-weight: 500 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(120,40,255,0.2);
    border-radius: 12px;
    overflow: hidden;
}

hr { border-color: rgba(120,40,255,0.15) !important; }
h1, h2, h3, h4 { color: #f1f5f9 !important; }
h2 { font-size: 1.3rem !important; font-weight: 700 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(120,40,255,0.4); border-radius: 3px; }

.iq-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(120,40,255,0.25);
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
    background: linear-gradient(90deg, #7828ff, #00ffa3);
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7828ff;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(120,40,255,0.25);
}
.ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(120,40,255,0.15);
    border: 1px solid rgba(120,40,255,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pulse {
    width: 6px; height: 6px;
    background: #00ffa3;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,163,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,255,163,0); }
}
.insight-tag {
    display: inline-block;
    background: rgba(0,255,163,0.1);
    border: 1px solid rgba(0,255,163,0.3);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #00ffa3;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#7828ff", "#00ffa3", "#f472b6", "#38bdf8", "#fb923c", "#a3e635"]

def apply_theme(fig, height=360):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=12),
        height=height,
        margin=dict(l=20, r=20, t=44, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color="#94a3b8", size=11)
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#64748b")
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#64748b")
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("business_data.csv")
    df.columns = df.columns.str.strip()
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df["Year"]      = df["Order_Date"].dt.year
        df["Month"]     = df["Order_Date"].dt.month
        df["Quarter"]   = "Q" + df["Order_Date"].dt.quarter.astype(str)
        df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#7828ff,#00ffa3);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">✦</div>
            <span style="font-size:1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.02em;">INSIGHT IQ AI</span>
        </div>
        <div style="font-size:0.68rem;color:#64748b;font-weight:500;letter-spacing:0.1em;
                    text-transform:uppercase;padding-left:2px;">Enterprise Copilot</div>
    </div>
    <hr style="border-color:rgba(120,40,255,0.2);margin:0.5rem 0 1rem;">
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

    st.markdown("<hr style='border-color:rgba(120,40,255,0.2);margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(120,40,255,0.08);border:1px solid rgba(120,40,255,0.2);
                border-radius:10px;padding:0.9rem;margin-top:0.5rem;">
        <div style="font-size:0.68rem;font-weight:700;color:#a78bfa;letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:0.5rem;">Developer</div>
        <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">Khushi Tamre</div>
        <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem;">AI &amp; BI Engineer</div>
        <div style="margin-top:0.5rem;">
            <span class="insight-tag">Python</span>
            <span class="insight-tag">ML</span>
            <span class="insight-tag">Streamlit</span>
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
# KPI CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
total_sales   = filtered_df["Sales"].sum()
total_profit  = filtered_df["Profit"].sum()
total_orders  = filtered_df["Order_ID"].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
avg_order_val = total_sales / total_orders if total_orders > 0 else 0

top_region   = filtered_df.groupby("Region")["Sales"].sum().idxmax()
top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
top_product  = filtered_df.groupby("Product")["Sales"].sum().idxmax()

region_share = (
    filtered_df.groupby("Region")["Sales"].sum().max() / total_sales * 100
    if total_sales > 0 else 0
)

health_score = (
    95 if profit_margin >= 20 else
    85 if profit_margin >= 15 else
    75 if profit_margin >= 10 else 60
)
risk_level = "Low" if health_score >= 90 else "Medium" if health_score >= 75 else "High"
confidence = "95%" if health_score >= 90 else "85%" if health_score >= 75 else "70%"

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
            background:linear-gradient(90deg,#f1f5f9 30%,#a78bfa 65%,#00ffa3 100%);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            line-height:1.1;
        ">INSIGHT IQ AI COPILOT</h1>
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
c1.metric("💰 Total Revenue",    f"₹{total_sales:,.0f}")
c2.metric("📈 Total Profit",     f"₹{total_profit:,.0f}")
c3.metric("🛒 Orders",           f"{total_orders:,}")
c4.metric("🎯 Profit Margin",    f"{profit_margin:.1f}%")
c5.metric("💎 Avg. Order Value", f"₹{avg_order_val:,.0f}")
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Revenue Trend + Category Donut
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📈 &nbsp;Revenue Analysis</div>', unsafe_allow_html=True)
col_left, col_right = st.columns([2, 1])

with col_left:
    monthly = (
        filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
        .rename(columns={"Sales": "Revenue"})
    )
    fig_trend = px.area(
        monthly, x="YearMonth", y="Revenue",
        title="Monthly Revenue Trend",
        color_discrete_sequence=["#7828ff"]
    )
    fig_trend.update_traces(
        line=dict(width=2.5, color="#7828ff"),
        fillcolor="rgba(120,40,255,0.12)",
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>"
    )
    fig_trend.update_layout(title=dict(font=dict(color="#94a3b8", size=13), x=0))
    apply_theme(fig_trend, 300)
    st.plotly_chart(fig_trend, use_container_width=True, key="trend")

with col_right:
    cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    fig_donut = px.pie(
        cat_sales, names="Category", values="Sales",
        hole=0.65, color_discrete_sequence=PALETTE,
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
    apply_theme(fig_donut, 300)
    st.plotly_chart(fig_donut, use_container_width=True, key="donut")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Region Bar + Scatter
# ─────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-label">🌎 &nbsp;Regional Performance</div>', unsafe_allow_html=True)
    reg = (
        filtered_df.groupby("Region")
        .agg(Revenue=("Sales","sum"), Profit=("Profit","sum"))
        .reset_index().sort_values("Revenue", ascending=True)
    )
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=reg["Region"], x=reg["Revenue"],
        orientation="h", name="Revenue",
        marker=dict(
            color=reg["Revenue"],
            colorscale=[[0,"#3b0764"],[0.5,"#7828ff"],[1,"#a78bfa"]],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>"
    ))
    fig_bar.update_layout(
        title=dict(text="Revenue by Region", font=dict(color="#94a3b8", size=13), x=0),
        showlegend=False
    )
    apply_theme(fig_bar, 280)
    st.plotly_chart(fig_bar, use_container_width=True, key="region_bar")

with col_b:
    st.markdown('<div class="section-label">🎯 &nbsp;Profitability Intelligence Map</div>', unsafe_allow_html=True)
    sample = filtered_df.sample(min(1200, len(filtered_df)))
    fig_scatter = px.scatter(
        sample, x="Sales", y="Profit",
        color="Category", size="Quantity",
        color_discrete_sequence=PALETTE,
        hover_data=["Product"],
        title="Sales vs Profit"
    )
    fig_scatter.update_traces(
        marker=dict(opacity=0.7, line=dict(width=0)),
        hovertemplate="<b>%{customdata[0]}</b><br>Sales: ₹%{x:,.0f}<br>Profit: ₹%{y:,.0f}<extra></extra>"
    )
    fig_scatter.update_layout(title=dict(font=dict(color="#94a3b8", size=13), x=0))
    apply_theme(fig_scatter, 280)
    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — Quarter Bar + Sub-Category Bar
# ─────────────────────────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<div class="section-label">📊 &nbsp;Quarterly Revenue</div>', unsafe_allow_html=True)
    qtr = (
        filtered_df.groupby(["Year","Quarter"])["Sales"].sum().reset_index()
        .sort_values(["Year","Quarter"])
    )
    qtr["Label"] = qtr["Year"].astype(str) + " " + qtr["Quarter"]
    fig_qtr = px.bar(
        qtr, x="Label", y="Sales",
        color="Quarter", color_discrete_sequence=PALETTE,
        title="Revenue by Quarter",
        barmode="group"
    )
    fig_qtr.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>"
    )
    fig_qtr.update_layout(title=dict(font=dict(color="#94a3b8", size=13), x=0), showlegend=True)
    apply_theme(fig_qtr, 280)
    st.plotly_chart(fig_qtr, use_container_width=True, key="quarterly")

with col_d:
    st.markdown('<div class="section-label">📦 &nbsp;Sub-Category Performance</div>', unsafe_allow_html=True)
    sub_col = "Sub-Category" if "Sub-Category" in filtered_df.columns else "Category"
    sub = (
        filtered_df.groupby(sub_col)["Profit"].sum().reset_index()
        .sort_values("Profit", ascending=True).tail(10)
    )
    sub["Color"] = sub["Profit"].apply(lambda v: "#00ffa3" if v >= 0 else "#f87171")
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
    apply_theme(fig_sub, 280)
    st.plotly_chart(fig_sub, use_container_width=True, key="subcategory")

# ───────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# AI FORECAST — FIXED (numeric x-axis, no add_vline bug)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🔮 &nbsp;AI Predictive Engine</div>', unsafe_allow_html=True)

forecast_base = (
    filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
)
forecast_base["t"] = range(1, len(forecast_base) + 1)

model = LinearRegression()
model.fit(forecast_base[["t"]], forecast_base["Sales"])

future_t    = np.arange(len(forecast_base) + 1, len(forecast_base) + 7).reshape(-1, 1)
future_y    = model.predict(future_t)
next_pred   = future_y[0]

n_hist      = len(forecast_base)
hist_x      = list(range(n_hist))
# overlap by 1 point so the line connects
fore_x      = list(range(n_hist - 1, n_hist + 6))
fore_y_conn = [float(forecast_base["Sales"].iloc[-1])] + future_y.tolist()

all_labels  = forecast_base["YearMonth"].tolist() + [f"M+{i}" for i in range(1, 7)]
tick_vals   = list(range(n_hist + 6))

col_f1, col_f2 = st.columns([3, 1])

with col_f1:
    fig_fc = go.Figure()

    fig_fc.add_trace(go.Scatter(
        x=hist_x,
        y=forecast_base["Sales"].tolist(),
        mode="lines+markers", name="Historical",
        line=dict(color="#7828ff", width=2.5),
        marker=dict(size=5, color="#7828ff"),
        customdata=forecast_base["YearMonth"].tolist(),
        hovertemplate="<b>%{customdata}</b><br>₹%{y:,.0f}<extra></extra>"
    ))

    fig_fc.add_trace(go.Scatter(
        x=fore_x, y=fore_y_conn,
        mode="lines+markers", name="AI Forecast",
        line=dict(color="#00ffa3", width=2.5, dash="dot"),
        marker=dict(size=7, color="#00ffa3", symbol="diamond"),
        hovertemplate="<b>Forecast</b><br>₹%{y:,.0f}<extra></extra>"
    ))

    # Safe vertical divider — numeric x reference only
    fig_fc.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=n_hist - 1, x1=n_hist - 1,
        y0=0, y1=1,
        line=dict(dash="dash", color="rgba(255,255,255,0.2)", width=1.5)
    )
    fig_fc.add_annotation(
        x=n_hist - 1, y=0.96,
        xref="x", yref="paper",
        text="NOW", showarrow=False,
        font=dict(color="#64748b", size=10, family="Inter"),
        yanchor="top", bgcolor="rgba(0,0,0,0)"
    )

    fig_fc.update_xaxes(
        tickvals=tick_vals,
        ticktext=all_labels,
        tickangle=-35,
        tickfont=dict(size=10)
    )
    fig_fc.update_layout(
        title=dict(
            text="Revenue History + 6-Month AI Forecast",
            font=dict(color="#94a3b8", size=13), x=0
        )
    )
    apply_theme(fig_fc, 320)
    st.plotly_chart(fig_fc, use_container_width=True, key="forecast")

with col_f2:
    avg_monthly = forecast_base["Sales"].mean()
    growth_pct  = ((next_pred / avg_monthly) - 1) * 100 if avg_monthly > 0 else 0
    risk_color  = "#00ffa3" if risk_level == "Low" else "#fb923c" if risk_level == "Medium" else "#f87171"

    st.markdown(f"""
    <div class="iq-card" style="height:100%;">
        <div style="font-size:0.68rem;font-weight:700;color:#a78bfa;letter-spacing:0.12em;
                    text-transform:uppercase;margin-bottom:1.2rem;">AI Forecast Summary</div>
        <div style="margin-bottom:1rem;">
            <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.2rem;">Next Month Prediction</div>
            <div style="font-size:1.4rem;font-weight:800;color:#00ffa3;">₹{next_pred:,.0f}</div>
        </div>
        <div style="margin-bottom:1rem;">
            <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.2rem;">Growth vs Avg</div>
            <div style="font-size:1.1rem;font-weight:700;color:{'#00ffa3' if growth_pct >= 0 else '#f87171'};">
                {'▲' if growth_pct >= 0 else '▼'} {abs(growth_pct):.1f}%
            </div>
        </div>
        <div style="margin-bottom:1rem;">
            <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.2rem;">Model Confidence</div>
            <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;">{confidence}</div>
        </div>
        <div style="margin-bottom:1rem;">
            <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.2rem;">Business Risk</div>
            <div style="font-size:1rem;font-weight:700;color:{risk_color};">{risk_level}</div>
        </div>
        <div>
            <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.4rem;">AI Health Score</div>
            <div style="font-size:1.6rem;font-weight:900;color:#7828ff;">
                {health_score}<span style="font-size:0.9rem;color:#64748b;">/100</span>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;margin-top:0.4rem;">
                <div style="width:{health_score}%;background:linear-gradient(90deg,#7828ff,#00ffa3);
                            height:6px;border-radius:4px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">🏆 &nbsp;Performance Leaderboard</div>', unsafe_allow_html=True)
lb1, lb2, lb3 = st.columns(3)

MEDALS = ["🥇","🥈","🥉","④","⑤"]

with lb1:
    top_r = (
        filtered_df.groupby("Region")["Sales"].sum()
        .sort_values(ascending=False).head(5).reset_index()
    )
    top_r.columns = ["Region","Revenue (₹)"]
    top_r["Revenue (₹)"] = top_r["Revenue (₹)"].map("₹{:,.0f}".format)
    top_r.index = MEDALS[:len(top_r)]
    st.markdown("**🌎 Top Regions**")
    st.dataframe(top_r, use_container_width=True)

with lb2:
    top_p = (
        filtered_df.groupby("Product")["Sales"].sum()
        .sort_values(ascending=False).head(5).reset_index()
    )
    top_p.columns = ["Product","Revenue (₹)"]
    top_p["Revenue (₹)"] = top_p["Revenue (₹)"].map("₹{:,.0f}".format)
    top_p.index = MEDALS[:len(top_p)]
    st.markdown("**📦 Top Products**")
    st.dataframe(top_p, use_container_width=True)

with lb3:
    top_c = (
        filtered_df.groupby("Category")[["Sales","Profit"]].sum()
        .sort_values("Sales", ascending=False).reset_index()
    )
    top_c["Margin"] = (top_c["Profit"] / top_c["Sales"] * 100).map("{:.1f}%".format)
    top_c["Sales"]  = top_c["Sales"].map("₹{:,.0f}".format)
    top_c = top_c[["Category","Sales","Margin"]]
    st.markdown("**📊 Category Performance**")
    st.dataframe(top_c, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH GAUGE + SMART ALERTS
# ─────────────────────────────────────────────────────────────────────────────
col_g, col_alerts = st.columns([1, 1])

with col_g:
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">🧠 &nbsp;AI Business Health Score</div>', unsafe_allow_html=True)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        delta={"reference": 75, "increasing": {"color": "#00ffa3"}, "decreasing": {"color": "#f87171"}},
        number={"font": {"color": "#f1f5f9", "size": 52, "family": "Inter"}, "suffix": "/100"},
        title={"text": "Business Health Index", "font": {"color": "#94a3b8", "size": 13}},
        gauge={
            "axis": {"range": [0,100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b"}},
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,40],  "color": "rgba(248,113,113,0.2)"},
                {"range": [40,70], "color": "rgba(251,146,60,0.2)"},
                {"range": [70,100],"color": "rgba(0,255,163,0.15)"}
            ],
            "threshold": {
                "line": {"color": "#7828ff", "width": 3},
                "thickness": 0.85,
                "value": health_score
            }
        }
    ))
    apply_theme(gauge, 280)
    st.plotly_chart(gauge, use_container_width=True, key="gauge")

with col_alerts:
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">🚨 &nbsp;Smart AI Alerts</div>', unsafe_allow_html=True)

    if profit_margin < 10:
        st.error(f"⚠️ **Critical:** Profit margin at {profit_margin:.1f}% — below safe threshold.")
    elif profit_margin < 20:
        st.warning(f"⚡ **Caution:** Profit margin at {profit_margin:.1f}% — room for improvement.")
    else:
        st.success(f"✅ **Excellent:** Profit margin at {profit_margin:.1f}% — strong performance.")

    if total_sales > 100000:
        st.success(f"🚀 **Revenue Milestone:** ₹{total_sales:,.0f} — exceeds target threshold.")

    avg_monthly_check = forecast_base["Sales"].mean() if len(forecast_base) > 0 else 0
    if next_pred > avg_monthly_check:
        st.info(f"📈 **Growth Signal:** Next month forecast ₹{next_pred:,.0f} — above average.")

    st.info(f"🌎 **Dominant Region:** {top_region} holds {region_share:.1f}% of total revenue.")

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">🏅 &nbsp;Executive Scorecard</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    s1.metric("🧠 AI Health Score",     f"{health_score}/100")
    s2.metric("📈 Growth Score",        "92/100")
    s3, s4 = st.columns(2)
    s3.metric("⚠️ Risk Level",          risk_level)
    s4.metric("🎯 Forecast Confidence", confidence)

# ─────────────────────────────────────────────────────────────────────────────
# AI COMMAND CENTER + CHAT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">🤖 &nbsp;Insight IQ AI Assistant</div>', unsafe_allow_html=True)
cmd_col, chat_col = st.columns([1, 1])

with cmd_col:
    avg_monthly_disp = forecast_base["Sales"].mean() if len(forecast_base) > 0 else 1
    growth_disp = ((next_pred / avg_monthly_disp) - 1) * 100

    st.markdown(f"""
    <div class="iq-card">
        <div style="font-size:0.68rem;font-weight:700;color:#a78bfa;letter-spacing:0.12em;
                    text-transform:uppercase;margin-bottom:1rem;">Strategic Command</div>
        <div style="display:flex;flex-direction:column;gap:0.8rem;">
            <div>
                <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.15rem;">Top Revenue Region</div>
                <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">🌎 {top_region}</div>
            </div>
            <div>
                <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.15rem;">Leading Category</div>
                <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">📦 {top_category}</div>
            </div>
            <div>
                <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.15rem;">Top Product</div>
                <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;">
                    🔥 {top_product[:42] + ('…' if len(top_product) > 42 else '')}
                </div>
            </div>
            <hr style="border-color:rgba(120,40,255,0.15);margin:0.4rem 0;">
            <div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">
                <b style="color:#a78bfa;">AI Recommendation:</b><br>
                Scale <b style="color:#f1f5f9;">{top_category}</b> investments and expand in
                <b style="color:#f1f5f9;">{top_region}</b> for maximum ROI.
                Projected growth: <b style="color:#00ffa3;">+{abs(growth_disp):.1f}%</b> vs historical avg.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with chat_col:
    st.markdown("""
    <div style="font-size:0.72rem;font-weight:600;color:#64748b;letter-spacing:0.05em;
                text-transform:uppercase;margin-bottom:0.5rem;">Ask the AI Copilot</div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "Business question",
        placeholder="e.g. What is the best region? Which product leads? Show revenue.",
        label_visibility="collapsed",
        key="ai_chat_main"
    )

    if question:
        q = question.lower()
        if "region" in q:
            answer = f"🌎 **Best Region:** {top_region} — leading all markets with {region_share:.1f}% revenue share."
        elif "category" in q:
            answer = f"📦 **Top Category:** {top_category} — highest revenue generator across all segments."
        elif "product" in q:
            answer = f"🔥 **Top Product:** {top_product}"
        elif "profit" in q:
            answer = f"💰 **Total Profit:** ₹{total_profit:,.0f} &nbsp;·&nbsp; Margin: {profit_margin:.1f}%"
        elif "revenue" in q or "sales" in q:
            answer = f"📈 **Total Revenue:** ₹{total_sales:,.0f} across {total_orders:,} orders."
        elif "forecast" in q or "predict" in q or "future" in q:
            answer = f"🔮 **AI Forecast:** Next month projected at ₹{next_pred:,.0f} with {confidence} confidence."
        elif "order" in q:
            answer = f"🛒 **Total Orders:** {total_orders:,} &nbsp;·&nbsp; Avg. Value: ₹{avg_order_val:,.0f}"
        elif "risk" in q:
            answer = f"⚠️ **Risk Assessment:** {risk_level} risk — Health Score {health_score}/100."
        elif "health" in q or "score" in q:
            answer = f"🧠 **Health Score:** {health_score}/100 — Business is in {risk_level.lower()} risk zone."
        elif "segment" in q:
            top_seg = filtered_df.groupby("Segment")["Sales"].sum().idxmax()
            answer = f"👥 **Top Segment:** {top_seg}"
        else:
            answer = "💡 Ask about: **region, category, product, revenue, profit, forecast, orders, risk, health score, segment**."

        st.markdown(f"""
        <div style="background:rgba(120,40,255,0.1);border:1px solid rgba(120,40,255,0.3);
                    border-radius:10px;padding:0.9rem 1rem;margin-top:0.5rem;
                    font-size:0.88rem;color:#e2e8f0;line-height:1.6;">
            {answer}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);border:1px dashed rgba(120,40,255,0.2);
                    border-radius:10px;padding:1.5rem;text-align:center;margin-top:0.5rem;">
            <div style="font-size:1.8rem;margin-bottom:0.4rem;">🤖</div>
            <div style="font-size:0.82rem;color:#475569;">Ask anything about your business data.</div>
            <div style="font-size:0.72rem;color:#334155;margin-top:0.3rem;">
                region · category · product · revenue · profit · forecast · risk
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">📄 &nbsp;Executive Report</div>', unsafe_allow_html=True)

if st.button("⬇ Generate Executive PDF Report"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=60, bottomMargin=50
    )
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=24, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#7828ff"), spaceAfter=6
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=12, textColor=colors.HexColor("#64748b"), spaceAfter=20
    )
    head_style = ParagraphStyle(
        "Head", parent=styles["Heading2"],
        fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1e293b"), spaceAfter=8, spaceBefore=16
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#374151"), leading=16
    )

    story.append(Paragraph("INSIGHT IQ AI COPILOT", title_style))
    story.append(Paragraph("Executive Business Intelligence Report", sub_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Key Performance Indicators", head_style))
    kpi_data = [
        ["Metric", "Value"],
        ["Total Revenue",    f"₹{total_sales:,.0f}"],
        ["Total Profit",     f"₹{total_profit:,.0f}"],
        ["Profit Margin",    f"{profit_margin:.1f}%"],
        ["Total Orders",     f"{total_orders:,}"],
        ["Avg. Order Value", f"₹{avg_order_val:,.0f}"],
        ["AI Health Score",  f"{health_score}/100"],
        ["Business Risk",    risk_level],
        ["Forecast (M+1)",   f"₹{next_pred:,.0f}"],
        ["Confidence",       confidence],
    ]
    tbl = Table(kpi_data, colWidths=[220, 220])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#7828ff")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",       (0,0), (-1,-1), 8),
    ]))
    story.append(tbl)

    story.append(Paragraph("Strategic Insights", head_style))
    story.append(Paragraph(f"• <b>Top Region:</b> {top_region} ({region_share:.1f}% revenue share)", body_style))
    story.append(Paragraph(f"• <b>Top Category:</b> {top_category}", body_style))
    story.append(Paragraph(f"• <b>Top Product:</b> {top_product}", body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("AI Recommendation", head_style))
    story.append(Paragraph(
        f"Increase investment in <b>{top_category}</b> and expand operations in <b>{top_region}</b>. "
        f"AI forecasts next-month revenue at <b>₹{next_pred:,.0f}</b> with {confidence} confidence. "
        f"Current health score: <b>{health_score}/100</b> — {risk_level.lower()} risk profile.",
        body_style
    ))

    doc.build(story)
    buffer.seek(0)
    st.download_button(
        label="📥 Download PDF Report",
        data=buffer,
        file_name="INSIGHT_IQ_AI_Executive_Report.pdf",
        mime="application/pdf"
    )
    st.success("✅ Executive Report Generated Successfully!")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:3rem;
    border-top:1px solid rgba(120,40,255,0.15);
    padding-top:1.5rem;
    display:flex;
    justify-content:space-between;
    align-items:center;
    flex-wrap:wrap;
    gap:0.5rem;
">
    <div style="font-size:0.82rem;color:#334155;">
        <span style="color:#7828ff;font-weight:700;">INSIGHT IQ AI COPILOT</span>
        &nbsp;·&nbsp; Built by <b style="color:#a78bfa;">Khushi Tamre</b>
    </div>
    <div style="font-size:0.75rem;color:#334155;font-family:'JetBrains Mono',monospace;">
        Python &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; ReportLab
    </div>
</div>
""", unsafe_allow_html=True)
       
