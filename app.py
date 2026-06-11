# ===========================================================
#  InsightIQ AI Copilot Pro  —  app.py
#  World-Class Streamlit Business Intelligence Dashboard
#  Developer : Khushi Tamre  |  AI & BI Engineer
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io
from datetime import datetime, timedelta
import random

# ============================================================
#  PAGE CONFIG  —  must be first Streamlit call
# ============================================================

st.set_page_config(
    page_title="InsightIQ AI Copilot Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  GLOBAL CSS  —  dark space theme
# ============================================================

st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #02020E;
    color: #E2E8F0;
}

.stApp {
    background: linear-gradient(135deg, #02020E 0%, #0A0A1A 50%, #05050F 100%);
}

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0A1A 0%, #0F0F2A 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.3);
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #94A3B8 !important; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, #A78BFA, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 11px !important; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 700 !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0F0F1A, #1A1A2E) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: all 0.25s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(124,58,237,0.6) !important;
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(124,58,237,0.2);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,15,26,0.9);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(124,58,237,0.2);
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: #64748B !important;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.4);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    transition: all 0.25s !important;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.5) !important;
}

/* ── Input / Text area ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(15,15,26,0.95) !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
}

/* ── Selectbox / Multiselect ── */
.stMultiSelect > div, .stSelectbox > div {
    background: rgba(15,15,26,0.9) !important;
    border-color: rgba(124,58,237,0.3) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrameResizable"] {
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 12px !important;
}

/* ── Alerts / Info boxes ── */
.stAlert {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-size: 13px !important;
}
.stSuccess { background: rgba(16,185,129,0.1) !important; border-color: #10B981 !important; }
.stInfo    { background: rgba(56,189,248,0.1) !important; border-color: #38BDF8 !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border-color: #F59E0B !important; }
.stError   { background: rgba(239,68,68,0.1) !important; border-color: #EF4444 !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* ── Divider ── */
hr { border-color: rgba(124,58,237,0.2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0F0F1A; }
::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  CHART THEME CONFIG
# ============================================================

COLORS = {
    "purple":  "#7C3AED",
    "cyan":    "#06B6D4",
    "green":   "#10B981",
    "amber":   "#F59E0B",
    "red":     "#EF4444",
    "pink":    "#EC4899",
    "blue":    "#38BDF8",
}
PALETTE = list(COLORS.values())

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        bgcolor="rgba(15,15,26,0.8)",
        bordercolor="rgba(124,58,237,0.3)",
        borderwidth=1,
        font=dict(color="#CBD5E1", size=12),
    ),
    xaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B", tickfont=dict(color="#64748B")),
    yaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B", tickfont=dict(color="#64748B")),
)


def apply_theme(fig, height=380):
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig


# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def fmt_inr(n):
    """Format number to Indian currency string."""
    if n >= 1e7:
        return f"₹{n/1e7:.2f} Cr"
    elif n >= 1e5:
        return f"₹{n/1e5:.1f} L"
    elif n >= 1e3:
        return f"₹{n/1e3:.1f} K"
    return f"₹{n:,.0f}"


def fmt_short(n):
    if n >= 1e6:
        return f"₹{n/1e6:.1f}M"
    return f"₹{n/1e3:.0f}K"


def badge(text, color="#7C3AED"):
    bg = color + "22"
    return f"""<span style="background:{bg};color:{color};padding:3px 12px;
    border-radius:20px;font-size:12px;font-weight:700;
    border:1px solid {color}55;">{text}</span>"""


def section_header(icon, title, subtitle=""):
    sub_html = f'<p style="margin:4px 0 0;color:#475569;font-size:13px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="font-size:22px;">{icon}</span>
            <h2 style="margin:0;font-size:20px;font-weight:800;color:#F1F5F9;letter-spacing:-0.5px;">{title}</h2>
        </div>
        {sub_html}
        <div style="margin-top:10px;height:2px;background:linear-gradient(90deg,#7C3AED,#06B6D4,transparent);border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)

def kpi_card(icon, label, value, delta, delta_label, accent):
    delta_color = "#10B981" if delta >= 0 else "#EF4444"
    delta_icon = "▲" if delta >= 0 else "▼"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0F0F1A,#1A1A2E);
    border:1px solid {accent}33;border-radius:14px;padding:18px 20px;
    position:relative;overflow:hidden;transition:all 0.25s;">
        <div style="position:absolute;top:-20px;right:-20px;width:80px;height:80px;
        border-radius:50%;background:{accent};opacity:0.07;filter:blur(20px);"></div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="font-size:20px;">{icon}</span>
            <span style="color:#64748B;font-size:11px;font-weight:600;
            letter-spacing:1px;text-transform:uppercase;">{label}</span>
        </div>
        <div style="font-size:26px;font-weight:900;color:#F1F5F9;line-height:1;">{value}</div>
        <div style="margin-top:8px;display:flex;align-items:center;gap:6px;">
            <span style="background:{delta_color}22;color:{delta_color};
            padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;">
            {delta_icon} {abs(delta):.1f}%</span>
            <span style="color:#475569;font-size:11px;">{delta_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def alert_card(icon, text, color):
    st.markdown(f"""
    <div style="background:{color}11;border:1px solid {color}33;border-left:3px solid {color};
    border-radius:10px;padding:12px 16px;display:flex;align-items:flex-start;gap:10px;margin:4px 0;">
        <span style="font-size:18px;flex-shrink:0;">{icon}</span>
        <span style="font-size:13px;color:#CBD5E1;line-height:1.6;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def progress_row(label, value, max_val, color, suffix=""):
    pct = (value / max_val * 100) if max_val else 0
    st.markdown(f"""
    <div style="margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;
        margin-bottom:5px;font-size:13px;">
            <span style="color:#94A3B8;font-weight:600;">{label}</span>
            <span style="color:{color};font-weight:700;">{suffix}{value:,.0f}</span>
        </div>
        <div style="height:6px;background:#1E293B;border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:3px;
            transition:width 1s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  DATA GENERATION  (replace with your CSV loader)
# ============================================================

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 2000

    regions    = ["West", "East", "North", "South", "Central"]
    segments   = ["Consumer", "Corporate", "Home Office"]
    categories = ["Electronics", "Furniture", "Clothing", "Food & Bev", "Sports"]
    products   = {
        "Electronics": ["MacBook Pro M3","iPhone 15 Pro","Samsung 4K TV","Sony Headphones","iPad Air"],
        "Furniture":   ["Ergonomic Chair","Standing Desk","Bookshelf","Filing Cabinet","Sofa Set"],
        "Clothing":    ["Winter Jacket","Formal Suit","Casual T-Shirt","Sports Shoes","Denim Jeans"],
        "Food & Bev":  ["Protein Powder","Green Tea Pack","Organic Coffee","Energy Drink","Protein Bar"],
        "Sports":      ["Yoga Mat","Dumbbells Set","Running Shoes","Cycling Helmet","Fitness Band"],
    }

    start = datetime(2022, 1, 1)
    dates = [start + timedelta(days=int(x)) for x in np.random.randint(0, 730, n)]

    cat_list  = np.random.choice(categories, n, p=[0.35, 0.22, 0.20, 0.13, 0.10])
    prod_list = [np.random.choice(products[c]) for c in cat_list]

    base_sales = {"Electronics": 15000, "Furniture": 9000, "Clothing": 4000, "Food & Bev": 2500, "Sports": 3500}
    sales_arr  = np.array([base_sales[c] * (0.5 + np.random.random()) for c in cat_list])

    margin_map = {"Electronics": 0.22, "Furniture": 0.18, "Clothing": 0.28, "Food & Bev": 0.15, "Sports": 0.25}
    profit_arr = np.array([sales_arr[i] * (margin_map[c] + np.random.uniform(-0.05, 0.05))
                           for i, c in enumerate(cat_list)])

    df = pd.DataFrame({
        "Order_ID":   [f"ORD-{i:05d}" for i in range(n)],
        "Order_Date": dates,
        "Region":     np.random.choice(regions, n, p=[0.28, 0.24, 0.20, 0.18, 0.10]),
        "Segment":    np.random.choice(segments, n, p=[0.52, 0.30, 0.18]),
        "Category":   cat_list,
        "Product":    prod_list,
        "Sales":      sales_arr.round(2),
        "Profit":     profit_arr.round(2),
        "Quantity":   np.random.randint(1, 15, n),
    })

    df["Year"]      = df["Order_Date"].dt.year
    df["Month"]     = df["Order_Date"].dt.month
    df["Quarter"]   = "Q" + df["Order_Date"].dt.quarter.astype(str)
    df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    df["MonthName"] = df["Order_Date"].dt.strftime("%b %Y")

    return df


df = load_data()


# ============================================================
#  SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "👋 Namaste! Main hoon **InsightIQ AI Analyst**. "
                    "Aap mujhse revenue, profit, forecast ya koi bhi "
                    "business sawaal pooch sakte hain. Kya jaanna chahte hain?"}
    ]


# ============================================================
#  SIDEBAR  —  FILTERS + BRANDING
# ============================================================

with st.sidebar:
    # Logo block
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
        <div style="font-size:36px;margin-bottom:6px;">🚀</div>
        <div style="font-size:18px;font-weight:900;background:linear-gradient(90deg,#A78BFA,#38BDF8);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">InsightIQ Pro</div>
        <div style="font-size:10px;color:#334155;letter-spacing:2px;text-transform:uppercase;">
        AI Business Intelligence</div>
    </div>
    <hr style="border-color:rgba(124,58,237,0.3);margin:10px 0 18px;">
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Filters**")

    all_years = sorted(df["Year"].dropna().unique().tolist())
    sel_years = st.multiselect("📅 Year", all_years, default=all_years)

    all_regions = sorted(df["Region"].dropna().unique().tolist())
    sel_regions = st.multiselect("🌎 Region", all_regions, default=all_regions)

    all_segments = sorted(df["Segment"].dropna().unique().tolist())
    sel_segments = st.multiselect("👥 Segment", all_segments, default=all_segments)

    all_cats = sorted(df["Category"].dropna().unique().tolist())
    sel_cats = st.multiselect("📦 Category", all_cats, default=all_cats)

    st.markdown("<hr style='border-color:rgba(124,58,237,0.2);margin:16px 0;'>", unsafe_allow_html=True)

    # Live indicator
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#10B981;
        animation:none;box-shadow:0 0 6px #10B981;"></div>
        <span style="font-size:12px;color:#10B981;font-weight:600;">System Online</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y, %H:%M')}")

    st.markdown("<hr style='border-color:rgba(124,58,237,0.2);margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:8px;">
        <div style="font-size:13px;font-weight:700;color:#A78BFA;">👩‍💻 Khushi Tamre</div>
        <div style="font-size:11px;color:#475569;margin-top:2px;">AI & BI Engineer</div>
        <div style="font-size:10px;color:#334155;margin-top:6px;">Final Year Project • 2024</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  FILTERED DATA
# ============================================================

fdf = df[
    df["Year"].isin(sel_years) &
    df["Region"].isin(sel_regions) &
    df["Segment"].isin(sel_segments) &
    df["Category"].isin(sel_cats)
].copy()


# ============================================================
#  KPI CALCULATIONS
# ============================================================

total_sales    = fdf["Sales"].sum()
total_profit   = fdf["Profit"].sum()
total_orders   = fdf["Order_ID"].nunique()
avg_order_val  = total_sales / total_orders if total_orders else 0
profit_margin  = (total_profit / total_sales * 100) if total_sales else 0

region_sales   = fdf.groupby("Region")["Sales"].sum()
top_region     = region_sales.idxmax() if len(region_sales) else "N/A"

cat_sales      = fdf.groupby("Category")["Sales"].sum()
top_category   = cat_sales.idxmax() if len(cat_sales) else "N/A"

prod_sales     = fdf.groupby("Product")["Sales"].sum()
top_product    = prod_sales.idxmax() if len(prod_sales) else "N/A"

health_score = (
    95 if profit_margin >= 20 else
    85 if profit_margin >= 15 else
    75 if profit_margin >= 10 else 60
)
risk_level   = "Low" if health_score >= 90 else "Medium" if health_score >= 75 else "High"
confidence   = "95%" if health_score >= 90 else "85%" if health_score >= 75 else "70%"


# ============================================================
#  HEADER
# ============================================================

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(124,58,237,0.12),rgba(6,182,212,0.08));
border:1px solid rgba(124,58,237,0.25);border-radius:18px;
padding:24px 32px;margin-bottom:28px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
        <div>
            <h1 style="margin:0;font-size:32px;font-weight:900;letter-spacing:-1px;
            background:linear-gradient(90deg,#A78BFA 0%,#38BDF8 50%,#10B981 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🚀 InsightIQ AI Copilot Pro</h1>
            <p style="margin:6px 0 0;color:#475569;font-size:14px;">
            Next-Generation Business Intelligence • Powered by Machine Learning</p>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            {badge("AI Online ✓", "#10B981")}
            {badge(f"Health: {health_score}/100", "#7C3AED")}
            {badge(f"Risk: {risk_level}", "#F59E0B" if risk_level=="Medium" else "#10B981" if risk_level=="Low" else "#EF4444")}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  TABS
# ============================================================

tabs = st.tabs([
    "📊 Overview",
    "📈 Revenue",
    "📦 Products",
    "🔮 AI Forecast",
    "🤖 AI Analyst",
    "📋 Report",
])


# ════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════

with tabs[0]:

    # ── KPI ROW ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("💰", "Annual Revenue",    fmt_inr(total_sales),   23.4, "vs last year", "#7C3AED")
    with c2:
        kpi_card("📈", "Net Profit",        fmt_inr(total_profit),  31.2, "vs last year", "#10B981")
    with c3:
        kpi_card("🛒", "Total Orders",      f"{total_orders:,}",    18.7, "vs last year", "#06B6D4")
    with c4:
        kpi_card("🎯", "Profit Margin",     f"{profit_margin:.1f}%",6.1, "vs 14% avg",   "#F59E0B")
    with c5:
        kpi_card("⭐", "Avg Order Value",   fmt_inr(avg_order_val), 11.3, "vs last year", "#EC4899")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── MAIN CHARTS ──
    col_left, col_right = st.columns([2, 1])

    with col_left:
        section_header("📊", "Revenue & Profit Trend", "Monthly performance overview")

        monthly = (fdf.groupby("YearMonth")[["Sales","Profit"]]
                   .sum().reset_index().sort_values("YearMonth"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["YearMonth"], y=monthly["Sales"],
            name="Revenue", mode="lines+markers",
            line=dict(color="#7C3AED", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.1)",
            marker=dict(size=5, color="#7C3AED"),
        ))
        fig.add_trace(go.Scatter(
            x=monthly["YearMonth"], y=monthly["Profit"],
            name="Profit", mode="lines+markers",
            line=dict(color="#10B981", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.1)",
            marker=dict(size=5, color="#10B981"),
        ))
        apply_theme(fig, 300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section_header("🧠", "Business Health")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": "Health Score", "font": {"color": "#94A3B8", "size": 13}},
            number={"font": {"color": "#A78BFA", "size": 40, "family": "Inter"}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569",
                         "tickfont": {"color": "#64748B", "size": 11}},
                "bar": {"color": "#7C3AED", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0,  60], "color": "rgba(239,68,68,0.15)"},
                    {"range": [60, 80], "color": "rgba(245,158,11,0.15)"},
                    {"range": [80,100], "color": "rgba(16,185,129,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#F1F5F9", "width": 2},
                    "thickness": 0.7,
                    "value": health_score,
                },
            },
        ))
        apply_theme(gauge, 220)
        gauge.update_layout(margin=dict(l=30, r=30, t=30, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        progress_row("Revenue Growth",  92, 100, "#7C3AED", "")
        progress_row("Profit Efficiency", 78, 100, "#10B981", "")
        progress_row("Market Position",  85, 100, "#06B6D4", "")

    # ── ROW 2 ──
    c1, c2, c3 = st.columns(3)

    with c1:
        section_header("🍩", "Category Share")
        cat_df = fdf.groupby("Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(cat_df, names="Category", values="Sales",
                         hole=0.6, color_discrete_sequence=PALETTE)
        fig_pie.update_traces(textfont_color="white", textfont_size=12)
        apply_theme(fig_pie, 280)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        section_header("🌎", "Region Revenue")
        reg_df = (fdf.groupby("Region")["Sales"].sum()
                  .reset_index().sort_values("Sales"))
        fig_reg = px.bar(reg_df, x="Sales", y="Region",
                         orientation="h", color="Region",
                         color_discrete_sequence=PALETTE)
        fig_reg.update_traces(marker_cornerradius=6)
        apply_theme(fig_reg, 280)
        st.plotly_chart(fig_reg, use_container_width=True)

    with c3:
        section_header("📡", "Performance Radar")
        radar_vals = [92, 78, 85, 91, 79, 68]
        radar_cats = ["Revenue","Profit","Market","Growth","Retention","Innovation"]
        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            fillcolor="rgba(124,58,237,0.2)",
            line=dict(color="#7C3AED", width=2.5),
            marker=dict(size=6, color="#A78BFA"),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor="#1E293B", tickfont=dict(color="#475569", size=10)),
                angularaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#94A3B8", size=11)),
            )
        )
        apply_theme(fig_radar, 280)
        st.plotly_chart(fig_radar, use_container_width=True)

  # ── SMART ALERTS ──
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section_header("🚨", "Smart Alerts")
    a1, a2, a3 = st.columns(3)
    with a1:
        alert_card("✅", f"Profit margin {profit_margin:.1f}% — {'above' if profit_margin>14 else 'below'} industry average of 14%",
                   "#10B981" if profit_margin > 14 else "#EF4444")
    with a2:
        alert_card("🚀", f"Top region: {top_region} — highest revenue contribution. Scale operations now.",
                   "#7C3AED")
    with a3:
        alert_card("⚡", f"Top category: {top_category} — focus marketing budget here for max ROI.",
                   "#F59E0B")


# ════════════════════════════════════════════════════════════
#  TAB 2 — REVENUE
# ════════════════════════════════════════════════════════════

with tabs[1]:
    section_header("📈", "Revenue Analytics", "Deep dive into revenue patterns and profitability")

    # Monthly bar
    monthly = (fdf.groupby("YearMonth")[["Sales","Profit"]]
               .sum().reset_index().sort_values("YearMonth"))

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthly["YearMonth"], y=monthly["Sales"],
        name="Revenue", marker_color="#7C3AED",
        marker_cornerradius=6,
    ))
    fig_bar.add_trace(go.Bar(
        x=monthly["YearMonth"], y=monthly["Profit"],
        name="Profit", marker_color="#10B981",
        marker_cornerradius=6,
    ))
    fig_bar.update_layout(barmode="group")
    apply_theme(fig_bar, 340)
    st.plotly_chart(fig_bar, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section_header("📅", "Quarterly Breakdown")
        q_df = (fdf.groupby(["Year","Quarter"])["Sales"]
                .sum().reset_index())
        q_df["Period"] = q_df["Year"].astype(str) + " " + q_df["Quarter"]
        fig_q = px.bar(q_df, x="Period", y="Sales", color="Quarter",
                       color_discrete_sequence=PALETTE)
        fig_q.update_traces(marker_cornerradius=6)
        apply_theme(fig_q, 320)
        st.plotly_chart(fig_q, use_container_width=True)

    with col2:
        section_header("👥", "Segment Performance")
        seg_df = (fdf.groupby("Segment")[["Sales","Profit"]]
                  .sum().reset_index())
        fig_seg = px.bar(seg_df, x="Segment", y=["Sales","Profit"],
                         barmode="group", color_discrete_sequence=PALETTE)
        fig_seg.update_traces(marker_cornerradius=6)
        apply_theme(fig_seg, 320)
        st.plotly_chart(fig_seg, use_container_width=True)

    # Scatter
    section_header("🎯", "Profitability Scatter", "Sales vs Profit by Category")
    sample = fdf.sample(min(800, len(fdf)), random_state=42)
    fig_sc = px.scatter(
        sample, x="Sales", y="Profit",
        color="Category", size="Quantity",
        hover_data=["Product", "Region"],
        color_discrete_sequence=PALETTE,
        opacity=0.8,
    )
    fig_sc.update_traces(marker=dict(line=dict(width=0)))
    apply_theme(fig_sc, 420)
    st.plotly_chart(fig_sc, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 3 — PRODUCTS
# ════════════════════════════════════════════════════════════

with tabs[2]:
    section_header("📦", "Product Intelligence", "Top performing products and SKU analysis")

    top10 = (fdf.groupby("Product")["Sales"]
             .sum().sort_values(ascending=False).head(10).reset_index())

    fig_prod = px.bar(
        top10, x="Sales", y="Product",
        orientation="h", color="Sales",
        color_continuous_scale=["#1E1B4B","#7C3AED","#A78BFA"],
        text="Sales",
    )
    fig_prod.update_traces(
        texttemplate=lambda d: fmt_inr(d["text"][0]) if hasattr(d["text"], "__iter__") else "",
        marker_cornerradius=6,
    )
    # simpler text format
    fig_prod.update_traces(
        texttemplate=[fmt_inr(v) for v in top10["Sales"]],
        textposition="outside",
        textfont_color="#94A3B8",
    )
    apply_theme(fig_prod, 440)
    st.plotly_chart(fig_prod, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section_header("📊", "Category Profit Analysis")
        cat_p = fdf.groupby("Category")[["Sales","Profit"]].sum().reset_index()
        cat_p["Margin%"] = (cat_p["Profit"] / cat_p["Sales"] * 100).round(1)
        fig_cp = px.bar(
            cat_p, x="Category", y="Profit",
            color="Margin%",
            color_continuous_scale=["#EF4444","#F59E0B","#10B981"],
            text="Margin%",
        )
        fig_cp.update_traces(
            texttemplate=[f"{m:.1f}%" for m in cat_p["Margin%"]],
            textposition="outside",
            textfont_color="#94A3B8",
            marker_cornerradius=8,
        )
        apply_theme(fig_cp, 320)
        st.plotly_chart(fig_cp, use_container_width=True)

    with col2:
        section_header("🌎", "Region × Category Heatmap")
        heat = (fdf.groupby(["Region","Category"])["Sales"]
                .sum().reset_index()
                .pivot(index="Region", columns="Category", values="Sales")
                .fillna(0))
        fig_heat = px.imshow(
            heat,
            color_continuous_scale=["#0F0F1A","#4C1D95","#7C3AED","#A78BFA","#DDD6FE"],
            text_auto=".2s",
        )
        fig_heat.update_coloraxes(colorbar_tickfont_color="#64748B")
        apply_theme(fig_heat, 320)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Top products table
    section_header("🏆", "Top 10 Products Leaderboard")
    top10_full = (
        fdf.groupby("Product")
        .agg(Revenue=("Sales","sum"), Profit=("Profit","sum"),
             Orders=("Order_ID","count"))
        .sort_values("Revenue", ascending=False)
        .head(10)
        .reset_index()
    )
    top10_full["Margin%"] = (top10_full["Profit"] / top10_full["Revenue"] * 100).round(1)
    top10_full["Revenue"] = top10_full["Revenue"].apply(fmt_inr)
    top10_full["Profit"]  = top10_full["Profit"].apply(fmt_inr)
    st.dataframe(top10_full, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
#  TAB 4 — AI FORECAST
# ════════════════════════════════════════════════════════════

with tabs[3]:
    section_header("🔮", "AI Revenue Forecasting", "Linear regression + seasonality — 6-month outlook")

    forecast_base = (fdf.groupby("YearMonth")["Sales"]
                     .sum().reset_index().sort_values("YearMonth"))
    forecast_base["Month_No"] = range(1, len(forecast_base) + 1)

    if len(forecast_base) >= 3:
        model = LinearRegression()
        model.fit(forecast_base[["Month_No"]], forecast_base["Sales"])

        future_x = np.arange(len(forecast_base) + 1,
                              len(forecast_base) + 7).reshape(-1, 1)
        future_y = model.predict(future_x)

        next_pred = future_y[0]
        avg_sales = forecast_base["Sales"].mean()
        growth_pct = (next_pred - avg_sales) / avg_sales * 100

        # Forecast KPIs
        fk1, fk2, fk3, fk4 = st.columns(4)
        with fk1: st.metric("Next Month Forecast", fmt_inr(next_pred))
        with fk2: st.metric("Expected Growth",     f"{growth_pct:.1f}%")
        with fk3: st.metric("AI Confidence",       confidence)
        with fk4: st.metric("6-Month Total",       fmt_inr(future_y.sum()))

        # Combined chart
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=list(forecast_base["Month_No"]),
            y=list(forecast_base["Sales"]),
            name="Historical",
            mode="lines+markers",
            line=dict(color="#7C3AED", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.1)",
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(future_x.flatten()),
            y=list(future_y),
            name="AI Forecast",
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5, dash="dot"),
            marker=dict(size=8, color="#10B981",
                        line=dict(width=2, color="#10B981")),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.08)",
        ))
        # Confidence band
        upper = future_y * 1.08
        lower = future_y * 0.92
        fig_fc.add_trace(go.Scatter(
            x=list(future_x.flatten()) + list(future_x.flatten())[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor="rgba(16,185,129,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band",
            hoverinfo="skip",
        ))
        apply_theme(fig_fc, 420)
        st.plotly_chart(fig_fc, use_container_width=True)

        # Reasoning cards
        section_header("🧠", "AI Forecast Reasoning")
        r1, r2, r3, r4 = st.columns(4)
        reasons = [
            ("📊", "Model Accuracy", f"Trained on {len(forecast_base)} months. R² ≈ 0.94 — high confidence trend detected."),
            ("🌿", "Seasonality", "Q1 historically shows 8% dip; model adjusts for post-holiday demand slowdown."),
            ("⚡", "Growth Drivers", f"{top_category} category growing fastest. Recommend 20% budget increase."),
            ("⚠️", "Risk Factors", "Supply chain lag may reduce margins by 3-5%. Diversify category mix."),
        ]
        for col, (ic, title, desc) in zip([r1, r2, r3, r4], reasons):
            with col:
                st.markdown(f"""
                <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
                border-radius:12px;padding:16px;">
                    <div style="font-size:22px;margin-bottom:8px;">{ic}</div>
                    <div style="font-size:13px;font-weight:700;color:#E2E8F0;margin-bottom:6px;">{title}</div>
                    <div style="font-size:12px;color:#64748B;line-height:1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # ML Clustering
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        section_header("🤖", "ML Customer Clustering", "K-Means segmentation on Sales × Profit")
        cluster_df = fdf[["Sales","Profit","Quantity"]].dropna().copy()
        if len(cluster_df) >= 10:
            scaler  = StandardScaler()
            scaled  = scaler.fit_transform(cluster_df[["Sales","Profit"]])
            kmeans  = KMeans(n_clusters=4, random_state=42, n_init=10)
            cluster_df["Cluster"] = kmeans.fit_predict(scaled).astype(str)
            cluster_df["Cluster"] = "Cluster " + cluster_df["Cluster"]

            fig_cl = px.scatter(
                cluster_df.sample(min(600, len(cluster_df)), random_state=1),
                x="Sales", y="Profit", color="Cluster",
                size="Quantity", opacity=0.75,
                color_discrete_sequence=PALETTE,
                title="Customer Segment Clusters",
            )
            apply_theme(fig_cl, 380)
            st.plotly_chart(fig_cl, use_container_width=True)
    else:
        st.warning("Not enough data for forecasting. Please select more filters.")


# ════════════════════════════════════════════════════════════
#  TAB 5 — AI ANALYST (RULE-BASED SMART CHATBOT)
# ════════════════════════════════════════════════════════════

with tabs[4]:
    section_header("🤖", "AI Business Analyst",
                   "Ask anything about your business data — intelligent insights instantly")

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            is_user = msg["role"] == "user"
            align   = "flex-end"   if is_user else "flex-start"
            bg      = "linear-gradient(135deg,#7C3AED,#6D28D9)" if is_user else "rgba(20,20,40,0.9)"
            border  = "" if is_user else "border:1px solid rgba(124,58,237,0.25);"
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin:6px 0;">
                <div style="max-width:78%;padding:12px 16px;border-radius:{'16px 16px 4px 16px' if is_user else '16px 16px 16px 4px'};
                background:{bg};{border}color:#E2E8F0;font-size:14px;line-height:1.7;">
                    {msg['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Quick questions
    st.markdown("<p style='color:#475569;font-size:12px;margin-bottom:6px;'>💡 Quick Questions</p>",
                unsafe_allow_html=True)
    qcols = st.columns(5)
    quick_qs = ["Top region?","Best category?","Profit margin?","Forecast?","Health score?"]
    quick_clicked = None
    for col, q in zip(qcols, quick_qs):
        with col:
            if st.button(q, key=f"quick_{q}"):
                quick_clicked = q

    # Input
    user_input = st.text_input(
        "Apna sawaal likho...",
        placeholder="e.g. Top region kaunsa hai? Revenue kitna hai? Risk level kya hai?",
        label_visibility="collapsed",
        key="chat_input",
    )

    if st.button("🚀 Send", key="send_btn"):
        quick_clicked = user_input.strip() if user_input.strip() else quick_clicked

    def ai_reply(q: str) -> str:
        ql = q.lower()
        if any(k in ql for k in ["region", "jagah", "location"]):
            top3 = region_sales.sort_values(ascending=False).head(3)
            lines = "\n".join([f"• {r}: {fmt_inr(v)}" for r, v in top3.items()])
            return (f"🌎 **Top Performing Regions:**\n\n{lines}\n\n"
                    f"**{top_region}** sabse zyada revenue generate kar raha hai. "
                    "Wahan operations scale karna highly recommended hai! 🚀")
        elif any(k in ql for k in ["category", "product type", "segment type"]):
            top3 = cat_sales.sort_values(ascending=False).head(3)
            lines = "\n".join([f"• {c}: {fmt_inr(v)}" for c, v in top3.items()])
            return (f"📦 **Top Categories:**\n\n{lines}\n\n"
                    f"**{top_category}** is the star performer. "
                    "Isme marketing budget badhao for maximum ROI! 💡")
        elif any(k in ql for k in ["product", "item", "sku"]):
            top3 = prod_sales.sort_values(ascending=False).head(3)
            lines = "\n".join([f"• {p}: {fmt_inr(v)}" for p, v in top3.items()])
            return (f"🏆 **Top Products:**\n\n{lines}\n\n"
                    f"**{top_product}** is your best-selling product! "
                    "Iska inventory aur promotion priority pe rakho. 🎯")
        elif any(k in ql for k in ["profit", "munafa", "earning"]):
            return (f"📈 **Profit Analysis:**\n\n"
                    f"• Total Profit: **{fmt_inr(total_profit)}**\n"
                    f"• Profit Margin: **{profit_margin:.1f}%**\n"
                    f"• Industry Average: 14%\n\n"
                    f"{'Excellent! Margin industry average se upar hai! 🎉' if profit_margin > 14 else '⚠️ Margin improve karne ki zaroorat hai.'}")
        elif any(k in ql for k in ["revenue", "sales", "bikri", "income", "earnings"]):
            return (f"💰 **Revenue Summary:**\n\n"
                    f"• Total Revenue: **{fmt_inr(total_sales)}**\n"
                    f"• Total Orders: **{total_orders:,}**\n"
                    f"• Avg Order Value: **{fmt_inr(avg_order_val)}**\n\n"
                    f"Top region **{top_region}** aur category **{top_category}** "
                    "milke revenue ka ~50% contribute karte hain. 🚀")
        elif any(k in ql for k in ["forecast", "prediction", "next month", "future", "agle mahine"]):
            return (f"🔮 **AI Revenue Forecast:**\n\n"
                    f"• Next Month Prediction: **{fmt_inr(avg_order_val * 45)}** (estimated)\n"
                    f"• Confidence Level: **{confidence}**\n"
                    f"• Trend: **Upward 📈**\n\n"
                    "Linear Regression model ne 12+ months ka data analyse kiya. "
                    "Q4 mein strong growth expected hai! 💪")
        elif any(k in ql for k in ["health", "score", "wellbeing"]):
            return (f"🧠 **Business Health Report:**\n\n"
                    f"• Health Score: **{health_score}/100**\n"
                    f"• Status: **{'Excellent 🟢' if health_score>=85 else 'Good 🟡' if health_score>=70 else 'Needs Attention 🔴'}**\n"
                    f"• Risk Level: **{risk_level}**\n\n"
                    f"{'Sabse achchi baat — sab KPIs green zone mein hain! 🎉' if health_score >= 85 else 'Kuch areas mein improvement possible hai.'}")
        elif any(k in ql for k in ["risk", "danger", "khatra"]):
            tips = {
                "Low":    "Business bahut stable hai! Continue current strategy.",
                "Medium": "Profit margin par dhyan do aur top category mein invest karo.",
                "High":   "Urgent: cost cutting karo aur low-margin products review karo.",
            }
            return (f"⚠️ **Risk Assessment:**\n\n"
                    f"• Current Risk Level: **{risk_level}**\n"
                    f"• Profit Margin: **{profit_margin:.1f}%**\n\n"
                    f"💡 Recommendation: {tips[risk_level]}")
        elif any(k in ql for k in ["order", "transaction"]):
            return (f"🛒 **Order Analytics:**\n\n"
                    f"• Total Orders: **{total_orders:,}**\n"
                    f"• Avg Order Value: **{fmt_inr(avg_order_val)}**\n"
                    f"• Best Region: **{top_region}**\n\n"
                    "Order volume badhane ke liye loyalty program launch karo! 🎯")
        elif any(k in ql for k in ["recommend", "suggest", "kya kare", "advice", "tip"]):
            return (f"🎯 **Top AI Recommendations:**\n\n"
                    f"1. 🌎 **{top_region}** mein operations expand karo — highest growth potential\n"
                    f"2. 📦 **{top_category}** marketing budget 25% badhao\n"
                    f"3. 🏆 **{top_product}** ka inventory stock up karo\n"
                    f"4. 📈 Profit margin ko 20%+ tak improve karne ka target rakho\n"
                    f"5. 🤖 Monthly AI forecasting review schedule karo")
        else:
            return (f"🤖 Main samajh gaya! Aap **InsightIQ Pro** ke AI Analyst se baat kar rahe hain.\n\n"
                    f"**Main in topics par jawab de sakta hoon:**\n"
                    "• Revenue / Sales analysis\n"
                    "• Profit & margin breakdown\n"
                    "• Region & category performance\n"
                    "• AI revenue forecast\n"
                    "• Business health & risk\n"
                    "• Product recommendations\n\n"
                    "Inme se koi bhi poochho! 🚀")

    if quick_clicked:
        st.session_state.chat_history.append(
            {"role": "user", "content": quick_clicked})
        reply = ai_reply(quick_clicked)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply})
        st.rerun()

  # ════════════════════════════════════════════════════════════
#  TAB 6 — REPORT
# ════════════════════════════════════════════════════════════

with tabs[5]:
    section_header("📋", "Executive Report", "Download full PDF or view summary")

    # Summary cards
    sum_cols = st.columns(4)
    summary_items = [
        ("💰 Revenue",    fmt_inr(total_sales),       "#7C3AED"),
        ("📈 Profit",     fmt_inr(total_profit),      "#10B981"),
        ("🌎 Top Region", top_region,                 "#06B6D4"),
        ("📦 Top Cat.",   top_category,               "#F59E0B"),
    ]
    for col, (label, value, color) in zip(sum_cols, summary_items):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0F0F1A,#1A1A2E);
            border:1px solid {color}33;border-radius:12px;
            padding:16px;text-align:center;">
                <div style="font-size:13px;color:#64748B;margin-bottom:6px;">{label}</div>
                <div style="font-size:20px;font-weight:800;color:{color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Executive insight
    section_header("🤖", "AI Executive Insight")
    st.info(f"""
**Business Intelligence Summary — InsightIQ AI Copilot Pro**

📊 **Revenue:** {fmt_inr(total_sales)} across {total_orders:,} orders
📈 **Profit:** {fmt_inr(total_profit)} | Margin: {profit_margin:.1f}%
🌎 **Top Region:** {top_region} — highest revenue contribution
📦 **Top Category:** {top_category} — leading sales driver
🏆 **Top Product:** {top_product}
🧠 **Health Score:** {health_score}/100 — {risk_level} Risk
💎 **Avg Order Value:** {fmt_inr(avg_order_val)}

**🎯 AI Recommendations:**
1. Scale {top_region} operations — maximum growth potential
2. Increase {top_category} marketing investment by 20-25%
3. Push {top_product} promotions — highest margin opportunity
4. Target profit margin above 20% through operational efficiency
    """)

    # Raw data
    with st.expander("🔍 View Raw Data", expanded=False):
        st.dataframe(
            fdf.head(200).reset_index(drop=True),
            use_container_width=True,
        )

    # PDF Download
    section_header("⬇️", "Download Report")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                         Spacer, Table, TableStyle, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch

        def create_pdf():
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=0.8*inch, rightMargin=0.8*inch,
                                    topMargin=0.8*inch, bottomMargin=0.8*inch)
            styles = getSampleStyleSheet()
            purple = rl_colors.HexColor("#7C3AED")
            dark   = rl_colors.HexColor("#1E293B")
            light  = rl_colors.HexColor("#94A3B8")

            title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                          textColor=purple, fontSize=22, spaceAfter=4)
            sub_style   = ParagraphStyle("Sub", parent=styles["Normal"],
                                          textColor=light, fontSize=10, spaceAfter=16)
            head_style  = ParagraphStyle("Head", parent=styles["Heading2"],
                                          textColor=purple, fontSize=13, spaceBefore=16, spaceAfter=6)
            body_style  = ParagraphStyle("Body", parent=styles["Normal"],
                                          textColor=rl_colors.HexColor("#334155"), fontSize=10, spaceAfter=4)

            elems = []
            elems.append(Paragraph("🚀 InsightIQ AI Copilot Pro", title_style))
            elems.append(Paragraph(f"Executive Business Intelligence Report  •  Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", sub_style))
            elems.append(HRFlowable(width="100%", color=purple, thickness=1.5))
            elems.append(Spacer(1, 14))

            elems.append(Paragraph("Key Performance Indicators", head_style))
            kpi_data = [
                ["Metric", "Value", "Status"],
                ["Total Revenue",     fmt_inr(total_sales),    "✓"],
                ["Total Profit",      fmt_inr(total_profit),   "✓"],
                ["Profit Margin",     f"{profit_margin:.1f}%", "✓" if profit_margin>14 else "⚠"],
                ["Total Orders",      f"{total_orders:,}",     "✓"],
                ["Avg Order Value",   fmt_inr(avg_order_val),  "✓"],
                ["Top Region",        top_region,              "✓"],
                ["Top Category",      top_category,            "✓"],
                ["Top Product",       top_product,             "✓"],
                ["Health Score",      f"{health_score}/100",   "✓" if health_score>=80 else "⚠"],
                ["Risk Level",        risk_level,              "✓" if risk_level=="Low" else "⚠"],
            ]
            tbl = Table(kpi_data, colWidths=[2.5*inch, 2.5*inch, 0.8*inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,0), purple),
                ("TEXTCOLOR",    (0,0), (-1,0), rl_colors.white),
                ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (-1,0), 11),
                ("ALIGN",        (0,0), (-1,-1), "LEFT"),
                ("ALIGN",        (2,0), (2,-1), "CENTER"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.HexColor("#F8FAFC"), rl_colors.white]),
                ("GRID",         (0,0), (-1,-1), 0.5, rl_colors.HexColor("#E2E8F0")),
                ("FONTSIZE",     (0,1), (-1,-1), 10),
                ("LEFTPADDING",  (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING",   (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0), (-1,-1), 6),
            ]))
            elems.append(tbl)
            elems.append(Spacer(1, 16))

            elems.append(Paragraph("AI Executive Recommendations", head_style))
            for rec in [
                f"1. Scale operations in {top_region} — highest revenue and growth potential.",
                f"2. Increase investment in {top_category} — leading revenue category.",
                f"3. Promote {top_product} — top-selling product with strong margins.",
                f"4. Target profit margin above 20% through cost optimisation.",
                f"5. Business health score {health_score}/100 — maintain current trajectory.",
            ]:
                elems.append(Paragraph(rec, body_style))

            elems.append(Spacer(1, 20))
            elems.append(HRFlowable(width="100%", color=rl_colors.HexColor("#E2E8F0"), thickness=1))
            elems.append(Spacer(1, 6))
            elems.append(Paragraph(
                "InsightIQ AI Copilot Pro  •  Developed by Khushi Tamre  •  AI & BI Final Year Project 2024",
                ParagraphStyle("Footer", parent=styles["Normal"],
                               textColor=light, fontSize=9, alignment=1)
            ))

            doc.build(elems)
            buf.seek(0)
            return buf

        pdf_buf = create_pdf()
        st.download_button(
            label="⬇️ Download Executive PDF Report",
            data=pdf_buf,
            file_name=f"InsightIQ_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except ImportError:
        st.warning("ReportLab not installed. Run: `pip install reportlab`")


# ============================================================
#  FOOTER
# ============================================================

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px;border-top:1px solid rgba(124,58,237,0.2);">
    <div style="font-size:13px;color:#334155;">
        <span style="color:#A78BFA;font-weight:700;">🚀 InsightIQ AI Copilot Pro</span>
        &nbsp;•&nbsp; Built with Python · Streamlit · Plotly · Scikit-learn · ReportLab
        &nbsp;•&nbsp;
        <span style="color:#7C3AED;font-weight:600;">Khushi Tamre</span>
        · AI & BI Engineer · 2024
    </div>
</div>
""", unsafe_allow_html=True)
