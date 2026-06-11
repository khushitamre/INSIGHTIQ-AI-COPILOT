import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 1. PAGE SETUP & GLOBAL STYLING
# --------------------------------------------------
st.set_page_config(
    page_title="INSIGHT IQ AI COPILOT v2.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk Glossy Theme CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #090d16, #0f172a, #1e293b);
    }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 255, 163, 0.5);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 55px !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00E5FF, #00FFA3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(0,229,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. OPTIMIZED DATA PIPELINE
# --------------------------------------------------
@st.cache_data
def load_and_process_data():
    # Production dataset simulation fallback standard structure
    try:
        df = pd.read_csv("business_data.csv")
    except FileNotFoundError:
        # Generate sample robust mock data if file doesn't exist for test run
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=24, freq="M")
        mock_data = {
            "Order_Date": np.repeat(dates, 10),
            "Region": np.random.choice(["North", "South", "East", "West"], 240),
            "Segment": np.random.choice(["Consumer", "Corporate", "Home Office"], 240),
            "Category": np.random.choice(["Technology", "Furniture", "Office Supplies"], 240),
            "Product": np.random.choice(["Pro Laptop", "Ergo Chair", "Smart Phone", "Desk Eco", "Paper Pack"], 240),
            "Sales": np.random.uniform(5000, 45000, 240),
            "Profit": np.random.uniform(-1000, 15000, 240),
            "Quantity": np.random.randint(1, 10, 240),
            "Order_ID": [f"ORD-{i:04d}" for i in range(1, 241)],
            "Customer_Name": [f"Client-{i}" for i in range(1, 241)]
        }
        df = pd.DataFrame(mock_data)

    df.columns = df.columns.str.strip()
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df = df.dropna(subset=["Order_Date"]).sort_values("Order_Date")
        df["Year"] = df["Order_Date"].dt.year
        df["Month"] = df["Order_Date"].dt.month
        df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    
    return df

df = load_and_process_data()

# --------------------------------------------------
# 3. INTERACTIVE SIDEBAR NAVIGATION & FILTERS
# --------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#00E5FF;'>⚡ CONTROL PANEL</h2>", unsafe_allow_html=True)
    st.write("Refine global scope analytics:")
    
    years = sorted(df["Year"].dropna().unique())
    selected_years = st.multiselect("Select Financial Year", years, default=years)

    regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.multiselect("Target Regions", regions, default=regions)

    segments = sorted(df["Segment"].dropna().unique())
    selected_segments = st.multiselect("Market Segments", segments, default=segments)

# Data filtering logic execution
filtered_df = df[
    (df["Year"].isin(selected_years)) &
    (df["Region"].isin(selected_regions)) &
    (df["Segment"].isin(selected_segments))
]

# Ensure app context remains stable if filter result is empty
if filtered_df.empty:
    st.warning("⚠️ No data available for selected filter combinations. Resetting to full view.")
    filtered_df = df

# --------------------------------------------------
# 4. EXECUTIVE HERO BANNER
# --------------------------------------------------
st.markdown("""
<div style="text-align:center; padding: 20px 0px 40px 0px;">
    <h1 class="main-title">INSIGHT IQ AI COPILOT v2.0</h1>
    <p style="color:#B8C1CC; font-size:18px; max-width:800px; margin:auto;">
        Advanced Enterprise Engine transforming multi-dimensional operational metrics into real-time forecasting, health audits, and strategic insights.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 5. CORE ANALYTICAL COMPUTATIONS
# --------------------------------------------------
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order_ID"].nunique()
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

# Calculated Business Metadata
top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax() if not filtered_df.empty else "N/A"
top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax() if not filtered_df.empty else "N/A"
top_product = filtered_df.groupby("Product")["Sales"].sum().idxmax() if not filtered_df.empty else "N/A"

# Health Score Evaluation Algorithm
if profit_margin >= 20: health_score = 95
elif profit_margin >= 15: health_score = 85
elif profit_margin >= 10: health_score = 75
else: health_score = 60

# --------------------------------------------------
# 6. DASHBOARD STRUCTURAL TABS
# --------------------------------------------------
tab_overview, tab_predictive, tab_ai_copilot, tab_reports = st.tabs([
    "📈 Enterprise Overview", 
    "🔮 Predictive Forecasting", 
    "🤖 Intelligent AI Copilot", 
    "📄 Governance & Reports"
])

# --------------------------------------------------
# TAB 1: ENTERPRISE OVERVIEW
# --------------------------------------------------
with tab_overview:
    # High-Level Metric Ribbon
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("💰 Total Revenue", f"₹{total_sales:,.2f}")
    m_col2.metric("📈 Net Profit", f"₹{total_profit:,.2f}")
    m_col3.metric("🛒 Volume Orders", f"{total_orders:,}")
    m_col4.metric("🎯 Profit Margin Ratio", f"{profit_margin:.2f}%")
    
    st.write("---")
    
    # Grid Visualizations
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.subheader("📊 Temporal Revenue Trend")
        monthly_sales = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
        fig_trend = px.line(monthly_sales, x="YearMonth", y="Sales", markers=True, template="plotly_dark")
        fig_trend.update_traces(line_color="#00E5FF", marker=dict(size=6, color="#00FFA3"))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with g_col2:
        st.subheader("📦 Market Share by Segment Category")
        category_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(category_sales, names="Category", values="Sales", hole=0.4, template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Cyan_r)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.write("---")
    
    # Secondary Analytical Matrices
    st.subheader("🎯 Profitability Deep-Dive Matrix")
    fig_scatter = px.scatter(
        filtered_df, x="Sales", y="Profit", color="Category", size="Quantity",
        hover_data=["Product", "Customer_Name"], template="plotly_dark",
        title="Cross-sectional Profit Margin Dispersion Map"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Performance Rankings Grid
    st.write("---")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("### 🌎 Top Regional Performers")
        reg_df = filtered_df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
        st.dataframe(reg_df, use_container_width=True)
    with r_col2:
        st.markdown("### 📦 High Volume Inventory Lines")
        prod_df = filtered_df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(5).reset_index()
        st.dataframe(prod_df, use_container_width=True)

# --------------------------------------------------
# TAB 2: PREDICTIVE FORECASTING
# --------------------------------------------------
with tab_predictive:
    st.subheader("🔮 Predictive Revenue Projections (Next 6 Months)")
    
    forecast_df = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
    if len(forecast_df) > 1:
        forecast_df["Month_Num"] = range(1, len(forecast_df) + 1)
        
        # Machine Learning Model Implementation
        X = forecast_df[["Month_Num"]]
        y = forecast_df["Sales"]
        model = LinearRegression().fit(X, y)
        
        # Extrapolate for next 6 steps
        future_idx = np.arange(len(forecast_df) + 1, len(forecast_df) + 7).reshape(-1, 1)
        future_preds = model.predict(future_idx)
        
        # Build dynamic presentation data frames
        future_months = [f"Month +{i}" for i in range(1, 7)]
        future_df = pd.DataFrame({"Month": future_months, "Predicted Revenue": future_preds})
        
        # Render visual prediction line
        fig_fc = px.line(future_df, x="Month", y="Predicted Revenue", markers=True, template="plotly_dark")
        fig_fc.update_traces(line_color="#00FFA3", marker=dict(size=8, color="#00E5FF"))
        st.plotly_chart(fig_fc, use_container_width=True)
        
        fc_col1, fc_col2 = st.columns(2)
        fc_col1.metric("🔮 Immediate Next Month Prediction", f"₹{future_preds[0]:,.2f}")
        fc_col2.metric("📊 Aggregated Historical Sample Scope", f"₹{y.sum():,.2f}")
    else:
        st.info("💡 Insufficient temporal trend depth to generate reliable linear predictive models. Please clear filters to expand data points.")

# --------------------------------------------------
# TAB 3: INTELLIGENT AI COPILOT
# --------------------------------------------------
with tab_ai_copilot:
    st.subheader("🤖 Context-Aware Strategic AI Assistant")
    st.write("Interact with your enterprise model directly to pull hidden metrics.")
    
    user_query = st.text_input("Formulate your query (e.g., 'What is our top region?', 'Show performance metrics')", key="copilot_query_box")
    
    if user_query:
        q = user_query.lower()
        st.markdown("#### **AI Core Agent Output:**")
        
        if "region" in q:
            st.success(f"🌎 **Market Vector Identification:** The dominant operations vector is situated in the **{top_region}** market tier.")
        elif "category" in q:
            st.success(f"📦 **Vertical Concentration:** Enterprise demand maps heaviest towards the **{top_category}** core portfolio component.")
        elif "product" in q:
            st.success(f"🔥 **SKU Penetration Velocity:** High-conviction product performance leader is identified as: **{top_product}**.")
        elif "profit" in q or "margin" in q:
            st.info(f"💰 **Financial Health Status:** Net margins computed to be **{profit_margin:.2f}%** totaling financial capture of **₹{total_profit:,.2f}**.")
        elif "revenue" in q or "sales" in q:
            st.info(f"📈 **Revenue Generation Matrix:** Total active top-line revenue generated equals **₹{total_sales:,.2f}** across specified filters.")
        else:
            st.warning("🤖 Core model requires strict parametric intents. Try inquiring regarding: **revenue**, **profit**, **region**, **category**, or **product** values.")

    st.write("---")
    
    # Gauge Health Matrix
    st.subheader("🧠 Enterprise Operational Health Scale")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Comprehensive Resilience Score", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00E5FF"},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ]
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

# --------------------------------------------------
# TAB 4: GOVERNANCE & REPORTS
# --------------------------------------------------
with tab_reports:
    st.subheader("📄 Automated Corporate PDF Engine")
    st.write("Generate pristine, single-click standard production executive summaries for board audits.")
    
    if st.button("🔧 Initialize Dynamic PDF Generation Pipeline", key="pdf_generation_trigger"):
        filename = "Insight_IQ_Executive_Report.pdf"
        c = canvas.Canvas(filename)
        
        # Meta configuration definitions
        c.setTitle("Insight IQ Governance Log")
        c.setFillColorRGB(0.05, 0.08, 0.15)
        c.rect(0, 0, 600, 900, fill=1)
        
        # Document Head Structural Settings
        c.setFillColorRGB(0.0, 0.9, 1.0)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, 800, "INSIGHT IQ AI COPILOT REPORT")
        
        c.setFillColorRGB(1.0, 1.0, 1.0)
        c.setFont("Helvetica", 12)
        c.drawString(50, 775, "Classification: Strict Executive Management Board Confidential")
        c.setStrokeColorRGB(0.0, 0.9, 1.0)
        c.line(50, 760, 550, 760)
        
        # Key Aggregations Reporting Data Layers
        c.drawString(50, 720, f"• Consolidated Gross Revenue: INR {total_sales:,.2f}")
        c.drawString(50, 695, f"• Retained Enterprise Net Profit: INR {total_profit:,.2f}")
        c.drawString(50, 670, f"• Computed Profit Run-Rate Margin: {profit_margin:.2f}%")
        c.drawString(50, 645, f"• Computed Health Index Metric: {health_score}/100")
        c.drawString(50, 620, f"• Alpha Regional Performance Hub: {top_region}")
        c.drawString(50, 595, f"• Dominant Consumer Focus Line: {top_category}")
        
        # Tactical Directives Segment Block
        c.setFillColorRGB(0.0, 1.0, 0.6)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 540, "Automated Strategic Recommendations Vector:")
        
        c.setFillColorRGB(1.0, 1.0, 1.0)
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(70, 515, f"1. Target market scaling activities inside localized {top_region} coordinates.")
        c.drawString(70, 495, f"2. Allocate expanded systemic liquidity lines to back robust {top_category} portfolios.")
        
        # Save and process stream deployment
        c.save()
        
        with open(filename, "rb") as f:
            st.download_button(
                label="⬇️ Download Verified Executive PDF Document",
                data=f,
                file_name=filename,
                mime="application/pdf"
            )
        st.success("✨ Automated Board Compilation Finished Successfully. Document Link Ready Below.")

    st.write("---")
    
    # Strategic Command Center Metrics Display
    st.subheader("🚀 Operational Risk Command Center")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.info(f"### 📈 Growth Blueprint\n\n* **Top Track:** {top_region}\n* **Leading Asset:** {top_category}\n\n*Action Policy:* Accelerate capital deployment into {top_category} lines immediately.")
    with rc2:
        risk_level = "Low" if profit_margin >= 15 else "Moderate" if profit_margin >= 10 else "High Critical"
        st.warning(f"### ⚠️ Threat Control Assessment\n\n* **Threat Level:** {risk_level}\n* **System Auditing Index:** {health_score}/100\n\n*Policy Stance:* Continuous algorithmic tracing of revenue fluctuations.")

    st.write("---")
    
    # Project Creator Credentials Core Signature
    st.subheader("👩‍💻 Core Architecture Team")
    st.markdown("""
    **Developer Lead:** Khushi Tamre  
    **Functional Role:** AI & Advanced Business Intelligence Architect  
    *Tech Stack Execution Mastery: Python Core, Streamlit Cloud Optimization, Vectorized Data Processing (Pandas/Numpy), Real-Time Visualization (Plotly Engine).*
    """)
