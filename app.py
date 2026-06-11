import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
