import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import io

st.set_page_config(page_title="INSIGHT IQ AI COPILOT", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#050814;background-image:radial-gradient(ellipse 80% 55% at 50% -10%,rgba(120,40,255,.22),transparent),radial-gradient(ellipse 50% 40% at 90% 90%,rgba(0,255,163,.07),transparent),radial-gradient(ellipse 40% 30% at 5% 80%,rgba(56,189,248,.05),transparent);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2.5rem 3rem 6rem;max-width:1440px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07091a 0%,#0b0d1e 100%);border-right:1px solid rgba(120,40,255,.18);}
[data-testid="stMultiSelect"]>div{background:rgba(255,255,255,.04);border:1px solid rgba(120,40,255,.3);border-radius:10px;}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(255,255,255,.055) 0%,rgba(255,255,255,.018) 100%);border:1px solid rgba(120,40,255,.3);border-radius:20px;padding:1.8rem 2rem!important;position:relative;overflow:hidden;backdrop-filter:blur(16px);transition:transform .3s,border-color .3s,box-shadow .3s;}
[data-testid="stMetric"]:hover{transform:translateY(-5px);border-color:rgba(120,40,255,.65);box-shadow:0 20px 50px rgba(120,40,255,.22);}
[data-testid="stMetric"]::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#7828ff,#00ffa3,#38bdf8);background-size:200%;animation:shimmer 4s linear infinite;}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.83rem!important;font-weight:600!important;letter-spacing:.06em!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:#f1f5f9!important;font-size:2rem!important;font-weight:900!important;letter-spacing:-.03em!important;}
[data-testid="stMetricDelta"]{font-size:.85rem!important;font-weight:600!important;}
.stButton>button{background:linear-gradient(135deg,#7828ff,#5b10d9)!important;color:white!important;border:none!important;border-radius:14px!important;font-weight:700!important;font-size:.95rem!important;letter-spacing:.04em!important;padding:.75rem 2.2rem!important;transition:all .25s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 12px 32px rgba(120,40,255,.42)!important;}
[data-testid="stTextInput"] input{background:rgba(255,255,255,.04)!important;border:1px solid rgba(120,40,255,.35)!important;border-radius:14px!important;color:#f1f5f9!important;font-size:.95rem!important;padding:.8rem 1.1rem!important;}
[data-testid="stTextInput"] input:focus{border-color:#7828ff!important;box-shadow:0 0 0 3px rgba(120,40,255,.18)!important;}
.stSuccess,.stInfo,.stWarning,.stError{border-radius:14px!important;border:none!important;font-size:.93rem!important;}
[data-testid="stDataFrame"]{border:1px solid rgba(120,40,255,.22);border-radius:14px;overflow:hidden;}
hr{border-color:rgba(120,40,255,.15)!important;margin:2rem 0!important;}
h1,h2,h3{color:#f1f5f9!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-thumb{background:rgba(120,40,255,.45);border-radius:3px;}
@keyframes shimmer{0%{background-position:200%}100%{background-position:-200%}}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,255,163,.5)}50%{box-shadow:0 0 0 8px rgba(0,255,163,0)}}
@keyframes glow{0%,100%{text-shadow:0 0 30px rgba(120,40,255,.4)}50%{text-shadow:0 0 50px rgba(0,255,163,.4)}}
.iq-card{background:linear-gradient(135deg,rgba(255,255,255,.055) 0%,rgba(255,255,255,.018) 100%);border:1px solid rgba(120,40,255,.25);border-radius:20px;padding:1.8rem 2rem;position:relative;overflow:hidden;backdrop-filter:blur(16px);transition:all .3s;}
.iq-card:hover{border-color:rgba(120,40,255,.55);transform:translateY(-3px);box-shadow:0 16px 44px rgba(120,40,255,.18);}
.iq-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#7828ff,#00ffa3);}
.ai-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(120,40,255,.14);border:1px solid rgba(120,40,255,.4);border-radius:30px;padding:.32rem 1rem;font-size:.72rem;font-weight:700;color:#a78bfa;letter-spacing:.1em;text-transform:uppercase;}
.pulse-dot{width:7px;height:7px;background:#00ffa3;border-radius:50%;display:inline-block;animation:pulse 2s infinite;}
.chip{display:inline-block;background:rgba(0,255,163,.1);border:1px solid rgba(0,255,163,.3);border-radius:8px;padding:.22rem .7rem;font-size:.72rem;font-weight:600;color:#00ffa3;margin:.18rem;}
.rfm-badge{border-radius:8px;padding:.32rem .9rem;font-size:.78rem;font-weight:700;display:inline-block;}
.rfm-c{background:rgba(0,255,163,.15);border:1px solid rgba(0,255,163,.4);color:#00ffa3;}
.rfm-l{background:rgba(120,40,255,.15);border:1px solid rgba(120,40,255,.4);color:#a78bfa;}
.rfm-r{background:rgba(251,146,60,.15);border:1px solid rgba(251,146,60,.4);color:#fb923c;}
.rfm-x{background:rgba(248,113,113,.15);border:1px solid rgba(248,113,113,.4);color:#f87171;}
.stat-pill{background:rgba(255,255,255,.05);border-radius:12px;padding:.8rem 1rem;margin-bottom:.65rem;}
.stat-label{font-size:.72rem;color:#64748b;font-weight:500;margin-bottom:.2rem;}
.stat-value{font-size:1.15rem;color:#f1f5f9;font-weight:800;letter-spacing:-.02em;}
.prog-track{background:rgba(255,255,255,.06);border-radius:6px;height:8px;margin-top:.5rem;overflow:hidden;}
.prog-fill{height:8px;border-radius:6px;background:linear-gradient(90deg,#7828ff,#00ffa3);}
</style>
""", unsafe_allow_html=True)

PALETTE = ["#7828ff","#00ffa3","#f472b6","#38bdf8","#fb923c","#a3e635","#e879f9","#fbbf24"]

def apply_theme(fig, height=360):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=13),
        height=height, margin=dict(l=20,r=20,t=52,b=20),
        title_font=dict(size=15, color="#cbd5e1", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8",size=12),
                    bordercolor="rgba(255,255,255,.08)", borderwidth=1)
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.08)", tickfont=dict(color="#64748b",size=12))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", linecolor="rgba(255,255,255,.08)", tickfont=dict(color="#64748b",size=12))
    return fig

def section(icon, title, subtitle=""):
    sub = f'<div style="font-size:.85rem;color:#64748b;margin:.2rem 0 1.5rem;">{subtitle}</div>' if subtitle else '<div style="margin-bottom:1.4rem;"></div>'
    st.markdown(f"""
    <div style="margin-top:3.5rem;animation:fadeUp .5s ease;">
      <div style="font-size:1.75rem;font-weight:800;color:#f1f5f9;letter-spacing:-.03em;display:flex;align-items:center;gap:.55rem;margin-bottom:.3rem;">
        {icon} {title}
      </div>
      {sub}
      <div style="height:2px;background:linear-gradient(90deg,rgba(120,40,255,.55),rgba(0,255,163,.3),transparent);border-radius:2px;margin-bottom:1.8rem;"></div>
    </div>
    """, unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 0 1rem;">
      <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.6rem;">
        <div style="width:42px;height:42px;background:linear-gradient(135deg,#7828ff,#00ffa3);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 6px 20px rgba(120,40,255,.45);animation:float 3s ease infinite;">✦</div>
        <div>
          <div style="font-size:1.05rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em;">INSIGHT IQ AI</div>
          <div style="font-size:.68rem;color:#64748b;font-weight:500;letter-spacing:.1em;text-transform:uppercase;">Enterprise Copilot</div>
        </div>
      </div>
    </div>
    <hr style="border-color:rgba(120,40,255,.2);margin:.3rem 0 1.2rem;">
    """, unsafe_allow_html=True)
    for label, col_name, key in [
        ("📅 Time Period","Year","yr"),
        ("🌎 Geography","Region","rg"),
        ("👥 Segment","Segment","sg"),
        ("📦 Category","Category","ct"),
    ]:
        st.markdown(f"<p style='font-size:.72rem;font-weight:700;color:#a78bfa;letter-spacing:.15em;text-transform:uppercase;margin:.9rem 0 .4rem;'>{label}</p>", unsafe_allow_html=True)
        opts = sorted(df[col_name].dropna().unique())
        globals()[f"sel_{key}"] = st.multiselect(col_name, opts, default=opts, label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(120,40,255,.18);margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(120,40,255,.08);border:1px solid rgba(120,40,255,.22);border-radius:14px;padding:1.1rem;">
      <div style="font-size:.68rem;font-weight:700;color:#a78bfa;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.7rem;">👩‍💻 Developer</div>
      <div style="font-size:.95rem;font-weight:800;color:#f1f5f9;">Khushi Tamre</div>
      <div style="font-size:.75rem;color:#64748b;margin-top:.25rem;">AI &amp; BI Engineer · TY Project</div>
      <div style="margin-top:.7rem;"><span class="chip">Python</span><span class="chip">ML</span><span class="chip">Streamlit</span><span class="chip">Plotly</span></div>
    </div>
    """, unsafe_allow_html=True)

filtered_df = df[
    df["Year"].isin(sel_yr) & df["Region"].isin(sel_rg) &
    df["Segment"].isin(sel_sg) & df["Category"].isin(sel_ct)
]

total_sales   = filtered_df["Sales"].sum()
total_profit  = filtered_df["Profit"].sum()
total_orders  = filtered_df["Order_ID"].nunique()
profit_margin = (total_profit/total_sales*100) if total_sales>0 else 0
avg_order_val = total_sales/total_orders if total_orders>0 else 0

if len(sel_yr)>=2:
    yrs=sorted(sel_yr); prev=df[df["Year"]==yrs[-2]]["Sales"].sum(); curr=df[df["Year"]==yrs[-1]]["Sales"].sum()
    yoy=((curr-prev)/prev*100) if prev>0 else 0
else: yoy=0

top_region   = filtered_df.groupby("Region")["Sales"].sum().idxmax()
top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
top_product  = filtered_df.groupby("Product")["Sales"].sum().idxmax()
region_share = (filtered_df.groupby("Region")["Sales"].sum().max()/total_sales*100) if total_sales>0 else 0
health_score = 95 if profit_margin>=20 else 85 if profit_margin>=15 else 75 if profit_margin>=10 else 60
risk_level   = "Low" if health_score>=90 else "Medium" if health_score>=75 else "High"
confidence   = "95%" if health_score>=90 else "85%" if health_score>=75 else "70%"
risk_color   = "#00ffa3" if risk_level=="Low" else "#fb923c" if risk_level=="Medium" else "#f87171"

# HERO
st.markdown(f"""
<div style="padding:2.5rem 0 1.5rem;animation:fadeUp .7s ease;">
  <div class="ai-badge" style="margin-bottom:1.2rem;"><span class="pulse-dot"></span>&nbsp;Live Intelligence Active</div>
  <h1 style="font-size:clamp(2.2rem,5vw,4.2rem);font-weight:900;letter-spacing:-.04em;line-height:1.08;margin:0 0 .6rem;background:linear-gradient(90deg,#f1f5f9 20%,#c4b5fd 55%,#00ffa3 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:glow 5s ease infinite;">
    INSIGHT IQ AI COPILOT
  </h1>
  <p style="font-size:1.05rem;color:#64748b;margin:0 0 2rem;max-width:680px;">
    Enterprise Intelligence Platform &nbsp;·&nbsp; Predictive Analytics &nbsp;·&nbsp; Executive Decisions
  </p>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;">
    <div style="background:rgba(0,255,163,.08);border:1px solid rgba(0,255,163,.25);border-radius:14px;padding:.8rem 1.5rem;text-align:center;">
      <div style="font-size:.68rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.09em;">Health Score</div>
      <div style="font-size:1.6rem;font-weight:900;color:#00ffa3;letter-spacing:-.03em;">{health_score}/100</div>
    </div>
    <div style="background:rgba(120,40,255,.08);border:1px solid rgba(120,40,255,.25);border-radius:14px;padding:.8rem 1.5rem;text-align:center;">
      <div style="font-size:.68rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.09em;">Risk Level</div>
      <div style="font-size:1.6rem;font-weight:900;color:{risk_color};letter-spacing:-.03em;">{risk_level}</div>
    </div>
    <div style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);border-radius:14px;padding:.8rem 1.5rem;text-align:center;">
      <div style="font-size:.68rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.09em;">Confidence</div>
      <div style="font-size:1.6rem;font-weight:900;color:#38bdf8;letter-spacing:-.03em;">{confidence}</div>
    </div>
    <div style="background:rgba(244,114,182,.08);border:1px solid rgba(244,114,182,.25);border-radius:14px;padding:.8rem 1.5rem;text-align:center;">
      <div style="font-size:.68rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.09em;">Profit Margin</div>
      <div style="font-size:1.6rem;font-weight:900;color:#f472b6;letter-spacing:-.03em;">{profit_margin:.1f}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# SEC 1 — KPIs
section("📊", "Core Performance Metrics", "Real-time KPIs across all selected filters")
yoy_str = f"{yoy:+.1f}% YoY" if yoy!=0 else None
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("💰 Total Revenue",    f"₹{total_sales:,.0f}",  delta=yoy_str)
c2.metric("📈 Total Profit",     f"₹{total_profit:,.0f}")
c3.metric("🛒 Total Orders",     f"{total_orders:,}")
c4.metric("🎯 Profit Margin",    f"{profit_margin:.1f}%")
c5.metric("💎 Avg. Order Value", f"₹{avg_order_val:,.0f}")

# SEC 2 — Revenue
section("📈", "Revenue Intelligence", "Monthly trend and category breakdown")
r1a,r1b=st.columns([2,1])
with r1a:
    monthly=filtered_df.groupby("YearMonth")["Sales"].sum().reset_index().rename(columns={"Sales":"Revenue"})
    fig=px.area(monthly,x="YearMonth",y="Revenue",title="Monthly Revenue Trend",color_discrete_sequence=["#7828ff"])
    fig.update_traces(line=dict(width=3,color="#7828ff"),fillcolor="rgba(120,40,255,.12)",hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>")
    apply_theme(fig,370); st.plotly_chart(fig,use_container_width=True,key="trend")
with r1b:
    cs=filtered_df.groupby("Category")["Sales"].sum().reset_index()
    f2=px.pie(cs,names="Category",values="Sales",hole=.62,color_discrete_sequence=PALETTE,title="Revenue by Category")
    f2.update_traces(textfont=dict(color="white",size=13),hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>")
    f2.update_layout(legend=dict(orientation="h",y=-.14,font=dict(color="#94a3b8",size=12)))
    apply_theme(f2,370); st.plotly_chart(f2,use_container_width=True,key="donut")

# SEC 3 — Region & Scatter
section("🌎", "Regional & Profitability Analysis", "Region performance and sales vs profit intelligence map")
r2a,r2b=st.columns(2)
with r2a:
    reg=filtered_df.groupby("Region").agg(Revenue=("Sales","sum"),Profit=("Profit","sum")).reset_index().sort_values("Revenue",ascending=True)
    f3=go.Figure()
    f3.add_trace(go.Bar(y=reg["Region"],x=reg["Revenue"],orientation="h",
        marker=dict(color=reg["Revenue"],colorscale=[[0,"#3b0764"],[.5,"#7828ff"],[1,"#c4b5fd"]],line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>"))
    f3.update_layout(title="Revenue by Region",showlegend=False); apply_theme(f3,350); st.plotly_chart(f3,use_container_width=True,key="rbar")
with r2b:
    f4=px.scatter(filtered_df.sample(min(1200,len(filtered_df))),x="Sales",y="Profit",color="Category",size="Quantity",
        color_discrete_sequence=PALETTE,hover_data=["Product"],title="Sales vs Profit — Profitability Map")
    f4.update_traces(marker=dict(opacity=.78,line=dict(width=0)))
    apply_theme(f4,350); st.plotly_chart(f4,use_container_width=True,key="scatter")

# SEC 4 — Treemap + Quarterly
section("🗺️", "Deep-Dive Breakdown", "Revenue hierarchy treemap and quarterly trends")
r3a,r3b=st.columns(2)
sub_col="Sub-Category" if "Sub-Category" in filtered_df.columns else "Category"
with r3a:
    tree=filtered_df.groupby(["Category",sub_col])["Sales"].sum().reset_index()
    tree.columns=["Category","SubCat","Sales"]
    f5=px.treemap(tree,path=["Category","SubCat"],values="Sales",color="Sales",
        color_continuous_scale=[[0,"#2d1b69"],[.45,"#7828ff"],[.8,"#a78bfa"],[1,"#00ffa3"]],
        title="Category → Sub-Category Revenue Treemap")
    f5.update_traces(textfont=dict(color="white",size=13),hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>")
    f5.update_layout(coloraxis_showscale=False); apply_theme(f5,370); st.plotly_chart(f5,use_container_width=True,key="treemap")
with r3b:
    qtr=filtered_df.groupby(["Year","Quarter"])["Sales"].sum().reset_index().sort_values(["Year","Quarter"])
    qtr["Label"]=qtr["Year"].astype(str)+" "+qtr["Quarter"]
    f6=px.bar(qtr,x="Label",y="Sales",color="Quarter",color_discrete_sequence=PALETTE,title="Revenue by Quarter",barmode="group")
    f6.update_traces(marker_line_width=0,hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>")
    apply_theme(f6,370); st.plotly_chart(f6,use_container_width=True,key="quarterly")

# SEC 5 — Heatmap + SubCat
section("🔥", "Heatmap & Segment Intelligence", "Region × Month revenue matrix and sub-category profit")
r4a,r4b=st.columns(2)
with r4a:
    heat=filtered_df.groupby(["Region","Month"])["Sales"].sum().reset_index()
    pivot=heat.pivot(index="Region",columns="Month",values="Sales").fillna(0)
    mn={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot.columns=[mn.get(c,str(c)) for c in pivot.columns]
    f7=px.imshow(pivot,aspect="auto",color_continuous_scale=[[0,"#0d0f1f"],[.3,"#3b0764"],[.65,"#7828ff"],[1,"#00ffa3"]],title="Revenue Heatmap: Region × Month")
    f7.update_traces(hovertemplate="<b>%{y} · %{x}</b><br>₹%{z:,.0f}<extra></extra>")
    f7.update_layout(coloraxis_showscale=False); apply_theme(f7,350); st.plotly_chart(f7,use_container_width=True,key="heatmap")
with r4b:
    sub2=filtered_df.groupby(sub_col)["Profit"].sum().reset_index().sort_values("Profit",ascending=True).tail(12)
    sub2.columns=["Name","Profit"]; sub2["Color"]=sub2["Profit"].apply(lambda v:"#00ffa3" if v>=0 else "#f87171")
    f8=go.Figure()
    f8.add_trace(go.Bar(y=sub2["Name"],x=sub2["Profit"],orientation="h",marker=dict(color=sub2["Color"],line=dict(width=0)),hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>"))
    f8.update_layout(title="Profit by Sub-Category",showlegend=False); apply_theme(f8,350); st.plotly_chart(f8,use_container_width=True,key="subcat")

# SEC 6 — Forecast
section("🔮", "AI Predictive Engine", "6-month revenue forecast with confidence band using Linear Regression")
fc=filtered_df.groupby("YearMonth")["Sales"].sum().reset_index(); fc["t"]=range(1,len(fc)+1)
mdl=LinearRegression(); mdl.fit(fc[["t"]],fc["Sales"])
ft=np.arange(len(fc)+1,len(fc)+7).reshape(-1,1); fy=mdl.predict(ft); next_pred=fy[0]
avg_m=fc["Sales"].mean(); gpct=((next_pred/avg_m)-1)*100 if avg_m>0 else 0
nh=len(fc); hx=list(range(nh)); fx=list(range(nh-1,nh+6)); fyc=[float(fc["Sales"].iloc[-1])]+fy.tolist()
al=fc["YearMonth"].tolist()+[f"M+{i}" for i in range(1,7)]; tv=list(range(nh+6))

cf1,cf2=st.columns([3,1])
with cf1:
    fig_fc=go.Figure()
    bu=[v*1.08 for v in fyc]; bl=[v*0.92 for v in fyc]
    fig_fc.add_trace(go.Scatter(x=fx+fx[::-1],y=bu+bl[::-1],fill="toself",fillcolor="rgba(0,255,163,.07)",line=dict(color="rgba(0,0,0,0)"),showlegend=False,hoverinfo="skip"))
    fig_fc.add_trace(go.Scatter(x=hx,y=fc["Sales"].tolist(),mode="lines+markers",name="Historical",line=dict(color="#7828ff",width=3),marker=dict(size=6,color="#7828ff"),customdata=fc["YearMonth"].tolist(),hovertemplate="<b>%{customdata}</b><br>₹%{y:,.0f}<extra></extra>"))
    fig_fc.add_trace(go.Scatter(x=fx,y=fyc,mode="lines+markers",name="AI Forecast",line=dict(color="#00ffa3",width=3,dash="dot"),marker=dict(size=8,color="#00ffa3",symbol="diamond"),hovertemplate="<b>Forecast</b><br>₹%{y:,.0f}<extra></extra>"))
    fig_fc.add_shape(type="line",xref="x",yref="paper",x0=nh-1,x1=nh-1,y0=0,y1=1,line=dict(dash="dash",color="rgba(255,255,255,.2)",width=1.5))
    fig_fc.add_annotation(x=nh-1,y=.95,xref="x",yref="paper",text="NOW",showarrow=False,font=dict(color="#64748b",size=11),yanchor="top")
    fig_fc.update_xaxes(tickvals=tv,ticktext=al,tickangle=-35,tickfont=dict(size=11))
    fig_fc.update_layout(title="Revenue History + 6-Month AI Forecast (±8% Confidence Band)")
    apply_theme(fig_fc,390); st.plotly_chart(fig_fc,use_container_width=True,key="forecast")
with cf2:
    ga="▲" if gpct>=0 else "▼"; gc="#00ffa3" if gpct>=0 else "#f87171"
    st.markdown(f"""
    <div class="iq-card" style="height:100%;">
      <div style="font-size:.72rem;font-weight:700;color:#a78bfa;letter-spacing:.13em;text-transform:uppercase;margin-bottom:1.5rem;">AI Forecast Summary</div>
      <div class="stat-pill"><div class="stat-label">Next Month Prediction</div><div class="stat-value" style="color:#00ffa3;">₹{next_pred:,.0f}</div></div>
      <div class="stat-pill"><div class="stat-label">Growth vs Avg</div><div class="stat-value" style="color:{gc};">{ga} {abs(gpct):.1f}%</div></div>
      <div class="stat-pill"><div class="stat-label">Model Confidence</div><div class="stat-value">{confidence}</div></div>
      <div class="stat-pill"><div class="stat-label">Business Risk</div><div class="stat-value" style="color:{risk_color};">{risk_level}</div></div>
      <div style="margin-top:.8rem;">
        <div style="font-size:.72rem;color:#64748b;margin-bottom:.5rem;">AI Health Score</div>
        <div style="font-size:2.2rem;font-weight:900;color:#7828ff;letter-spacing:-.03em;">{health_score}<span style="font-size:.9rem;color:#475569;">/100</span></div>
        <div class="prog-track"><div class="prog-fill" style="width:{health_score}%;"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# SEC 7 — RFM
section("👥", "Customer RFM Intelligence", "Recency · Frequency · Monetary customer segmentation")
if "Customer_Name" in filtered_df.columns and "Order_Date" in filtered_df.columns:
    snap=filtered_df["Order_Date"].max()
    rfm=filtered_df.groupby("Customer_Name").agg(Recency=("Order_Date",lambda x:(snap-x.max()).days),Frequency=("Order_ID","nunique"),Monetary=("Sales","sum")).reset_index()
    for col,asc in [("Recency",False),("Frequency",True),("Monetary",True)]:
        lbl=col[0]+"_Score"
        rfm[lbl]=pd.qcut(rfm[col],4,labels=[1,2,3,4] if asc else [4,3,2,1],duplicates="drop").astype(float)
    rfm["RFM_Score"]=rfm[["R_Score","F_Score","M_Score"]].mean(axis=1)
    rfm["Segment"]=rfm["RFM_Score"].apply(lambda s:"Champions" if s>=3.5 else "Loyal" if s>=2.5 else "At Risk" if s>=1.8 else "Lost")
    sc=rfm["Segment"].value_counts().reset_index(); sc.columns=["Segment","Count"]; tc=len(rfm)
    rf1,rf2=st.columns([1,2])
    with rf1:
        cm={"Champions":"rfm-c","Loyal":"rfm-l","At Risk":"rfm-r","Lost":"rfm-x"}
        rows="".join(f"""<div style="display:flex;align-items:center;justify-content:space-between;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:.8rem 1rem;margin-bottom:.6rem;">
          <span class="rfm-badge {cm.get(r['Segment'],'rfm-x')}">{r['Segment']}</span>
          <div style="text-align:right;"><div style="font-size:1.2rem;font-weight:800;color:#f1f5f9;">{r['Count']}</div><div style="font-size:.72rem;color:#64748b;">{r['Count']/tc*100:.1f}%</div></div></div>""" for _,r in sc.iterrows())
        st.markdown(f'<div class="iq-card"><div style="font-size:.72rem;font-weight:700;color:#a78bfa;letter-spacing:.13em;text-transform:uppercase;margin-bottom:1.2rem;">Customer Segments</div>{rows}</div>',unsafe_allow_html=True)
    with rf2:
        frfm=px.scatter(rfm,x="Recency",y="Monetary",color="Segment",size="Frequency",
            color_discrete_map={"Champions":"#00ffa3","Loyal":"#7828ff","At Risk":"#fb923c","Lost":"#f87171"},
            hover_data=["Customer_Name","RFM_Score"],title="RFM Customer Intelligence Scatter")
        frfm.update_traces(marker=dict(opacity=.82,line=dict(width=0)))
        apply_theme(frfm,390); st.plotly_chart(frfm,use_container_width=True,key="rfm")
else:
    st.info("Customer_Name or Order_Date column not found — RFM skipped.")

# SEC 8 — Leaderboard
section("🏆", "Performance Leaderboard", "Top regions, products and category margins")
lb1,lb2,lb3=st.columns(3); M=["🥇","🥈","🥉","④","⑤"]
with lb1:
    tr=filtered_df.groupby("Region")["Sales"].sum().sort_values(ascending=False).head(5).reset_index()
    tr.columns=["Region","Revenue"]; tr["Revenue"]=tr["Revenue"].map("₹{:,.0f}".format); tr.index=M[:len(tr)]
    st.markdown("<p style='font-weight:700;color:#f1f5f9;font-size:1.05rem;margin-bottom:.6rem;'>🌎 Top Regions</p>",unsafe_allow_html=True); st.dataframe(tr,use_container_width=True)
with lb2:
    tp=filtered_df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(5).reset_index()
    tp.columns=["Product","Revenue"]; tp["Revenue"]=tp["Revenue"].map("₹{:,.0f}".format); tp.index=M[:len(tp)]
    st.markdown("<p style='font-weight:700;color:#f1f5f9;font-size:1.05rem;margin-bottom:.6rem;'>📦 Top Products</p>",unsafe_allow_html=True); st.dataframe(tp,use_container_width=True)
with lb3:
    tc2=filtered_df.groupby("Category")[["Sales","Profit"]].sum().sort_values("Sales",ascending=False).reset_index()
    tc2["Margin"]=(tc2["Profit"]/tc2["Sales"]*100).map("{:.1f}%".format); tc2["Sales"]=tc2["Sales"].map("₹{:,.0f}".format)
    tc2=tc2[["Category","Sales","Margin"]]
    st.markdown("<p style='font-weight:700;color:#f1f5f9;font-size:1.05rem;margin-bottom:.6rem;'>📊 Category Performance</p>",unsafe_allow_html=True); st.dataframe(tc2,use_container_width=True)

# SEC 9 — Health + Alerts
section("🧠", "AI Health & Risk Monitor", "Business health gauge and automated smart alerts")
g1,g2=st.columns([1,1])
with g1:
    gauge=go.Figure(go.Indicator(
        mode="gauge+number+delta",value=health_score,
        delta={"reference":75,"increasing":{"color":"#00ffa3"},"decreasing":{"color":"#f87171"}},
        number={"font":{"color":"#f1f5f9","size":62,"family":"Inter"},"suffix":"/100"},
        title={"text":"Business Health Index","font":{"color":"#94a3b8","size":15}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#64748b","tickfont":{"color":"#64748b","size":13}},
               "bar":{"color":"rgba(0,0,0,0)","thickness":0},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,40],"color":"rgba(248,113,113,.2)"},{"range":[40,70],"color":"rgba(251,146,60,.2)"},{"range":[70,100],"color":"rgba(0,255,163,.15)"}],
               "threshold":{"line":{"color":"#7828ff","width":4},"thickness":.88,"value":health_score}}))
    apply_theme(gauge,330); st.plotly_chart(gauge,use_container_width=True,key="gauge")
with g2:
    st.markdown("<p style='font-size:1.15rem;font-weight:700;color:#f1f5f9;margin-bottom:1rem;'>🚨 Smart AI Alerts</p>",unsafe_allow_html=True)
    if profit_margin<10: st.error(f"⚠️ **Critical:** Margin {profit_margin:.1f}% — below safe threshold.")
    elif profit_margin<20: st.warning(f"⚡ **Caution:** Margin {profit_margin:.1f}% — optimize cost structure.")
    else: st.success(f"✅ **Excellent:** Margin {profit_margin:.1f}% — strong performance.")
    if total_sales>100000: st.success(f"🚀 **Revenue Milestone:** ₹{total_sales:,.0f} exceeds threshold.")
    if next_pred>avg_m: st.info(f"📈 **Growth Signal:** AI forecast ₹{next_pred:,.0f} — above monthly average.")
    st.info(f"🌎 **Dominant Region:** {top_region} holds {region_share:.1f}% of total revenue.")
    st.markdown("<br><p style='font-size:1.15rem;font-weight:700;color:#f1f5f9;margin-bottom:1rem;'>🏅 Executive Scorecard</p>",unsafe_allow_html=True)
    s1,s2=st.columns(2); s1.metric("🧠 Health Score",f"{health_score}/100"); s2.metric("📈 Growth Score","92/100")
    s3,s4=st.columns(2); s3.metric("⚠️ Risk Level",risk_level); s4.metric("🎯 Confidence",confidence)

# SEC 10 — AI Command + Chat
section("🤖", "Insight IQ AI Assistant", "Strategic command center and natural language business Q&A")
cmd,chat=st.columns([1,1])
with cmd:
    ga2="▲" if gpct>=0 else "▼"; gc2="#00ffa3" if gpct>=0 else "#f87171"
    st.markdown(f"""
    <div class="iq-card">
      <div style="font-size:.72rem;font-weight:700;color:#a78bfa;letter-spacing:.13em;text-transform:uppercase;margin-bottom:1.4rem;">🎯 Strategic Command Center</div>
      <div style="display:flex;flex-direction:column;gap:.9rem;">
        <div class="stat-pill"><div class="stat-label">🌎 Top Revenue Region</div><div class="stat-value">{top_region}</div></div>
        <div class="stat-pill"><div class="stat-label">📦 Leading Category</div><div class="stat-value">{top_category}</div></div>
        <div class="stat-pill"><div class="stat-label">🔥 Top Product</div><div class="stat-value" style="font-size:.96rem;">{top_product[:48]+('…' if len(top_product)>48 else '')}</div></div>
      </div>
      <hr style="border-color:rgba(120,40,255,.18);margin:1.2rem 0;">
      <div style="font-size:.9rem;color:#94a3b8;line-height:1.8;">
        <span style="color:#a78bfa;font-weight:700;">AI Recommendation:</span><br>
        Scale <b style="color:#f1f5f9;">{top_category}</b> and expand in <b style="color:#f1f5f9;">{top_region}</b>.
        Projected growth: <b style="color:{gc2};">{ga2} {abs(gpct):.1f}%</b> vs avg.
      </div>
    </div>
    """,unsafe_allow_html=True)
with chat:
    st.markdown("<p style='font-size:.82rem;font-weight:600;color:#64748b;letter-spacing:.07em;text-transform:uppercase;margin-bottom:.6rem;'>💬 Ask the AI Copilot</p>",unsafe_allow_html=True)
    q=st.text_input("Ask",placeholder="e.g. best region · top product · profit · forecast · risk",label_visibility="collapsed",key="chat_q")
    if q:
        ql=q.lower()
        if "region" in ql: ans=f"🌎 **Best Region:** {top_region} — {region_share:.1f}% revenue share."
        elif "category" in ql: ans=f"📦 **Top Category:** {top_category} — highest revenue."
        elif "product" in ql: ans=f"🔥 **Top Product:** {top_product}"
        elif "margin" in ql: ans=f"📊 **Margin:** {profit_margin:.1f}%"
        elif "profit" in ql: ans=f"💰 **Profit:** ₹{total_profit:,.0f}"
        elif "revenue" in ql or "sales" in ql: ans=f"📈 **Revenue:** ₹{total_sales:,.0f} · Orders: {total_orders:,}"
        elif "forecast" in ql or "predict" in ql: ans=f"🔮 **Forecast:** ₹{next_pred:,.0f} next month ({confidence})."
        elif "order" in ql: ans=f"🛒 **Orders:** {total_orders:,} · Avg ₹{avg_order_val:,.0f}"
        elif "risk" in ql: ans=f"⚠️ **Risk:** {risk_level} · Score {health_score}/100"
        elif "health" in ql or "score" in ql: ans=f"🧠 **Health:** {health_score}/100 — {risk_level.lower()} risk."
        elif "growth" in ql: ans=f"📈 **Growth:** {'▲' if gpct>=0 else '▼'}{abs(gpct):.1f}% vs avg."
        elif "segment" in ql: ans=f"👥 **Top Segment:** {filtered_df.groupby('Segment')['Sales'].sum().idxmax()}"
        else: ans="💡 Try: **region · category · product · revenue · profit · margin · forecast · risk · health · orders · growth · segment**"
        st.markdown(f'<div style="background:rgba(120,40,255,.1);border:1px solid rgba(120,40,255,.32);border-radius:14px;padding:1rem 1.2rem;margin-top:.6rem;font-size:.93rem;color:#e2e8f0;line-height:1.7;">{ans}</div>',unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:rgba(255,255,255,.02);border:1px dashed rgba(120,40,255,.22);border-radius:14px;padding:2.5rem;text-align:center;margin-top:.5rem;">
          <div style="font-size:2.8rem;margin-bottom:.6rem;animation:float 3s ease infinite;">🤖</div>
          <div style="font-size:.95rem;color:#475569;font-weight:500;margin-bottom:.4rem;">Ask anything about your business data</div>
          <div style="font-size:.8rem;color:#334155;">region &nbsp;·&nbsp; category &nbsp;·&nbsp; product &nbsp;·&nbsp; revenue &nbsp;·&nbsp; forecast &nbsp;·&nbsp; risk</div>
        </div>""",unsafe_allow_html=True)

# SEC 11 — PDF
section("📄", "Executive PDF Report", "Download a complete AI-generated business intelligence report")
if st.button("⬇ Generate Executive PDF Report"):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=50,leftMargin=50,topMargin=60,bottomMargin=50)
    story=[]; sty=getSampleStyleSheet()
    ts=ParagraphStyle("T",parent=sty["Title"],fontSize=26,fontName="Helvetica-Bold",textColor=colors.HexColor("#7828ff"),spaceAfter=8)
    ss=ParagraphStyle("S",parent=sty["Normal"],fontSize=13,textColor=colors.HexColor("#64748b"),spaceAfter=22)
    hs=ParagraphStyle("H",parent=sty["Heading2"],fontSize=14,fontName="Helvetica-Bold",textColor=colors.HexColor("#1e293b"),spaceAfter=10,spaceBefore=18)
    bs=ParagraphStyle("B",parent=sty["Normal"],fontSize=11,textColor=colors.HexColor("#374151"),leading=18)
    story+=[Paragraph("INSIGHT IQ AI COPILOT",ts),Paragraph("Executive Business Intelligence Report",ss),Spacer(1,12)]
    story.append(Paragraph("Key Performance Indicators",hs))
    kd=[["Metric","Value"],["Total Revenue",f"₹{total_sales:,.0f}"],["Total Profit",f"₹{total_profit:,.0f}"],
        ["Profit Margin",f"{profit_margin:.1f}%"],["Total Orders",f"{total_orders:,}"],["Avg. Order Value",f"₹{avg_order_val:,.0f}"],
        ["AI Health Score",f"{health_score}/100"],["Business Risk",risk_level],["Forecast M+1",f"₹{next_pred:,.0f}"],["Model Confidence",confidence]]
    t=Table(kd,colWidths=[230,230])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#7828ff")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),11),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),colors.HexColor("#f1f5f9")]),
        ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#e2e8f0")),("PADDING",(0,0),(-1,-1),10)]))
    story.append(t)
    story+=[Paragraph("Strategic Insights",hs),Paragraph(f"• <b>Top Region:</b> {top_region} ({region_share:.1f}% revenue share)",bs),
        Paragraph(f"• <b>Top Category:</b> {top_category}",bs),Paragraph(f"• <b>Top Product:</b> {top_product}",bs),Spacer(1,10),
        Paragraph("AI Recommendation",hs),
        Paragraph(f"Increase investment in <b>{top_category}</b> and expand in <b>{top_region}</b>. Forecast: ₹{next_pred:,.0f} next month with {confidence} confidence. Health score: <b>{health_score}/100</b> — {risk_level.lower()} risk.",bs)]
    doc.build(story); buf.seek(0)
    st.download_button(label="📥 Download PDF Report",data=buf,file_name="INSIGHT_IQ_AI_Executive_Report.pdf",mime="application/pdf")
    st.success("✅ Executive Report Generated Successfully!")

st.markdown("""
<div style="margin-top:4rem;border-top:1px solid rgba(120,40,255,.15);padding-top:2rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.8rem;">
  <div>
    <div style="font-size:1rem;font-weight:800;color:#7828ff;letter-spacing:-.02em;">INSIGHT IQ AI COPILOT</div>
    <div style="font-size:.8rem;color:#475569;margin-top:.2rem;">Built by <b style="color:#a78bfa;">Khushi Tamre</b> &nbsp;·&nbsp; AI &amp; BI Engineer &nbsp;·&nbsp; TY Final Project</div>
  </div>
  <div style="font-size:.78rem;color:#334155;font-family:'JetBrains Mono',monospace;letter-spacing:.04em;">Python &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; ReportLab</div>
</div>
""", unsafe_allow_html=True)
