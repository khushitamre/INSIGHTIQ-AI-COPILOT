import { useState, useEffect, useRef } from "react";
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, RadarChart,
  PolarGrid, PolarAngleAxis, Radar,Cr
} from "recharts";

# // ─── SAMPLE DATA ────────────────────────────────────────────────────────────

const monthlyRevenue = [
  { month: "Jan", revenue: 420000, profit: 84000, orders: 312 },
  { month: "Feb", revenue: 380000, profit: 68400, orders: 287 },
  { month: "Mar", revenue: 510000, profit: 102000, orders: 398 },
  { month: "Apr", revenue: 490000, profit: 93100, orders: 372 },
  { month: "May", revenue: 620000, profit: 136400, orders: 481 },
  { month: "Jun", revenue: 580000, profit: 110200, orders: 453 },
  { month: "Jul", revenue: 710000, profit: 156200, orders: 541 },
  { month: "Aug", revenue: 680000, profit: 142800, orders: 519 },
  { month: "Sep", revenue: 750000, profit: 172500, orders: 578 },
  { month: "Oct", revenue: 820000, profit: 189200, orders: 634 },
  { month: "Nov", revenue: 910000, profit: 218400, orders: 701 },
  { month: "Dec", revenue: 980000, profit: 245000, orders: 756 },
];

const forecast = [
  { month: "Jan '25", revenue: 1050000, type: "forecast" },
  { month: "Feb '25", revenue: 1120000, type: "forecast" },
  { month: "Mar '25", revenue: 1190000, type: "forecast" },
];

const categoryData = [
  { name: "Electronics", value: 38, color: "#7C3AED" },
  { name: "Furniture", value: 24, color: "#06B6D4" },
  { name: "Clothing", value: 19, color: "#10B981" },
  { name: "Food & Bev", value: 12, color: "#F59E0B" },
  { name: "Sports", value: 7, color: "#EF4444" },
];

const regionData = [
  { region: "West", sales: 3200000, growth: 18 },
  { region: "East", sales: 2800000, growth: 12 },
  { region: "North", sales: 1900000, growth: 22 },
  { region: "South", sales: 2100000, growth: 9 },
  { region: "Central", sales: 1500000, growth: 31 },
];

const radarData = [
  { metric: "Revenue", score: 88 },
  { metric: "Profit", score: 72 },
  { metric: "Growth", score: 91 },
  { metric: "Retention", score: 79 },
  { metric: "Efficiency", score: 85 },
  { metric: "Innovation", score: 68 },
];

const topProducts = [
  { name: "MacBook Pro M3", revenue: 842000, margin: 22 },
  { name: "iPhone 15 Pro", revenue: 731000, margin: 28 },
  { name: "Samsung 4K TV", revenue: 612000, margin: 18 },
  { name: "Sony Headphones", revenue: 498000, margin: 35 },
  { name: "iPad Air", revenue: 421000, margin: 25 },
];

# // ─── HELPERS ─────────────────────────────────────────────────────────────────
def fmt(n):
    if n >= 1e7:
        return f"₹{n/1e7:.2f}Cr"
    elif n >= 1e5:
        return f"₹{n/1e5:.1f}L"
    else:
        return f"₹{n:,.0f}"

def fmt_short(n):
    if n >= 1e6:
        return f"₹{n/1e6:.1f}M"
    else:
        return f"₹{n/1000:.0f}K"

# // ─── ANIMATED NUMBER ─────────────────────────────────────────────────────────

function AnimatedNumber({ value, prefix = "", suffix = "", duration = 1500 }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = value / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= value) {
        setDisplay(value);
        clearInterval(timer);
      } else {
        setDisplay(Math.floor(start));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [value]);
  return (
    <span>
      {prefix}
      {display.toLocaleString()}
      {suffix}
    </span>
  );
}

# // ─── KPI CARD ────────────────────────────────────────────────────────────────

function KpiCard({ icon, label, value, sub, color, trend }) {
  return (
    <div
      style={{
        background: "linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 100%)",
        border: `1px solid ${color}33`,
        borderRadius: 16,
        padding: "20px 24px",
        position: "relative",
        overflow: "hidden",
        transition: "transform 0.2s, box-shadow 0.2s",
        cursor: "default",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-4px)";
        e.currentTarget.style.boxShadow = `0 12px 40px ${color}33`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* glow orb */}
      <div
        style={{
          position: "absolute",
          top: -30,
          right: -30,
          width: 100,
          height: 100,
          borderRadius: "50%",
          background: color,
          opacity: 0.08,
          filter: "blur(30px)",
        }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span style={{ fontSize: 22 }}>{icon}</span>
        <span style={{ color: "#94A3B8", fontSize: 13, fontWeight: 500, letterSpacing: 1, textTransform: "uppercase" }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: "#F1F5F9", lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 6 }}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 700,
            color: trend > 0 ? "#10B981" : "#EF4444",
            background: trend > 0 ? "#10B98120" : "#EF444420",
            padding: "2px 8px",
            borderRadius: 20,
          }}
        >
          {trend > 0 ? "▲" : "▼"} {Math.abs(trend)}%
        </span>
        <span style={{ color: "#64748B", fontSize: 12 }}>{sub}</span>
      </div>
    </div>
  );
}

# // ─── SECTION HEADER ──────────────────────────────────────────────────────────

function SectionHeader({ icon, title, subtitle }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
        <span style={{ fontSize: 22 }}>{icon}</span>
        <h2
          style={{
            margin: 0,
            fontSize: 20,
            fontWeight: 800,
            color: "#F1F5F9",
            letterSpacing: "-0.5px",
          }}
        >
          {title}
        </h2>
      </div>
      {subtitle && (
        <p style={{ margin: 0, color: "#64748B", fontSize: 13 }}>{subtitle}</p>
      )}
      <div
        style={{
          marginTop: 12,
          height: 2,
          background: "linear-gradient(90deg, #7C3AED, #06B6D4, transparent)",
          borderRadius: 2,
        }}
      />
    </div>
  );
}

# // ─── GLASS CARD ──────────────────────────────────────────────────────────────

function Card({ children, style = {} }) {
  return (
    <div
      style={{
        background: "rgba(15,15,26,0.8)",
        border: "1px solid rgba(124,58,237,0.2)",
        borderRadius: 16,
        padding: 24,
        backdropFilter: "blur(12px)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

# // ─── AI CHAT ─────────────────────────────────────────────────────────────────

function AiAssistant({ kpis }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "👋 Namaste! Main hoon **InsightIQ AI Analyst**. Aap mujhse business performance, trends, forecasting ya koi bhi data-related sawaal pooch sakte hain. Kya jaanna chahte hain?",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const ask = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput("");
    setMessages((p) => [...p, { role: "user", text: userMsg }]);
    setLoading(true);
    try {
      const systemPrompt = `You are InsightIQ, a world-class AI Business Analyst. 
Current business data:
- Total Annual Revenue: ₹${kpis.revenue.toLocaleString()}
- Total Profit: ₹${kpis.profit.toLocaleString()}
- Profit Margin: ${kpis.margin}%
- Total Orders: ${kpis.orders.toLocaleString()}
- Top Region: West (₹3.2Cr, +18% growth)
- Top Category: Electronics (38% share)
- Top Product: MacBook Pro M3 (₹8.42L revenue)
- Business Health Score: 87/100
- Risk Level: Low
- Next Month AI Forecast: ₹10.5Cr

Answer in a mix of Hindi and English (Hinglish) — professional yet friendly. 
Be data-specific. Give actionable insights. Use emojis strategically.
Keep answers concise but impactful (3-5 sentences max unless asked for detail).`;

      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: systemPrompt,
          messages: [{ role: "user", content: userMsg }],
        }),
      });
      const data = await res.json();
      const text = data.content?.map((c) => c.text || "").join("") || "Kuch error aaya. Please retry karein.";
      setMessages((p) => [...p, { role: "assistant", text }]);
    } catch {
      setMessages((p) => [...p, { role: "assistant", text: "⚠️ API error. Please try again." }]);
    }
    setLoading(false);
  };

  return (
    <Card>
      <SectionHeader icon="🤖" title="AI Business Analyst" subtitle="Claude-powered — real-time business intelligence" />
      <div
        style={{
          height: 340,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          paddingRight: 4,
          marginBottom: 16,
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              justifyContent: m.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                maxWidth: "82%",
                padding: "12px 16px",
                borderRadius: m.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                background:
                  m.role === "user"
                    ? "linear-gradient(135deg, #7C3AED, #6D28D9)"
                    : "rgba(30,30,50,0.9)",
                border: m.role === "assistant" ? "1px solid rgba(124,58,237,0.3)" : "none",
                color: "#E2E8F0",
                fontSize: 14,
                lineHeight: 1.6,
                whiteSpace: "pre-wrap",
              }}
            >
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: "flex", gap: 6, padding: "12px 16px" }}>
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: "#7C3AED",
                  animation: `bounce 1.2s ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div style={{ display: "flex", gap: 10 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="Apna sawaal poochein... (e.g. 'Top region kaunsa hai?')"
          style={{
            flex: 1,
            background: "rgba(30,30,50,0.9)",
            border: "1px solid rgba(124,58,237,0.4)",
            borderRadius: 12,
            padding: "12px 16px",
            color: "#E2E8F0",
            fontSize: 14,
            outline: "none",
          }}
        />
        <button
          onClick={ask}
          disabled={loading}
          style={{
            background: loading ? "#3D1F7A" : "linear-gradient(135deg, #7C3AED, #6D28D9)",
            border: "none",
            borderRadius: 12,
            padding: "12px 20px",
            color: "white",
            fontSize: 18,
            cursor: loading ? "not-allowed" : "pointer",
            transition: "opacity 0.2s",
          }}
        >
          {loading ? "⏳" : "🚀"}
        </button>
      </div>
      <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
        {["Top products?", "Revenue forecast?", "Risk analysis?", "Growth tips?"].map((q) => (
          <button
            key={q}
            onClick={() => { setInput(q); }}
            style={{
              background: "rgba(124,58,237,0.15)",
              border: "1px solid rgba(124,58,237,0.3)",
              borderRadius: 20,
              padding: "4px 14px",
              color: "#A78BFA",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            {q}
          </button>
        ))}
      </div>
    </Card>
  );
}


# // ─── CUSTOM TOOLTIP ──────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "#0F0F1A",
        border: "1px solid rgba(124,58,237,0.5)",
        borderRadius: 10,
        padding: "10px 16px",
        fontSize: 13,
        color: "#E2E8F0",
      }}
    >
      <p style={{ margin: "0 0 6px", color: "#A78BFA", fontWeight: 700 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ margin: "2px 0", color: p.color }}>
          {p.name}: {typeof p.value === "number" && p.value > 1000 ? fmtShort(p.value) : p.value}
        </p>
      ))}
    </div>
  );
};

# // ─── HEALTH GAUGE ────────────────────────────────────────────────────────────

function HealthGauge({ score }) {
  const angle = (score / 100) * 180 - 90;
  return (
    <div style={{ textAlign: "center", padding: "12px 0" }}>
      <svg viewBox="0 0 200 120" style={{ width: "100%", maxWidth: 240 }}>
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#EF4444" />
            <stop offset="50%" stopColor="#F59E0B" />
            <stop offset="100%" stopColor="#10B981" />
          </linearGradient>
        </defs>
        <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="#1E293B" strokeWidth="16" strokeLinecap="round" />
        <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="url(#gaugeGrad)" strokeWidth="16" strokeLinecap="round" />
        <line
          x1="100" y1="100"
          x2={100 + 65 * Math.cos((angle * Math.PI) / 180)}
          y2={100 + 65 * Math.sin((angle * Math.PI) / 180)}
          stroke="white" strokeWidth="3" strokeLinecap="round"
        />
        <circle cx="100" cy="100" r="6" fill="#7C3AED" />
        <text x="100" y="90" textAnchor="middle" fill="#F1F5F9" fontSize="22" fontWeight="800">{score}</text>
        <text x="100" y="110" textAnchor="middle" fill="#64748B" fontSize="11">HEALTH SCORE</text>
      </svg>
      <div style={{ marginTop: -8 }}>
        <span
          style={{
            background: score >= 85 ? "#10B98120" : score >= 70 ? "#F59E0B20" : "#EF444420",
            color: score >= 85 ? "#10B981" : score >= 70 ? "#F59E0B" : "#EF4444",
            padding: "4px 16px",
            borderRadius: 20,
            fontSize: 13,
            fontWeight: 700,
          }}
        >
          {score >= 85 ? "🟢 Excellent" : score >= 70 ? "🟡 Good" : "🔴 Needs Attention"}
        </span>
      </div>
    </div>
  );
}

# // ─── MAIN APP ─────────────────────────────────────────────────────────────────

export default function InsightIQPro() {
  const [activeTab, setActiveTab] = useState("overview");

  const totalRevenue = monthlyRevenue.reduce((s, m) => s + m.revenue, 0);
  const totalProfit = monthlyRevenue.reduce((s, m) => s + m.profit, 0);
  const totalOrders = monthlyRevenue.reduce((s, m) => s + m.orders, 0);
  const margin = ((totalProfit / totalRevenue) * 100).toFixed(1);

  const kpis = { revenue: totalRevenue, profit: totalProfit, orders: totalOrders, margin };

  const tabs = [
    { id: "overview", label: "📊 Overview" },
    { id: "revenue", label: "📈 Revenue" },
    { id: "products", label: "📦 Products" },
    { id: "forecast", label: "🔮 Forecast" },
    { id: "ai", label: "🤖 AI Analyst" },
  ];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #020209 0%, #0A0A1A 50%, #050510 100%)",
        fontFamily: "'Inter', 'Segoe UI', sans-serif",
        color: "#E2E8F0",
      }}
    >
      <style>{`
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0F0F1A; }
        ::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 2px; }
        @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
        .tab-btn:hover { background: rgba(124,58,237,0.2) !important; }
      `}</style>

      {/* ── HEADER ── */}
      <div
        style={{
          background: "rgba(10,10,26,0.95)",
          borderBottom: "1px solid rgba(124,58,237,0.3)",
          padding: "0 32px",
          position: "sticky",
          top: 0,
          zIndex: 100,
          backdropFilter: "blur(20px)",
        }}
      >
        <div style={{ maxWidth: 1400, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div
              style={{
                width: 38,
                height: 38,
                borderRadius: 10,
                background: "linear-gradient(135deg, #7C3AED, #06B6D4)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 20,
              }}
            >
              🚀
            </div>
            <div>
              <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: "-0.5px", background: "linear-gradient(90deg, #A78BFA, #06B6D4)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                InsightIQ Pro
              </div>
              <div style={{ fontSize: 10, color: "#475569", letterSpacing: 2, textTransform: "uppercase" }}>
                AI Business Intelligence
              </div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {tabs.map((t) => (
              <button
                key={t.id}
                className="tab-btn"
                onClick={() => setActiveTab(t.id)}
                style={{
                  background: activeTab === t.id ? "rgba(124,58,237,0.3)" : "transparent",
                  border: activeTab === t.id ? "1px solid rgba(124,58,237,0.6)" : "1px solid transparent",
                  borderRadius: 8,
                  padding: "6px 14px",
                  color: activeTab === t.id ? "#A78BFA" : "#64748B",
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#10B981", animation: "pulse 2s infinite" }} />
            <span style={{ fontSize: 12, color: "#10B981", fontWeight: 600 }}>Live Data</span>
          </div>
        </div>
      </div>

      {/* ── CONTENT ── */}
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "32px", animation: "fadeIn 0.4s ease" }}>

        {/* ══ OVERVIEW TAB ══ */}
        {activeTab === "overview" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
            {/* Hero */}
            <div style={{ textAlign: "center", padding: "12px 0 4px" }}>
              <h1 style={{ fontSize: 36, fontWeight: 900, letterSpacing: "-1px", margin: "0 0 8px", background: "linear-gradient(90deg, #A78BFA 0%, #06B6D4 50%, #10B981 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                Business Command Center
              </h1>
              <p style={{ color: "#475569", fontSize: 15, margin: 0 }}>
                FY 2024 • Real-time AI-powered analytics • Last updated: {new Date().toLocaleTimeString()}
              </p>
            </div>

            {/* KPI Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 16 }}>
              <KpiCard icon="💰" label="Annual Revenue" value={<AnimatedNumber value={totalRevenue} prefix="₹" />} sub="vs last year" trend={23} color="#7C3AED" />
              <KpiCard icon="📈" label="Net Profit" value={<AnimatedNumber value={totalProfit} prefix="₹" />} sub="vs last year" trend={31} color="#10B981" />
              <KpiCard icon="🛒" label="Total Orders" value={<AnimatedNumber value={totalOrders} />} sub="vs last year" trend={18} color="#06B6D4" />
              <KpiCard icon="🎯" label="Profit Margin" value={`${margin}%`} sub="industry avg 14%" trend={6} color="#F59E0B" />
              <KpiCard icon="⭐" label="Health Score" value="87/100" sub="excellent range" trend={9} color="#EF4444" />
            </div>

            {/* Row 2 */}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20 }}>
              {/* Area chart */}
              <Card>
                <SectionHeader icon="📊" title="Revenue & Profit Trend" subtitle="Monthly performance across FY 2024" />
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={monthlyRevenue}>
                    <defs>
                      <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#7C3AED" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#7C3AED" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="profGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="month" stroke="#475569" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#475569" tick={{ fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ color: "#94A3B8", fontSize: 13 }} />
                    <Area type="monotone" dataKey="revenue" name="Revenue" stroke="#7C3AED" fill="url(#revGrad)" strokeWidth={2.5} />
                    <Area type="monotone" dataKey="profit" name="Profit" stroke="#10B981" fill="url(#profGrad)" strokeWidth={2.5} />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>

              {/* Health + Radar */}
              <Card>
                <SectionHeader icon="🧠" title="Business Health" />
                <HealthGauge score={87} />
                <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
                  {[
                    { label: "Revenue Growth", val: 92, color: "#7C3AED" },
                    { label: "Profit Efficiency", val: 78, color: "#10B981" },
                    { label: "Market Position", val: 85, color: "#06B6D4" },
                  ].map((item) => (
                    <div key={item.label}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12, color: "#94A3B8" }}>
                        <span>{item.label}</span><span style={{ color: item.color }}>{item.val}%</span>
                      </div>
                      <div style={{ height: 5, background: "#1E293B", borderRadius: 3 }}>
                        <div style={{ height: "100%", width: `${item.val}%`, background: item.color, borderRadius: 3, transition: "width 1s ease" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            {/* Row 3 */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20 }}>
              {/* Donut */}
              <Card>
                <SectionHeader icon="🍩" title="Category Share" />
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={categoryData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} dataKey="value" paddingAngle={3}>
                      {categoryData.map((c, i) => <Cell key={i} fill={c.color} />)}
                    </Pie>
                    <Tooltip formatter={(v) => `${v}%`} contentStyle={{ background: "#0F0F1A", border: "1px solid #7C3AED33", borderRadius: 10, color: "#E2E8F0" }} />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
                  {categoryData.map((c) => (
                    <div key={c.name} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#94A3B8" }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: c.color }} />
                      {c.name} {c.value}%
                    </div>
                  ))}
                </div>
              </Card>

              {/* Region bars */}
              <Card>
                <SectionHeader icon="🌎" title="Region Performance" />
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={regionData} layout="vertical" barSize={14}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" horizontal={false} />
                    <XAxis type="number" stroke="#475569" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v / 1e6).toFixed(1)}M`} />
                    <YAxis type="category" dataKey="region" stroke="#475569" tick={{ fontSize: 12, fill: "#94A3B8" }} width={55} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="sales" name="Sales" radius={[0, 6, 6, 0]}>
                      {regionData.map((_, i) => <Cell key={i} fill={["#7C3AED", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"][i]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              {/* Radar */}
              <Card>
                <SectionHeader icon="📡" title="Performance Radar" />
                <ResponsiveContainer width="100%" height={240}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#1E293B" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: "#94A3B8", fontSize: 11 }} />
                    <Radar name="Score" dataKey="score" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.25} strokeWidth={2} />
                    <Tooltip contentStyle={{ background: "#0F0F1A", border: "1px solid #7C3AED33", borderRadius: 10, color: "#E2E8F0", fontSize: 13 }} />
                  </RadarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Alert strip */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
              {[
                { icon: "✅", text: "Profit margin 22.3% — above industry average of 14%", color: "#10B981", bg: "#10B98115" },
                { icon: "🚀", text: "AI forecast: +12% revenue growth expected next quarter", color: "#7C3AED", bg: "#7C3AED15" },
                { icon: "⚡", text: "West region showing strongest 18% YoY growth — scale now", color: "#F59E0B", bg: "#F59E0B15" },
              ].map((a, i) => (
                <div key={i} style={{ background: a.bg, border: `1px solid ${a.color}40`, borderRadius: 12, padding: "14px 16px", display: "flex", gap: 10, alignItems: "flex-start" }}>
                  <span style={{ fontSize: 18 }}>{a.icon}</span>
                  <span style={{ fontSize: 13, color: "#CBD5E1", lineHeight: 1.5 }}>{a.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══ REVENUE TAB ══ */}
        {activeTab === "revenue" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <SectionHeader icon="📈" title="Revenue Analytics" subtitle="Deep dive into revenue patterns and profitability" />
            <Card>
              <h3 style={{ color: "#94A3B8", margin: "0 0 16px", fontSize: 14, fontWeight: 600 }}>MONTHLY REVENUE VS PROFIT</h3>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={monthlyRevenue} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="month" stroke="#475569" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 12 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ color: "#94A3B8", fontSize: 13 }} />
                  <Bar dataKey="revenue" name="Revenue" fill="#7C3AED" radius={[6, 6, 0, 0]} barSize={22} />
                  <Bar dataKey="profit" name="Profit" fill="#10B981" radius={[6, 6, 0, 0]} barSize={22} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <Card>
                <h3 style={{ color: "#94A3B8", margin: "0 0 16px", fontSize: 14, fontWeight: 600 }}>QUARTERLY BREAKDOWN</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {[
                    { q: "Q1 (Jan–Mar)", rev: 1310000, growth: 8 },
                    { q: "Q2 (Apr–Jun)", rev: 1690000, growth: 15 },
                    { q: "Q3 (Jul–Sep)", rev: 2140000, growth: 24 },
                    { q: "Q4 (Oct–Dec)", rev: 2710000, growth: 32 },
                  ].map((q) => (
                    <div key={q.q} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ width: 80, fontSize: 13, color: "#64748B" }}>{q.q}</div>
                      <div style={{ flex: 1, height: 32, background: "#1E293B", borderRadius: 8, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${(q.rev / 2710000) * 100}%`, background: "linear-gradient(90deg, #7C3AED, #06B6D4)", borderRadius: 8, display: "flex", alignItems: "center", paddingLeft: 12 }}>
                          <span style={{ fontSize: 12, fontWeight: 700, color: "white" }}>{fmtShort(q.rev)}</span>
                        </div>
                      </div>
                      <div style={{ width: 50, fontSize: 12, color: "#10B981", fontWeight: 700, textAlign: "right" }}>+{q.growth}%</div>
                    </div>
                  ))}
                </div>
              </Card>
              <Card>
                <h3 style={{ color: "#94A3B8", margin: "0 0 16px", fontSize: 14, fontWeight: 600 }}>SEGMENT ANALYSIS</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  {[
                    { seg: "Consumer", rev: 4200000, pct: 56, color: "#7C3AED" },
                    { seg: "Corporate", rev: 2100000, pct: 28, color: "#06B6D4" },
                    { seg: "Home Office", rev: 1200000, pct: 16, color: "#10B981" },
                  ].map((s) => (
                    <div key={s.seg}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 13 }}>
                        <span style={{ color: "#CBD5E1", fontWeight: 600 }}>{s.seg}</span>
                        <span style={{ color: s.color }}>{fmtShort(s.rev)} ({s.pct}%)</span>
                      </div>
                      <div style={{ height: 8, background: "#1E293B", borderRadius: 4 }}>
                        <div style={{ height: "100%", width: `${s.pct}%`, background: s.color, borderRadius: 4 }} />
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

          {/* ══ FORECAST TAB ══ */}
        {activeTab === "forecast" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <SectionHeader icon="🔮" title="AI Revenue Forecasting" subtitle="Machine learning powered 3-month revenue prediction" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
              {forecast.map((f, i) => (
                <Card key={f.month} style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 12, color: "#64748B", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>{f.month}</div>
                  <div style={{ fontSize: 32, fontWeight: 900, color: "#A78BFA", marginBottom: 6 }}>{fmtShort(f.revenue)}</div>
                  <div style={{ fontSize: 12, color: "#10B981", fontWeight: 600 }}>
                    +{(((f.revenue - 980000) / 980000) * 100).toFixed(1)}% vs Dec '24
                  </div>
                  <div style={{ marginTop: 12, fontSize: 11, color: "#475569" }}>
                    Confidence: {95 - i * 5}%
                  </div>
                </Card>
              ))}
            </div>
            <Card>
              <h3 style={{ color: "#94A3B8", margin: "0 0 16px", fontSize: 14, fontWeight: 600 }}>HISTORICAL + FORECAST TREND</h3>
              <ResponsiveContainer width="100%" height={340}>
                <AreaChart data={[...monthlyRevenue.map(m => ({ ...m, type: "actual" })), ...forecast.map(f => ({ month: f.month, revenue: f.revenue, profit: null, orders: null, type: "forecast" }))]}>
                  <defs>
                    <linearGradient id="fg1" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#7C3AED" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#7C3AED" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="month" stroke="#475569" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="revenue" name="Revenue" stroke="#7C3AED" fill="url(#fg1)" strokeWidth={2.5} strokeDasharray={(d) => d?.type === "forecast" ? "6 4" : ""} />
                </AreaChart>
              </ResponsiveContainer>
              <div style={{ marginTop: 12, display: "flex", gap: 16, justifyContent: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#94A3B8" }}>
                  <div style={{ width: 20, height: 2, background: "#7C3AED" }} /> Historical Data
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#94A3B8" }}>
                  <div style={{ width: 20, height: 2, background: "#7C3AED", borderTop: "2px dashed #7C3AED" }} /> AI Forecast
                </div>
              </div>
            </Card>
            <Card>
              <h3 style={{ color: "#94A3B8", margin: "0 0 16px", fontSize: 14, fontWeight: 600 }}>🤖 AI FORECAST REASONING</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {[
                  { icon: "📊", title: "Linear Regression Model", desc: "12 months of historical data used to train growth pattern. R² score: 0.94 — very high confidence in trend." },
                  { icon: "🌿", title: "Seasonal Adjustment", desc: "Q1 traditionally shows 8% dip; model accounts for post-holiday slowdown in Jan–Feb." },
                  { icon: "⚡", title: "Growth Drivers", desc: "Electronics category growing 28% YoY. West region expansion expected to add ₹2.1Cr incremental revenue." },
                  { icon: "⚠️", title: "Risk Factors", desc: "Supply chain disruptions could reduce Electronics supply by 15%. Hedge with Furniture and Sports categories." },
                ].map((item) => (
                  <div key={item.title} style={{ background: "rgba(124,58,237,0.07)", borderRadius: 10, padding: "14px 16px", border: "1px solid rgba(124,58,237,0.15)" }}>
                    <div style={{ fontSize: 18, marginBottom: 6 }}>{item.icon}</div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "#E2E8F0", marginBottom: 4 }}>{item.title}</div>
                    <div style={{ fontSize: 12, color: "#64748B", lineHeight: 1.6 }}>{item.desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* ══ AI TAB ══ */}
        {activeTab === "ai" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <AiAssistant kpis={kpis} />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
              {[
                { icon: "🎯", title: "Top Recommendation", text: `Scale West region operations — highest growth potential at +18% YoY with ₹3.2Cr revenue base.` },
                { icon: "💡", title: "Quick Win", text: "Sony Headphones has 35% margin — highest in portfolio. Increase marketing budget by 20% for immediate ROI." },
                { icon: "⚠️", title: "Watch Out", text: "Food & Beverage category declining at -3% QoQ. Consider product line refresh or discontinuation strategy." },
              ].map((r) => (
                <div key={r.title} style={{ background: "rgba(124,58,237,0.08)", border: "1px solid rgba(124,58,237,0.2)", borderRadius: 12, padding: "18px 20px" }}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>{r.icon}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "#E2E8F0", marginBottom: 6 }}>{r.title}</div>
                  <div style={{ fontSize: 13, color: "#64748B", lineHeight: 1.6 }}>{r.text}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* FOOTER */}
        <div style={{ marginTop: 40, paddingTop: 24, borderTop: "1px solid rgba(124,58,237,0.2)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontSize: 13, color: "#475569" }}>
            <span style={{ color: "#A78BFA", fontWeight: 700 }}>InsightIQ Pro</span> — Built with React • Recharts • Claude AI
          </div>
          <div style={{ fontSize: 12, color: "#334155" }}>© 2024 Khushi Tamre • AI & BI Engineer</div>
        </div>
      </div>
    </div>
  );
}


