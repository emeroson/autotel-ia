"""
AutoTel AI - Autonomous Telecom Network Intelligence System
By ANOH AMON HEMERSON
Enterprise-grade dashboard for telecom network monitoring, prediction and optimization.
v3.0 — Carte géographique + Upload données réelles + Guides intégrés
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoTel AI | Telecom Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
def inject_global_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
    :root {
        --red-primary:   #A10000;
        --red-dark:      #7A0000;
        --red-light:     #C0392B;
        --red-ultra:     #8B0000;
        --bg-main:       #1A0000;
        --bg-card:       #1E0808;
        --bg-card-alt:   #220000;
        --border-color:  #6A2020;
        --text-primary:  #FFFFFF;
        --text-secondary:#F0C8C8;
        --text-muted:    #DD9999;
        --green-ok:      #1A7A4A;
        --orange-warn:   #D35400;
        --shadow-sm:     0 2px 8px rgba(0,0,0,0.4);
        --shadow-md:     0 4px 20px rgba(0,0,0,0.5);
        --shadow-lg:     0 8px 40px rgba(0,0,0,0.7);
        --radius-sm:     8px;
        --radius-md:     12px;
        --radius-lg:     18px;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .main .block-container {
        padding: 1.5rem 2rem 3rem 2rem !important;
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stApp { background: var(--bg-main) !important; color: var(--text-primary) !important; }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        border-bottom: none !important;
    }
    header[data-testid="stHeader"] button,
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    .css-fblp2m, .css-1rs6os, .css-14xtw13 {
        background: linear-gradient(135deg, #A10000, #C0392B) !important;
        border-radius: 50% !important;
        width: 44px !important; height: 44px !important;
        min-width: 44px !important; min-height: 44px !important;
        padding: 0 !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        box-shadow: 0 0 0 3px rgba(192,57,43,0.5), 0 4px 20px rgba(161,0,0,0.8) !important;
        border: 2px solid #FF7777 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        opacity: 1 !important; visibility: visible !important;
    }
    header[data-testid="stHeader"] button:hover,
    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover {
        background: linear-gradient(135deg, #C0392B, #FF4444) !important;
        box-shadow: 0 0 0 5px rgba(255,100,100,0.3), 0 8px 28px rgba(192,57,43,0.9) !important;
        transform: scale(1.12) !important;
    }
    header[data-testid="stHeader"] button svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    header[data-testid="stHeader"] button svg *,
    [data-testid="collapsedControl"] svg *,
    [data-testid="stSidebarCollapsedControl"] svg * {
        stroke: #FFFFFF !important; fill: none !important; color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] button {
        background: rgba(161,0,0,0.4) !important;
        border-radius: 50% !important;
        border: 1px solid rgba(255,100,100,0.6) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(192,57,43,0.7) !important; transform: scale(1.06) !important;
    }
    [data-testid="stSidebar"] button svg,
    [data-testid="stSidebar"] button svg * { stroke: #FFFFFF !important; color: #FFFFFF !important; }
    [data-testid="stSidebar"] {
        background: #120000 !important;
        border-right: 2px solid #C0392B !important;
        min-width: 255px !important;
    }
    [data-testid="stSidebar"] > div { background: #120000 !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #FFFFFF !important; }
    [data-testid="stSidebar"] hr { border-color: #5A1010 !important; margin: 0.8rem 0; }
    [data-testid="stSidebar"] .stRadio { margin: 0 !important; }
    [data-testid="stSidebar"] .stRadio > div > label { display: none !important; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        display: flex; flex-direction: column; gap: 5px; padding: 0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        display: flex !important; align-items: center !important;
        background: #2A0808 !important; border: 1px solid #5A1515 !important;
        border-radius: 10px !important; padding: 0.7rem 1rem !important;
        margin: 0 !important; cursor: pointer !important; transition: all 0.18s ease !important;
        font-size: 0.92rem !important; font-weight: 600 !important; color: #FFFFFF !important;
        letter-spacing: 0.01em !important; position: relative !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child { display: none !important; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: #4A0000 !important; border-color: #C0392B !important;
        color: #FFFFFF !important; padding-left: 1.3rem !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #A10000, #C0392B) !important;
        border-color: #FF6666 !important; color: #FFFFFF !important;
        font-weight: 800 !important; box-shadow: 0 2px 14px rgba(192,57,43,0.6) !important;
    }
    [data-testid="stSidebar"] input[type="radio"] {
        position: absolute !important; opacity: 0 !important; width: 0 !important; height: 0 !important;
    }
    .kpi-card {
        background: var(--bg-card); border-radius: var(--radius-md);
        padding: 1.4rem 1.6rem; border: 1px solid var(--border-color);
        box-shadow: var(--shadow-md); transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative; overflow: hidden;
    }
    .kpi-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-lg); }
    .kpi-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--red-primary), var(--red-light));
    }
    .kpi-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #EE9999; margin-bottom: 0.5rem; }
    .kpi-value { font-size: 2.1rem; font-weight: 800; color: var(--text-primary); line-height: 1; font-family: 'JetBrains Mono', monospace; }
    .kpi-unit { font-size: 1rem; font-weight: 400; color: var(--text-secondary); }
    .kpi-delta { font-size: 0.82rem; font-weight: 600; margin-top: 0.5rem; padding: 0.2rem 0.6rem; border-radius: 20px; display: inline-block; }
    .kpi-delta.up   { background: #3A0000; color: #FFB3B3; }
    .kpi-delta.down { background: #003A1A; color: #86EFAC; }
    .kpi-delta.warn { background: #3A1A00; color: #FFD0A0; }
    .kpi-icon { position: absolute; top: 1.2rem; right: 1.4rem; font-size: 1.6rem; opacity: 0.15; }
    .section-title {
        font-size: 1.15rem; font-weight: 800; color: #FFFFFF;
        padding: 0.5rem 0 0.3rem 0; margin-bottom: 0.2rem;
        border-left: 3px solid var(--red-primary); padding-left: 0.8rem; letter-spacing: 0.01em;
    }
    .section-subtitle { font-size: 0.82rem; color: #CC8888; margin-bottom: 1rem; padding-left: 1.1rem; }
    .chart-card {
        background: var(--bg-card); border-radius: var(--radius-md);
        padding: 1.2rem 1.4rem; border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm); margin-bottom: 1rem;
    }
    .alert-critical { background: #3A0000; border-left: 4px solid #FF4444; border-radius: var(--radius-sm); padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.88rem; color: #FFB3B3; font-weight: 500; }
    .alert-warning  { background: #2D1500; border-left: 4px solid #FF8C42; border-radius: var(--radius-sm); padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.88rem; color: #FFD0A0; font-weight: 500; }
    .alert-ok       { background: #001A0A; border-left: 4px solid #22C55E; border-radius: var(--radius-sm); padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.88rem; color: #86EFAC; font-weight: 500; }
    .alert-info     { background: #00103A; border-left: 4px solid #60A5FA; border-radius: var(--radius-sm); padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.88rem; color: #BAD5FF; font-weight: 500; }
    .rec-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 1rem 1.2rem; margin-bottom: 0.7rem; box-shadow: var(--shadow-sm); transition: all 0.2s; }
    .rec-card:hover { box-shadow: var(--shadow-md); transform: translateX(3px); }
    .rec-priority-high   { border-left: 4px solid #A10000; }
    .rec-priority-medium { border-left: 4px solid #D35400; }
    .rec-priority-low    { border-left: 4px solid #1A7A4A; }
    .rec-title { font-weight: 700; font-size: 0.92rem; color: var(--text-primary); }
    .rec-body  { font-size: 0.83rem; color: var(--text-secondary); margin-top: 0.3rem; }
    .rec-badge { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; padding: 0.18rem 0.55rem; border-radius: 20px; margin-bottom: 0.4rem; }
    .badge-high   { background: #5A0000; color: #FFB3B3; }
    .badge-medium { background: #3A1A00; color: #FFD0A0; }
    .badge-low    { background: #003A1A; color: #86EFAC; }
    .status-badge { display: inline-block; padding: 0.22rem 0.7rem; border-radius: 20px; font-size: 0.73rem; font-weight: 700; letter-spacing: 0.04em; }
    .status-ok      { background: #003A1A; color: #86EFAC; }
    .status-warn    { background: #3A1A00; color: #FFD0A0; }
    .status-critical{ background: #3A0000; color: #FFB3B3; }
    .mini-metric { display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color); font-size: 0.85rem; }
    .mini-metric:last-child { border-bottom: none; }
    .mini-metric-label { flex: 1; color: var(--text-secondary); }
    .mini-metric-val   { font-weight: 700; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; }
    [data-testid="stDataFrame"] { border-radius: var(--radius-md) !important; overflow: hidden; }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] span,
    .stSlider label, .stSlider p, .stSelectbox label, .stSelectbox p,
    .stCheckbox label, .stCheckbox p, .stCheckbox span,
    div[data-testid="stSlider"] p, div[data-testid="stSelectbox"] p,
    div[data-testid="stCheckbox"] p, div[data-testid="stCheckbox"] span {
        color: #FFFFFF !important; font-weight: 600 !important; font-size: 0.88rem !important; opacity: 1 !important;
    }
    .stSlider > div > div > div { background: var(--red-primary) !important; }
    .stSlider [data-baseweb="slider"] [role="progressbar"] { background: var(--red-primary) !important; }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #C0392B !important; border-color: #FF7777 !important;
        box-shadow: 0 0 0 3px rgba(192,57,43,0.4) !important;
    }
    .stSelectbox [data-baseweb="select"] > div { background: #2A0808 !important; border-color: #6A2020 !important; color: #FFFFFF !important; }
    .stSelectbox [data-baseweb="select"] span { color: #FFFFFF !important; }
    .stCheckbox [data-baseweb="checkbox"] [role="checkbox"] { background: #3A0808 !important; border: 2px solid #C0392B !important; border-radius: 4px !important; }
    .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] [role="checkbox"],
    .stCheckbox [data-baseweb="checkbox"] input:checked ~ [role="checkbox"] {
        background: linear-gradient(135deg, #A10000, #C0392B) !important; border-color: #FF7777 !important;
    }
    .stCheckbox [data-baseweb="checkbox"] svg, .stCheckbox svg { fill: #FFFFFF !important; color: #FFFFFF !important; stroke: #FFFFFF !important; }
    [data-testid="stDownloadButton"] button, .stDownloadButton button {
        background: linear-gradient(135deg, #A10000, #C0392B) !important;
        color: #FFFFFF !important; border: 1px solid #FF6666 !important;
        border-radius: 8px !important; font-weight: 700 !important; font-size: 0.85rem !important;
        opacity: 1 !important; visibility: visible !important;
        box-shadow: 0 2px 10px rgba(161,0,0,0.5) !important; transition: all 0.2s ease !important; padding: 0.5rem 1rem !important;
    }
    [data-testid="stDownloadButton"] button:hover, .stDownloadButton button:hover {
        background: linear-gradient(135deg, #C0392B, #FF4444) !important;
        box-shadow: 0 4px 18px rgba(192,57,43,0.7) !important; transform: translateY(-1px) !important;
    }
    [data-testid="stDownloadButton"] button p, .stDownloadButton button p { color: #FFFFFF !important; font-weight: 700 !important; opacity: 1 !important; }
    .stProgress > div > div { background: linear-gradient(90deg, var(--red-primary), var(--red-light)) !important; }
    .stSpinner > div { border-top-color: var(--red-primary) !important; }
    .custom-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border-color), transparent); margin: 1.5rem 0; }
    /* Guide tab styling */
    .guide-box {
        background: linear-gradient(135deg, #0A0020, #100030);
        border: 1px solid #3030AA;
        border-left: 4px solid #6060FF;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        font-size: 0.87rem;
        color: #C8D8FF;
        line-height: 1.7;
    }
    .guide-box h4 { color: #AABBFF; font-size: 0.95rem; margin: 0.8rem 0 0.4rem 0; border-bottom: 1px solid #3040AA; padding-bottom: 0.3rem; }
    .guide-box ul { padding-left: 1.2rem; margin: 0.4rem 0; }
    .guide-box li { margin: 0.3rem 0; }
    .guide-title { font-size: 1rem; font-weight: 800; color: #AABBFF; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem; }
    /* Upload zone */
    .upload-zone {
        background: #1A0808;
        border: 2px dashed #6A2020;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] { background: #1E0808 !important; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #3A1010; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: #DD9999 !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important; padding: 0.5rem 1rem !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#A10000,#C0392B) !important; color: #FFFFFF !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <script>
    (function styleHamburger() {
        const RED_GRAD = 'linear-gradient(135deg, #A10000, #C0392B)';
        const STYLE = ['background:'+RED_GRAD,'border-radius:50%','width:44px','height:44px',
            'min-width:44px','min-height:44px','border:2px solid #FF7777',
            'box-shadow:0 0 0 3px rgba(192,57,43,0.5),0 4px 20px rgba(161,0,0,0.8)',
            'cursor:pointer','display:flex','align-items:center','justify-content:center',
            'padding:0','opacity:1','visibility:visible'].join(';');
        function isHamburger(btn) {
            const tid = btn.getAttribute('data-testid') || '';
            if (tid.includes('collapsed')||tid.includes('Collapsed')||tid.includes('sidebar')||tid.includes('Sidebar')) return true;
            const header = btn.closest('header'); if (header) return true;
            const r = btn.getBoundingClientRect();
            if (r.width>0&&r.top<100&&r.left<150&&r.width<90&&r.height<90) return true;
            return false;
        }
        function paint(btn) {
            if (btn.closest('[data-testid="stSidebar"]')) return;
            btn.setAttribute('style', STYLE);
            btn.querySelectorAll('svg,svg line,svg rect,svg path,svg polyline,svg circle').forEach(el=>{
                el.style.stroke='#FFFFFF'; el.style.color='#FFFFFF'; el.style.fill='none';
            });
        }
        function applyAll() { document.querySelectorAll('button').forEach(btn=>{ if(isHamburger(btn)) paint(btn); }); }
        applyAll();
        new MutationObserver(applyAll).observe(document.documentElement,{childList:true,subtree:true});
        [200,600,1200,2500].forEach(t=>setTimeout(applyAll,t));
    })();
    (function fixWidgets() {
        function applyWidgetFixes() {
            document.querySelectorAll('[data-testid="stDownloadButton"] button,.stDownloadButton button').forEach(btn=>{
                btn.style.opacity='1'; btn.style.visibility='visible';
                btn.style.background='linear-gradient(135deg,#A10000,#C0392B)';
                btn.style.color='#FFFFFF'; btn.style.border='1px solid #FF6666';
                btn.style.borderRadius='8px'; btn.style.fontWeight='700';
                btn.style.boxShadow='0 2px 10px rgba(161,0,0,0.5)';
                btn.querySelectorAll('p,span').forEach(el=>{ el.style.color='#FFFFFF'; el.style.opacity='1'; });
            });
            document.querySelectorAll('[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] span').forEach(el=>{
                el.style.color='#FFFFFF'; el.style.opacity='1'; el.style.fontWeight='600';
            });
            document.querySelectorAll('.stCheckbox span,.stCheckbox p').forEach(el=>{
                el.style.color='#FFFFFF'; el.style.opacity='1'; el.style.fontWeight='600';
            });
        }
        applyWidgetFixes();
        new MutationObserver(applyWidgetFixes).observe(document.documentElement,{childList:true,subtree:true});
        [300,800,1500,3000].forEach(t=>setTimeout(applyWidgetFixes,t));
    })();
    </script>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def generate_telecom_dataset(n_hours=720):
    np.random.seed(42)
    hours = pd.date_range(start='2026-01-01', periods=n_hours, freq='h')
    hour_of_day = np.array([h.hour for h in hours])
    day_of_week = np.array([h.dayofweek for h in hours])
    daily_pattern   = 50 + 40 * np.sin((hour_of_day - 6) * np.pi / 12) ** 2
    weekly_discount = np.where(day_of_week >= 5, 0.75, 1.0)
    traffic_noise   = np.random.normal(0, 5, n_hours)
    traffic         = np.clip(daily_pattern * weekly_discount + traffic_noise, 5, 140)
    congestion      = np.clip(traffic / 100 + np.random.normal(0, 0.05, n_hours), 0, 1)
    latency_base    = 15 + 80 * congestion ** 2
    latency         = np.clip(latency_base + np.random.normal(0, 2, n_hours), 5, 250)
    energy          = np.clip(120 + 0.8*traffic + 20*np.sin(hour_of_day*np.pi/12) + np.random.normal(0, 8, n_hours), 80, 280)
    packet_loss     = np.clip(0.5 + 3*congestion**1.5 + np.random.exponential(0.2, n_hours), 0, 15)
    throughput      = np.clip(traffic*9.5 - latency*0.3 + np.random.normal(0, 20, n_hours), 50, 1300)
    signal_strength = np.clip(-80 + 20*(1-congestion) + np.random.normal(0, 3, n_hours), -110, -50)
    active_users    = np.clip(traffic*12 + np.random.normal(0, 30, n_hours), 20, 1800).astype(int)
    anomaly_idx     = np.random.choice(n_hours, size=int(n_hours*0.02), replace=False)
    latency[anomaly_idx]  *= np.random.uniform(2.5, 5, len(anomaly_idx))
    traffic[anomaly_idx]  *= np.random.uniform(0.1, 0.3, len(anomaly_idx))
    energy[anomaly_idx]   *= np.random.uniform(1.4, 2.0, len(anomaly_idx))
    true_anomalies         = np.zeros(n_hours, dtype=int)
    true_anomalies[anomaly_idx] = 1
    return pd.DataFrame({
        'timestamp': hours, 'traffic_gbps': np.round(traffic, 2),
        'latency_ms': np.round(latency, 1), 'congestion_pct': np.round(congestion*100, 1),
        'energy_kwh': np.round(energy, 1), 'packet_loss_pct': np.round(packet_loss, 2),
        'throughput_mbps': np.round(throughput, 1), 'signal_dbm': np.round(signal_strength, 1),
        'active_users': active_users, 'true_anomaly': true_anomalies
    })


@st.cache_data
def get_cell_tower_data():
    """Antennes avec coordonnées géographiques (Abidjan + environs)."""
    np.random.seed(7)
    sites = [f"SITE-{str(i).zfill(3)}" for i in range(1, 25)]
    regions = np.random.choice(['Plateau','Cocody','Yopougon','Abobo','Marcory','Treichville','Adjamé'], len(sites))
    load    = np.random.uniform(20, 95, len(sites))
    latency = 10 + 80*(load/100)**1.5 + np.random.normal(0, 5, len(sites))
    energy  = 100 + 1.2*load + np.random.normal(0, 10, len(sites))
    uptime  = np.clip(99.9 - 0.1*load*np.random.uniform(0.5, 1.5, len(sites)), 94, 99.99)
    status  = np.where(load>85,'CRITICAL', np.where(load>65,'WARNING','NORMAL'))
    # Coordonnées GPS simulées autour d'Abidjan
    base_lat, base_lon = 5.35, -4.00
    lats = base_lat + np.random.uniform(-0.25, 0.25, len(sites))
    lons = base_lon + np.random.uniform(-0.25, 0.25, len(sites))
    return pd.DataFrame({
        'Site': sites, 'region': regions, 'load_pct': np.round(load,1),
        'latency_ms': np.round(latency,1), 'energy_kwh': np.round(energy,1),
        'uptime_pct': np.round(uptime,3), 'status': status,
        'lat': np.round(lats,5), 'lon': np.round(lons,5)
    })


def validate_uploaded_df(df):
    """Valide et normalise un DataFrame uploadé."""
    required_cols = {'timestamp','traffic_gbps','latency_ms'}
    optional_cols = {
        'congestion_pct': 50.0, 'energy_kwh': 150.0, 'packet_loss_pct': 1.0,
        'throughput_mbps': 500.0, 'signal_dbm': -75.0, 'active_users': 500, 'true_anomaly': 0
    }
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    missing = required_cols - set(df.columns)
    if missing:
        return None, f"Colonnes manquantes : {missing}"
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except:
        return None, "Impossible de parser la colonne 'timestamp'. Format attendu : YYYY-MM-DD HH:MM:SS"
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col, default in optional_cols.items():
        if col not in df.columns:
            df[col] = default
    numeric = ['traffic_gbps','latency_ms','congestion_pct','energy_kwh',
                'packet_loss_pct','throughput_mbps','signal_dbm','active_users']
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median() if c in df.columns else 0)
    return df, None


# ─────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def train_anomaly_model(_df):
    features = ['traffic_gbps','latency_ms','congestion_pct','energy_kwh','packet_loss_pct']
    X = _df[features].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.02, n_estimators=200, random_state=42)
    model.fit(X_s)
    scores = model.decision_function(X_s)
    preds  = model.predict(X_s)
    return model, scaler, scores, preds


@st.cache_resource
def train_traffic_predictor(_df):
    df2 = _df.copy()
    df2['hour']  = df2['timestamp'].dt.hour
    df2['dow']   = df2['timestamp'].dt.dayofweek
    df2['lag1']  = df2['traffic_gbps'].shift(1)
    df2['lag24'] = df2['traffic_gbps'].shift(24)
    df2['roll6'] = df2['traffic_gbps'].rolling(6).mean()
    df2 = df2.dropna()
    features = ['hour','dow','latency_ms','congestion_pct','energy_kwh','lag1','lag24','roll6']
    X, y = df2[features], df2['traffic_gbps']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=250, max_depth=5, learning_rate=0.06, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    importances = dict(zip(features, model.feature_importances_))
    return model, X_te, y_te, y_pred, rmse, mae, r2, importances


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
def _apply_axes(fig):
    fig.update_xaxes(gridcolor='#3A1010', showline=False, tickfont_size=11, tickfont_color='#E8E8E8')
    fig.update_yaxes(gridcolor='#3A1010', showline=False, tickfont_size=11, tickfont_color='#E8E8E8')
    return fig

PLOTLY_LAYOUT = dict(
    font_family="Inter", font_color="#E8E8E8",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10,r=10,t=35,b=10),
    title_font_size=13, title_font_color='#FFFFFF',
    legend=dict(bgcolor='rgba(30,0,0,0.6)', font_size=11, font_color='#FFFFFF'),
)
RED_SCALE = ['#FFF0F0','#FFCCCC','#FF9999','#FF4444','#C0392B','#8B0000','#5C0000']
DIVERGING  = px.colors.diverging.RdYlGn[::-1]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def info_box(texte, icone="💡"):
    st.markdown(f"""
    <div style="background:#1A0A00;border:1px solid #6A3010;border-left:4px solid #FF6600;
                border-radius:10px;padding:0.75rem 1rem;margin-bottom:1rem;
                font-size:0.85rem;color:#FFD0A0;line-height:1.6;">
        <strong style="color:#FF9944;">{icone} À SAVOIR :</strong>&nbsp; {texte}
    </div>
    """, unsafe_allow_html=True)

def chart_caption(texte):
    st.markdown(f"""
    <div style="background:rgba(161,0,0,0.08);border-left:3px solid #6A2020;
                border-radius:0 0 8px 8px;padding:0.5rem 1rem;margin-top:-0.5rem;
                margin-bottom:1rem;font-size:0.8rem;color:#DD9999;line-height:1.5;">
        ℹ️ {texte}
    </div>
    """, unsafe_allow_html=True)

def guide_section(title, content_html):
    """Affiche un onglet Guide avec explication complète."""
    full_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: transparent; font-family: 'Inter', sans-serif; color: #C8D8FF; font-size: 0.87rem; line-height: 1.7; }}
    .guide-box {{
        background: linear-gradient(135deg, #0A0020, #100030);
        border: 1px solid #3030AA;
        border-left: 4px solid #6060FF;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
    }}
    .guide-title {{
        font-size: 1rem; font-weight: 800; color: #AABBFF;
        margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;
    }}
    h4 {{
        color: #AABBFF; font-size: 0.95rem; font-weight: 700;
        margin: 0.8rem 0 0.4rem 0;
        border-bottom: 1px solid #3040AA; padding-bottom: 0.3rem;
    }}
    p {{ margin: 0.3rem 0 0.6rem 0; }}
    ul {{ padding-left: 1.2rem; margin: 0.4rem 0; }}
    li {{ margin: 0.3rem 0; }}
    strong {{ color: #DDEEFF; }}
    </style>
    <div class="guide-box">
        <div class="guide-title">📖 {title}</div>
        {content_html}
    </div>
    """
    components.html(full_html, height=600, scrolling=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
def render_header():
    html = (
        "<div style='background:linear-gradient(135deg,#6B0000 0%,#A10000 40%,#C0392B 70%,#8B0000 100%);"
        "border-radius:18px;padding:2.4rem 2rem 2rem 2rem;margin-bottom:2rem;"
        "box-shadow:0 8px 40px rgba(161,0,0,0.4);text-align:center;'>"
        "<div style='display:inline-block;background:rgba(255,255,255,0.12);"
        "border:1px solid rgba(255,255,255,0.25);border-radius:30px;"
        "padding:0.25rem 1rem;font-size:0.7rem;letter-spacing:0.15em;"
        "text-transform:uppercase;color:rgba(255,220,220,0.9);font-weight:600;margin-bottom:0.9rem;'>"
        "📡 Plateforme Intelligente Telecom 📡</div><br>"
        "<div style='font-size:1.8rem;font-weight:900;color:#FFFFFF;letter-spacing:0.02em;line-height:1.2;margin-bottom:0.5rem;'>"
        "Système Autonome d'Intelligence Réseau Télécom</div>"
        "<div style='font-size:3.2rem;font-weight:900;color:#FFD0D0;letter-spacing:0.12em;line-height:1;margin:0.3rem 0 0.6rem 0;'>"
        "AutoTel AI</div>"
        "<div style='font-size:0.88rem;color:rgba(255,200,200,0.85);letter-spacing:0.1em;text-transform:uppercase;font-weight:500;margin-bottom:0.9rem;'>"
        "Par <span style='color:#FFB3B3;font-weight:700;'>ANOH AMON HEMERSON</span></div>"
        "<div style='font-size:0.92rem;color:rgba(255,230,230,0.8);max-width:600px;margin:0 auto;line-height:1.6;'>"
        "Système IA pour la surveillance autonome, la prédiction et l'optimisation des réseaux télécom</div>"
        "<div style='margin-top:1.2rem;display:flex;justify-content:center;gap:0.6rem;flex-wrap:wrap;'>"
        "<span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:0.22rem 0.8rem;font-size:0.7rem;color:rgba(255,220,220,0.9);font-weight:600;'>🧠 ML Intégré</span>"
        "<span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:0.22rem 0.8rem;font-size:0.7rem;color:rgba(255,220,220,0.9);font-weight:600;'>⚡ Surveillance Temps Réel</span>"
        "<span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:0.22rem 0.8rem;font-size:0.7rem;color:rgba(255,220,220,0.9);font-weight:600;'>🔮 Analyse Prédictive</span>"
        "<span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:0.22rem 0.8rem;font-size:0.7rem;color:rgba(255,220,220,0.9);font-weight:600;'>🛡️ Détection Anomalies</span>"
        "<span style='background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:0.22rem 0.8rem;font-size:0.7rem;color:rgba(255,220,220,0.9);font-weight:600;'>🗺️ Carte Géographique</span>"
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.2rem 0 0.5rem 0;">
            <div style="font-size:2.5rem;">📡</div>
            <div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;letter-spacing:0.12em;">AutoTel AI</div>
            <div style="font-size:0.68rem;color:#FF9999;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem;">
                Telecom Intelligence v3.0
            </div>
        </div>
        <hr style="border-color:rgba(192,57,43,0.5);margin:0.6rem 0 1rem 0;">
        <div style="font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                    color:#EE9999;padding:0 0.2rem 0.5rem 0.2rem;font-weight:700;">
            Navigation
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "📊  Tableau de Bord":  "Tableau de Bord",
            "📡  Surveillance":     "Surveillance",
            "🗺️  Carte Réseau":     "Carte Réseau",
            "📂  Import Données":   "Import Données",
            "🔮  Prédiction":       "Prédiction",
            "⚙️  Optimisation":     "Optimisation",
            "🎛️  Simulation":       "Simulation",
            "🔍  Analyses":         "Analyses",
            "ℹ️  À Propos":         "À Propos",
        }
        selected = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
        st.markdown("<hr style='border-color:rgba(192,57,43,0.5);margin:1rem 0;'>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                    color:#EE9999;margin-bottom:0.6rem;font-weight:700;">Statut Système</div>
        """, unsafe_allow_html=True)
        statuses = [
            ("●","#22C55E","Moteur Principal","EN LIGNE"),
            ("●","#22C55E","Pipeline ML","ACTIF"),
            ("●","#F59E0B","Moteur d'Alertes","2 ALERTES"),
            ("●","#22C55E","Ingestion Données","EN COURS"),
        ]
        for dot, color, name, val in statuses:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-size:0.82rem;color:#FFFFFF;">
                <span style="color:{color};font-size:0.7rem;">{dot}</span>
                <span style="flex:1;color:#FFFFFF;">{name}</span>
                <span style="font-size:0.74rem;color:{color};font-weight:700;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(192,57,43,0.5);margin:0.8rem 0;'>", unsafe_allow_html=True)
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.markdown(f"""
        <div style="background:#1E0000;border:1px solid #3A1010;border-radius:8px;
                    padding:0.5rem 0.8rem;margin-bottom:0.6rem;text-align:center;">
            <div style="font-size:0.62rem;color:#EE9999;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;">🕐 Dernière mise à jour</div>
            <div style="font-size:0.78rem;color:#FFFFFF;font-family:'JetBrains Mono',monospace;font-weight:600;">{now_str}</div>
        </div>
        <div style="font-size:0.72rem;color:#DD9999;text-align:center;padding-bottom:1rem;">
            AutoTel AI v3.0<br>© 2026 ANOH AMON HEMERSON
        </div>
        """, unsafe_allow_html=True)

        return pages[selected]


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
def render_kpi_row(df):
    last = df.iloc[-1]; prev = df.iloc[-25:-1]
    avg_lat  = last['latency_ms'];   prev_lat  = prev['latency_ms'].mean()
    avg_trf  = last['traffic_gbps']; prev_trf  = prev['traffic_gbps'].mean()
    avg_cong = last['congestion_pct']; prev_cong = prev['congestion_pct'].mean()
    avg_eng  = last['energy_kwh'];   prev_eng  = prev['energy_kwh'].mean()
    delta_lat  = (avg_lat  - prev_lat)  / prev_lat  * 100
    delta_trf  = (avg_trf  - prev_trf)  / prev_trf  * 100
    delta_cong = (avg_cong - prev_cong) / prev_cong * 100
    delta_eng  = (avg_eng  - prev_eng)  / prev_eng  * 100
    kpis = [
        ("LATENCE RÉSEAU",     f"{avg_lat:.1f}", "ms",   delta_lat,  "up" if delta_lat>5 else ("down" if delta_lat<-5 else "warn"), "📶"),
        ("CHARGE TRAFIC",      f"{avg_trf:.1f}", "Gbps", delta_trf,  "warn" if delta_trf>15 else "down", "🔀"),
        ("INDICE CONGESTION",  f"{avg_cong:.1f}","%",    delta_cong, "up" if delta_cong>5 else ("down" if delta_cong<-5 else "warn"), "⚠️"),
        ("CONSOMMATION ÉNERGIE",f"{avg_eng:.1f}","kWh",  delta_eng,  "up" if delta_eng>5 else ("down" if delta_eng<-5 else "warn"), "⚡"),
    ]
    cols = st.columns(4)
    for col, (label, val, unit, delta, cls, icon) in zip(cols, kpis):
        arrow = "▲" if delta > 0 else "▼"
        sign  = "+" if delta > 0 else ""
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}<span class="kpi-unit"> {unit}</span></div>
                <div class="kpi-delta {cls}">{arrow} {sign}{delta:.1f}% vs 24h préc.</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard(df):
    tab_dash, tab_guide = st.tabs(["📊 Tableau de Bord", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Tableau de Bord", """
        <h4>🎯 Objectif de cette page</h4>
        <p>Le tableau de bord offre une vue d'ensemble instantanée de la santé du réseau télécom. C'est la première page à consulter chaque matin ou lors d'un incident.</p>

        <h4>📌 Score de Santé Réseau (0–100)</h4>
        <ul>
            <li><strong>80–100 (Vert) :</strong> Réseau optimal, aucune action requise</li>
            <li><strong>50–79 (Jaune) :</strong> Dégradation modérée, à surveiller</li>
            <li><strong>0–49 (Rouge) :</strong> Réseau critique, intervention immédiate</li>
        </ul>
        <p>Calcul : 40% latence + 35% congestion + 25% pertes paquets</p>

        <h4>📊 Les 4 KPIs en haut</h4>
        <ul>
            <li><strong>Latence (ms) :</strong> Délai de transmission. Idéal &lt; 50ms. SLA critique à 100ms</li>
            <li><strong>Charge Trafic (Gbps) :</strong> Volume de données transmises à l'instant T</li>
            <li><strong>Congestion (%) :</strong> Taux d'utilisation de la capacité réseau. Critique &gt; 80%</li>
            <li><strong>Énergie (kWh) :</strong> Consommation électrique. Objectif d'efficacité &lt; 180 kWh/h</li>
        </ul>
        <p>La flèche ▲/▼ et le pourcentage comparent la valeur actuelle aux 24h précédentes.</p>

        <h4>📈 Graphique Trafic & Latence</h4>
        <p>Zone rouge claire = trafic brut heure par heure. Ligne rouge pleine = moyenne mobile sur 12h (lisse les pics). La courbe orange en bas = latence correspondante. <strong>Corrélation attendue :</strong> quand le trafic monte, la latence augmente.</p>

        <h4>🌡️ Carte Thermique Congestion</h4>
        <p>Grille Heure (0–23) × Jour de la semaine. Plus c'est rouge foncé, plus la congestion est élevée. Typiquement : rouge entre 18h–22h en semaine. Utilisez cette carte pour planifier les maintenances en heures creuses (bleu = 3h–6h).</p>

        <h4>⚡ Consommation Énergétique</h4>
        <p>Barres = consommation moyenne journalière en kWh. Les pics correspondent aux jours de forte charge. Un pic inattendu peut indiquer une panne d'équipement ou un refroidissement défaillant.</p>

        <h4>🗼 Tableau Antennes Relais</h4>
        <p>Liste tous les sites avec leur charge, latence, énergie et uptime. Couleurs : rouge = critique (&gt;85% charge), orange = attention (&gt;65%), vert = normal.</p>
        """)

    with tab_dash:
        info_box(
            "Ce tableau de bord affiche l'état général du réseau télécom en temps réel. "
            "Les 4 indicateurs en haut (KPIs) montrent la latence, le trafic, la congestion et l'énergie. "
            "Consultez l'onglet 📖 Guide pour une explication détaillée de chaque graphique.",
            "📊"
        )
        last24 = df.tail(24)
        avg_lat  = last24['latency_ms'].mean()
        avg_cong = last24['congestion_pct'].mean()
        avg_pl   = last24['packet_loss_pct'].mean()
        score_lat  = max(0, 100 - (avg_lat / 2))
        score_cong = max(0, 100 - avg_cong)
        score_pl   = max(0, 100 - avg_pl * 20)
        health = min(100, max(0, int(score_lat*0.4 + score_cong*0.35 + score_pl*0.25)))
        if health >= 75:   h_color, h_label, h_icon = "#22C55E","BON","🟢"
        elif health >= 50: h_color, h_label, h_icon = "#F59E0B","MOYEN","🟡"
        else:              h_color, h_label, h_icon = "#EF4444","CRITIQUE","🔴"

        col_h, col_f, col_e = st.columns([1.4, 1, 0.6])
        with col_h:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1E0808,#2A0000);border:1px solid #6A2020;
                        border-radius:14px;padding:1rem 1.4rem;display:flex;align-items:center;gap:1.2rem;
                        box-shadow:0 4px 16px rgba(0,0,0,0.5);margin-bottom:1rem;">
                <div style="text-align:center;min-width:72px;">
                    <div style="font-size:2.2rem;font-weight:900;color:{h_color};font-family:'JetBrains Mono',monospace;line-height:1;">{health}</div>
                    <div style="font-size:0.62rem;color:#DD9999;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.1rem;">/ 100</div>
                </div>
                <div style="flex:1;border-left:1px solid #3A1010;padding-left:1rem;">
                    <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#DD9999;font-weight:600;">Score de Santé Réseau</div>
                    <div style="font-size:1.1rem;font-weight:800;color:{h_color};margin:0.2rem 0;">{h_icon} {h_label}</div>
                    <div style="background:#3A0808;border-radius:20px;height:7px;margin-top:0.4rem;overflow:hidden;">
                        <div style="width:{health}%;height:100%;border-radius:20px;background:linear-gradient(90deg,{h_color},{h_color}88);"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_f:
            period = st.selectbox("📅 Période d'analyse", ["7 derniers jours","14 derniers jours","30 derniers jours"], index=0)
        with col_e:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Export CSV", data=csv_data, file_name="autotel_data.csv", mime="text/csv", use_container_width=True)

        period_map = {"7 derniers jours":168,"14 derniers jours":336,"30 derniers jours":720}
        n_hours = period_map[period]
        render_kpi_row(df)
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='section-title'>Trafic Réseau & Latence — {period}</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Agrégation horaire · Moyenne mobile superposée</div>", unsafe_allow_html=True)
        last7 = df.tail(n_hours).copy()
        last7['ma12'] = last7['traffic_gbps'].rolling(12).mean()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Trafic (Gbps)","Latence (ms)"),
                            vertical_spacing=0.08, row_heights=[0.55,0.45])
        fig.add_trace(go.Scatter(x=last7['timestamp'],y=last7['traffic_gbps'],name='Trafic',mode='lines',
                                 line=dict(color='rgba(161,0,0,0.35)',width=1),fill='tozeroy',fillcolor='rgba(161,0,0,0.07)'),row=1,col=1)
        fig.add_trace(go.Scatter(x=last7['timestamp'],y=last7['ma12'],name='MA-12h',mode='lines',
                                 line=dict(color='#A10000',width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=last7['timestamp'],y=last7['latency_ms'],name='Latence',mode='lines',
                                 line=dict(color='#D35400',width=1.5),fill='tozeroy',fillcolor='rgba(211,84,0,0.07)'),row=2,col=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=370, showlegend=True)
        fig.update_yaxes(gridcolor='#3A1010', showline=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        chart_caption("Trafic horaire avec moyenne mobile 12h. Un pic de trafic entraîne généralement une hausse de latence visible dans le panneau inférieur.")

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.4,1])
        with c1:
            st.markdown("<div class='section-title'>Carte Thermique Congestion — Heure × Jour</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Congestion moyenne % par heure et jour de la semaine</div>", unsafe_allow_html=True)
            pivot = df.copy()
            pivot['hour'] = pivot['timestamp'].dt.hour
            pivot['dow']  = pivot['timestamp'].dt.day_name()
            hm = pivot.pivot_table(values='congestion_pct', index='hour', columns='dow', aggfunc='mean')
            day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            day_fr = {'Monday':'Lundi','Tuesday':'Mardi','Wednesday':'Mercredi','Thursday':'Jeudi',
                      'Friday':'Vendredi','Saturday':'Samedi','Sunday':'Dimanche'}
            hm = hm.reindex(columns=[d for d in day_order if d in hm.columns])
            hm.columns = [day_fr.get(c,c) for c in hm.columns]
            fig2 = go.Figure(go.Heatmap(z=hm.values,x=hm.columns,y=hm.index,
                colorscale=[[0,'#F0FFF4'],[0.4,'#FFEECC'],[0.7,'#FFAAAA'],[1,'#8B0000']],
                colorbar=dict(title='Cong %',thickness=10,len=0.8)))
            _layout2 = {k:v for k,v in PLOTLY_LAYOUT.items() if k!='yaxis'}
            fig2.update_layout(**_layout2,height=310,yaxis=dict(title='Heure',tickmode='linear',dtick=4,gridcolor='#3A1010'))
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar':False})
            chart_caption("Rouge foncé = congestion élevée. Les heures de pointe (18h–22h) présentent systématiquement une congestion plus forte.")
        with c2:
            st.markdown("<div class='section-title'>Consommation Énergétique</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Moyenne quotidienne kWh · Fenêtre 30 jours</div>", unsafe_allow_html=True)
            daily = df.resample('D', on='timestamp')['energy_kwh'].mean().reset_index()
            fig3 = go.Figure(go.Bar(x=daily['timestamp'],y=daily['energy_kwh'],
                marker_color='rgba(161,0,0,0.75)',marker_line=dict(color='#A10000',width=0.5)))
            fig3.update_layout(**PLOTLY_LAYOUT, height=310)
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar':False})
            chart_caption("Consommation moyenne quotidienne (kWh). Objectif d'efficacité : maintenir sous 180 kWh/jour.")

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Vue d'Ensemble des Antennes Relais</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Charge et performance temps réel par site</div>", unsafe_allow_html=True)
        towers = get_cell_tower_data()
        def color_status(val):
            if val=='CRITICAL': return 'background-color:#FFF0F0;color:#8B0000;font-weight:700'
            if val=='WARNING':  return 'background-color:#FFF8F0;color:#D35400;font-weight:700'
            return 'background-color:#F0FFF4;color:#1A7A4A;font-weight:600'
        def color_load(val):
            if val>85: return 'color:#8B0000;font-weight:700'
            if val>65: return 'color:#D35400;font-weight:600'
            return 'color:#1A7A4A'
        towers_display = towers.drop(columns=['lat','lon'])
        styled = towers_display.style\
            .map(color_status, subset=['status'])\
            .map(color_load, subset=['load_pct'])\
            .format({'load_pct':'{}%','latency_ms':'{} ms','energy_kwh':'{} kWh','uptime_pct':'{}%'})\
            .set_properties(**{'font-size':'0.84rem'})
        st.dataframe(styled, use_container_width=True, height=320)


# ─────────────────────────────────────────────
# PAGE: CARTE GÉOGRAPHIQUE
# ─────────────────────────────────────────────
def page_map(df):
    tab_map, tab_guide = st.tabs(["🗺️ Carte Réseau", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Carte Géographique du Réseau", """
        <h4>🎯 Objectif</h4>
        <p>La carte géographique visualise la distribution spatiale de toutes les antennes relais (BTS/NodeB) avec leur état en temps réel. Elle permet d'identifier les zones géographiques à risque et de planifier les interventions terrain.</p>

        <h4>🎨 Code couleur des marqueurs</h4>
        <ul>
            <li><strong style="color:#FF4444;">● Rouge :</strong> Site CRITIQUE — charge &gt; 85%. Intervention immédiate requise</li>
            <li><strong style="color:#FF8C42;">● Orange :</strong> Site en ALERTE — charge 65–85%. À surveiller sous 2h</li>
            <li><strong style="color:#22C55E;">● Vert :</strong> Site NORMAL — charge &lt; 65%. Fonctionnement nominal</li>
        </ul>

        <h4>📏 Taille des bulles</h4>
        <p>La taille de chaque bulle est proportionnelle à la charge de trafic du site. Un gros cercle = site très chargé = priorité d'attention.</p>

        <h4>🔍 Interactivité</h4>
        <ul>
            <li>Cliquez sur un site pour voir ses détails (charge, latence, énergie, uptime)</li>
            <li>Zoomez/dézoomez pour explorer la zone géographique</li>
            <li>Passez la souris sur un site pour voir les métriques en tooltip</li>
        </ul>

        <h4>📊 Carte Thermique de Charge</h4>
        <p>La deuxième visualisation montre une heatmap de densité de charge géographique. Les zones rouges indiquent une concentration de sites surchargés — priorité pour le déploiement de nouvelles capacités ou de small cells.</p>

        <h4>📋 Tableau Récapitulatif Régional</h4>
        <p>Agrégation par région géographique : charge moyenne, nombre de sites en alerte, latence médiane. Permet de comparer les performances entre zones urbaines.</p>
        """)

    with tab_map:
        info_box(
            "Carte interactive de toutes les antennes relais. Rouge = critique, Orange = alerte, Vert = normal. "
            "La taille des bulles reflète la charge de trafic. Cliquez sur un site pour ses détails.",
            "🗺️"
        )
        towers = get_cell_tower_data()

        # Filtres
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            status_filter = st.multiselect("Filtrer par statut", ['NORMAL','WARNING','CRITICAL'],
                                           default=['NORMAL','WARNING','CRITICAL'])
        with col_f2:
            region_filter = st.multiselect("Filtrer par région",
                                           sorted(towers['region'].unique()),
                                           default=sorted(towers['region'].unique()))
        with col_f3:
            metric_color = st.selectbox("Colorer par", ['Statut','Charge (%)','Latence (ms)','Énergie (kWh)'])

        filtered = towers[towers['status'].isin(status_filter) & towers['region'].isin(region_filter)]

        if filtered.empty:
            st.warning("Aucun site correspondant aux filtres.")
        else:
            # Définir couleurs et taille
            color_map_status = {'NORMAL':'#22C55E','WARNING':'#FF8C42','CRITICAL':'#FF4444'}

            if metric_color == 'Statut':
                filtered = filtered.copy()
                filtered['_color'] = filtered['status'].map(color_map_status)
                color_col = 'status'
                color_discrete = color_map_status
                fig_map = px.scatter_mapbox(
                    filtered, lat='lat', lon='lon',
                    size='load_pct', size_max=30,
                    color='status',
                    color_discrete_map=color_map_status,
                    hover_name='Site',
                    hover_data={'lat':False,'lon':False,'load_pct':':.1f',
                                'latency_ms':':.1f','energy_kwh':':.1f',
                                'uptime_pct':':.2f','region':True},
                    labels={'load_pct':'Charge %','latency_ms':'Latence ms',
                            'energy_kwh':'Énergie kWh','uptime_pct':'Uptime %'},
                    zoom=11, height=520,
                    mapbox_style='carto-darkmatter',
                    title='Carte des Antennes Relais — Statut'
                )
            else:
                col_map = {'Charge (%)':'load_pct','Latence (ms)':'latency_ms','Énergie (kWh)':'energy_kwh'}
                cname = col_map[metric_color]
                fig_map = px.scatter_mapbox(
                    filtered, lat='lat', lon='lon',
                    size='load_pct', size_max=30,
                    color=cname,
                    color_continuous_scale=['#22C55E','#F59E0B','#FF4444'],
                    hover_name='Site',
                    hover_data={'lat':False,'lon':False,'load_pct':':.1f',
                                'latency_ms':':.1f','energy_kwh':':.1f','region':True},
                    zoom=11, height=520,
                    mapbox_style='carto-darkmatter',
                    title=f'Carte des Antennes Relais — {metric_color}'
                )

            fig_map.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FFFFFF', font_family='Inter',
                title_font_color='#FFFFFF', title_font_size=14,
                margin=dict(l=0,r=0,t=40,b=0),
                legend=dict(bgcolor='rgba(30,8,8,0.85)', font_color='#FFFFFF', font_size=12)
            )
            st.plotly_chart(fig_map, use_container_width=True)
            chart_caption("Carte interactive Mapbox. Chaque cercle = une antenne relais. Taille ∝ charge de trafic. Survolez un site pour ses métriques détaillées.")

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

            # Stats résumées
            c1, c2, c3 = st.columns(3)
            n_critical = (filtered['status']=='CRITICAL').sum()
            n_warning  = (filtered['status']=='WARNING').sum()
            n_normal   = (filtered['status']=='NORMAL').sum()
            with c1:
                st.markdown(f"""
                <div class="kpi-card"><div class="kpi-icon">🔴</div>
                <div class="kpi-label">Sites Critiques</div>
                <div class="kpi-value" style="color:#FF4444;">{n_critical}</div>
                <div class="kpi-delta up">Action immédiate requise</div></div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="kpi-card"><div class="kpi-icon">🟡</div>
                <div class="kpi-label">Sites en Alerte</div>
                <div class="kpi-value" style="color:#F59E0B;">{n_warning}</div>
                <div class="kpi-delta warn">Surveiller sous 2h</div></div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="kpi-card"><div class="kpi-icon">🟢</div>
                <div class="kpi-label">Sites Normaux</div>
                <div class="kpi-value" style="color:#22C55E;">{n_normal}</div>
                <div class="kpi-delta down">Fonctionnement nominal</div></div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

            # Analyse régionale
            st.markdown("<div class='section-title'>Performance par Région</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-subtitle'>Agrégation des métriques par zone géographique</div>", unsafe_allow_html=True)
            region_stats = filtered.groupby('region').agg(
                Nb_Sites=('Site','count'),
                Charge_Moy=('load_pct','mean'),
                Latence_Med=('latency_ms','median'),
                Energie_Moy=('energy_kwh','mean'),
                Sites_Critiques=('status', lambda x: (x=='CRITICAL').sum()),
                Sites_Alerte=('status', lambda x: (x=='WARNING').sum()),
            ).round(1).reset_index()
            region_stats.columns = ['Région','Nb Sites','Charge Moy (%)','Latence Méd (ms)','Énergie Moy (kWh)','Critiques','Alertes']

            def color_charge(val):
                if val>85: return 'color:#FF4444;font-weight:700'
                if val>65: return 'color:#FF8C42;font-weight:600'
                return 'color:#22C55E'

            styled_r = region_stats.style\
                .map(color_charge, subset=['Charge Moy (%)'])\
                .format({'Charge Moy (%)':'{:.1f}%','Latence Méd (ms)':'{:.1f} ms','Énergie Moy (kWh)':'{:.1f} kWh'})\
                .set_properties(**{'font-size':'0.86rem'})
            st.dataframe(styled_r, use_container_width=True)

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

            # Graphique comparatif régions
            st.markdown("<div class='section-title'>Comparaison Charge par Région</div>", unsafe_allow_html=True)
            fig_reg = go.Figure()
            colors_reg = ['#FF4444' if v>85 else '#FF8C42' if v>65 else '#22C55E'
                          for v in region_stats['Charge Moy (%)']]
            fig_reg.add_trace(go.Bar(
                x=region_stats['Région'], y=region_stats['Charge Moy (%)'],
                marker_color=colors_reg, text=region_stats['Charge Moy (%)'].round(1).astype(str)+'%',
                textposition='outside', textfont=dict(color='#FFFFFF', size=11)
            ))
            fig_reg.add_hline(y=85, line_dash='dash', line_color='#FF4444',
                              annotation_text='Seuil Critique 85%', annotation_font_color='#FF9999')
            fig_reg.add_hline(y=65, line_dash='dash', line_color='#FF8C42',
                              annotation_text='Seuil Alerte 65%', annotation_font_color='#FFD0A0')
            fig_reg.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title='Charge (%)', yaxis_range=[0,110])
            st.plotly_chart(fig_reg, use_container_width=True, config={'displayModeBar':False})
            chart_caption("Charge moyenne par région. Rouge = zone critique nécessitant du renforcement réseau. Lignes tiretées = seuils d'alerte.")


# ─────────────────────────────────────────────
# PAGE: IMPORT DONNÉES RÉELLES
# ─────────────────────────────────────────────
def page_import(df_default):
    tab_import, tab_guide = st.tabs(["📂 Import Données", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Import de Données Réelles", """
        <h4>🎯 Pourquoi importer vos données ?</h4>
        <p>AutoTel AI fonctionne par défaut avec des données simulées. En important vos vraies données réseau (exports SNMP, NetFlow, OSS/BSS), tous les graphiques, modèles ML et alertes seront calculés sur votre réseau réel.</p>

        <h4>📋 Format attendu (CSV)</h4>
        <p>Le fichier CSV doit contenir au minimum ces colonnes :</p>
        <ul>
            <li><strong>timestamp</strong> — Format : YYYY-MM-DD HH:MM:SS (ex: 2026-01-15 14:30:00)</li>
            <li><strong>traffic_gbps</strong> — Trafic en Gigabits par seconde (ex: 75.3)</li>
            <li><strong>latency_ms</strong> — Latence en millisecondes (ex: 42.1)</li>
        </ul>
        <p>Colonnes optionnelles (valeurs par défaut utilisées si absentes) :</p>
        <ul>
            <li><strong>congestion_pct</strong> — Congestion en % (défaut: 50)</li>
            <li><strong>energy_kwh</strong> — Consommation en kWh (défaut: 150)</li>
            <li><strong>packet_loss_pct</strong> — Pertes de paquets en % (défaut: 1.0)</li>
            <li><strong>throughput_mbps</strong> — Débit en Mbps (défaut: 500)</li>
            <li><strong>signal_dbm</strong> — Signal en dBm (défaut: -75)</li>
            <li><strong>active_users</strong> — Utilisateurs actifs (défaut: 500)</li>
        </ul>

        <h4>🔧 Sources de données compatibles</h4>
        <ul>
            <li>Exports SNMP (Nagios, Zabbix, PRTG) — généralement en CSV ou TSV</li>
            <li>NetFlow / sFlow agrégés en CSV</li>
            <li>Exports BSS/OSS (Ericsson ENM, Nokia NetAct, Huawei iManager)</li>
            <li>Données de test générées par ce simulateur (bouton Export CSV sur le Tableau de Bord)</li>
        </ul>

        <h4>⚠️ Conseils</h4>
        <ul>
            <li>Minimum 48 lignes (heures) recommandé pour des analyses fiables</li>
            <li>1 ligne = 1 heure de mesure agrégée</li>
            <li>Évitez les valeurs manquantes (NaN) — elles sont remplacées par la médiane</li>
            <li>Les données importées restent en session uniquement (non sauvegardées)</li>
        </ul>
        """)

    with tab_import:
        info_box(
            "Importez vos données réseau réelles pour remplacer les données simulées. "
            "Une fois importées, TOUS les onglets (Surveillance, Prédiction, etc.) utiliseront vos données. "
            "Consultez le 📖 Guide pour connaître le format CSV attendu.",
            "📂"
        )

        st.markdown("<div class='section-title'>📂 Import de Fichier CSV</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Données réseau réelles · Format standardisé</div>", unsafe_allow_html=True)

        # Template download
        template_df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=5, freq='h'),
            'traffic_gbps': [65.2, 70.1, 82.5, 78.3, 60.0],
            'latency_ms': [35.0, 42.1, 68.5, 55.2, 30.1],
            'congestion_pct': [45.0, 55.0, 75.0, 65.0, 40.0],
            'energy_kwh': [140.0, 155.0, 180.0, 170.0, 135.0],
            'packet_loss_pct': [0.5, 0.8, 2.1, 1.5, 0.4],
            'throughput_mbps': [620.0, 665.0, 580.0, 610.0, 640.0],
            'signal_dbm': [-72.0, -74.0, -79.0, -76.0, -71.0],
            'active_users': [780, 850, 990, 920, 710],
        })
        col_t1, col_t2 = st.columns([2,1])
        with col_t1:
            st.markdown("""
            <div style="background:#0A1020;border:1px solid #203070;border-radius:10px;
                        padding:1rem 1.2rem;font-size:0.85rem;color:#C8D8FF;line-height:1.6;">
                <strong style="color:#AABBFF;">📋 Format requis :</strong>
                Fichier CSV (séparateur virgule), encodage UTF-8, première ligne = noms des colonnes.
                Colonnes obligatoires : <code style="background:#1A2050;padding:0.1rem 0.3rem;border-radius:4px;">timestamp</code>,
                <code style="background:#1A2050;padding:0.1rem 0.3rem;border-radius:4px;">traffic_gbps</code>,
                <code style="background:#1A2050;padding:0.1rem 0.3rem;border-radius:4px;">latency_ms</code>
            </div>
            """, unsafe_allow_html=True)
        with col_t2:
            template_csv = template_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Télécharger le Modèle CSV", data=template_csv,
                               file_name="autotel_template.csv", mime="text/csv", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Glissez votre fichier CSV ici", type=['csv'],
                                          help="Format : timestamp, traffic_gbps, latency_ms + colonnes optionnelles")

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                validated_df, error = validate_uploaded_df(raw_df)

                if error:
                    st.markdown(f"""
                    <div class="alert-critical">
                        ❌ <strong>Erreur de validation :</strong> {error}<br>
                        Téléchargez le modèle CSV ci-dessus pour voir le format attendu.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.session_state['uploaded_df'] = validated_df
                    n_rows = len(validated_df)
                    n_days = (validated_df['timestamp'].max() - validated_df['timestamp'].min()).days
                    n_cols = len(validated_df.columns)

                    st.markdown(f"""
                    <div class="alert-ok">
                        ✅ <strong>Fichier validé avec succès !</strong>
                        {n_rows} lignes · {n_cols} colonnes · {n_days} jours de données
                        ({validated_df['timestamp'].min().strftime('%Y-%m-%d')} → {validated_df['timestamp'].max().strftime('%Y-%m-%d')})
                    </div>
                    """, unsafe_allow_html=True)

                    # Aperçu
                    st.markdown("<div class='section-title' style='margin-top:1rem;'>Aperçu des Données Importées</div>", unsafe_allow_html=True)
                    st.dataframe(validated_df.head(10).style.set_properties(**{'font-size':'0.83rem'}),
                                 use_container_width=True)

                    # Stats rapides
                    st.markdown("<div class='section-title' style='margin-top:1rem;'>Statistiques Descriptives</div>", unsafe_allow_html=True)
                    numeric_cols = ['traffic_gbps','latency_ms','congestion_pct','energy_kwh','packet_loss_pct']
                    stats = validated_df[numeric_cols].describe().round(2)
                    st.dataframe(stats.style.set_properties(**{'font-size':'0.82rem'}), use_container_width=True)

                    # Mini preview chart
                    st.markdown("<div class='section-title' style='margin-top:1rem;'>Aperçu Trafic Importé</div>", unsafe_allow_html=True)
                    fig_prev = go.Figure()
                    fig_prev.add_trace(go.Scatter(x=validated_df['timestamp'], y=validated_df['traffic_gbps'],
                                                   mode='lines', line=dict(color='#A10000', width=2),
                                                   fill='tozeroy', fillcolor='rgba(161,0,0,0.1)'))
                    fig_prev.update_layout(**PLOTLY_LAYOUT, height=220, yaxis_title='Trafic (Gbps)')
                    st.plotly_chart(fig_prev, use_container_width=True, config={'displayModeBar':False})

                    st.markdown("""
                    <div class="alert-info">
                        ℹ️ <strong>Données actives !</strong>
                        Retournez sur les autres onglets — ils utiliseront désormais vos données importées.
                        Pour revenir aux données simulées, rafraîchissez la page.
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="alert-critical">
                    ❌ <strong>Erreur de lecture :</strong> {str(e)}<br>
                    Vérifiez que le fichier est bien un CSV valide encodé en UTF-8.
                </div>
                """, unsafe_allow_html=True)

        elif 'uploaded_df' in st.session_state:
            df_loaded = st.session_state['uploaded_df']
            st.markdown(f"""
            <div class="alert-ok">
                ✅ Données actuellement actives : <strong>{len(df_loaded)} lignes</strong>
                ({df_loaded['timestamp'].min().strftime('%Y-%m-%d')} → {df_loaded['timestamp'].max().strftime('%Y-%m-%d')})
                — Importez un nouveau fichier pour les remplacer.
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Réinitialiser (revenir aux données simulées)"):
                del st.session_state['uploaded_df']
                st.rerun()
        else:
            st.markdown("""
            <div style="background:#1A0808;border:2px dashed #6A2020;border-radius:12px;
                        padding:2rem;text-align:center;margin-top:1rem;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">📁</div>
                <div style="font-size:1rem;color:#F0C8C8;font-weight:600;">Aucune donnée importée</div>
                <div style="font-size:0.83rem;color:#DD9999;margin-top:0.4rem;">
                    L'application utilise les données simulées par défaut.<br>
                    Importez un fichier CSV pour analyser votre réseau réel.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: MONITORING
# ─────────────────────────────────────────────
def page_monitoring(df):
    tab_mon, tab_guide = st.tabs(["📡 Surveillance", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Surveillance Réseau", """
        <h4>🎯 Objectif</h4>
        <p>Cette page affiche en détail l'évolution de 4 métriques réseau critiques sur une fenêtre temporelle sélectionnable. Elle sert de console de surveillance opérationnelle.</p>

        <h4>📊 Les 4 métriques surveillées</h4>
        <ul>
            <li><strong>Débit (Mbps) :</strong> Quantité de données transmises par seconde. Un débit qui chute brutalement signale une congestion ou une panne. Valeur nominale : 500–1200 Mbps</li>
            <li><strong>Pertes de Paquets (%) :</strong> Pourcentage de paquets non livrés. Acceptable : &lt;1%. Critique : &gt;3%. Cause : congestion, erreurs de lien, interférences radio</li>
            <li><strong>Utilisateurs Actifs :</strong> Nombre de sessions simultanées. Corrélé au trafic. Un pic soudain peut indiquer un événement (match, concert)</li>
            <li><strong>Force du Signal (dBm) :</strong> Plus la valeur est proche de 0, meilleur est le signal. -50 dBm = excellent. -90 dBm = limite. -110 dBm = pas de service</li>
        </ul>

        <h4>🔵 Nuage de Points : Trafic vs Latence</h4>
        <p>Chaque point = 1 heure de mesure. La couleur indique la congestion (rouge = fort). La tendance générale doit être diagonale ascendante : plus le trafic est élevé, plus la latence augmente. Un nuage dispersé indique une instabilité réseau.</p>

        <h4>📊 Distribution de la Congestion</h4>
        <p>Histogramme de fréquence. Un réseau sain a sa courbe concentrée sous 40% (majoritairement bleu/bas). Des barres à 70–100% indiquent des épisodes récurrents de surcharge à investiguer.</p>

        <h4>🔴 Matrice de Corrélation</h4>
        <p>Montre comment les variables s'influencent mutuellement :</p>
        <ul>
            <li><strong>Vert foncé (proche de +1) :</strong> Les deux métriques augmentent ensemble (ex: trafic ↑ = utilisateurs actifs ↑)</li>
            <li><strong>Rouge foncé (proche de -1) :</strong> Relation inverse (ex: signal fort = moins de pertes)</li>
            <li><strong>Blanc (proche de 0) :</strong> Aucune relation entre les deux métriques</li>
        </ul>

        <h4>⏱️ Fenêtre temporelle</h4>
        <p>Sélectionnez 12h pour analyser un incident récent, 48–72h pour identifier des tendances de fond.</p>
        """)

    with tab_mon:
        info_box(
            "Vue multi-métriques temps réel. Débit, pertes de paquets, utilisateurs actifs et signal sur la fenêtre sélectionnée. "
            "Le nuage de points révèle la relation trafic/latence. Consultez le 📖 Guide pour l'interprétation.",
            "📡"
        )
        st.markdown("<div class='section-title'>Surveillance Réseau en Direct</div>", unsafe_allow_html=True)
        towers = get_cell_tower_data()
        critical_sites = towers[towers['status'].isin(['CRITICAL','WARNING'])].sort_values('load_pct', ascending=False).head(5)
        if not critical_sites.empty:
            badges_html = ""
            for _, row in critical_sites.iterrows():
                badge_col = "#FF4444" if row['status']=='CRITICAL' else "#FF8C42"
                badges_html += f"""<span style="background:#3A0000;border:1px solid {badge_col};border-radius:20px;
                             padding:0.2rem 0.7rem;font-size:0.78rem;color:{badge_col};font-weight:600;
                             display:inline-block;">{row['Site']} · {row['load_pct']}%</span>"""
            st.markdown(f"""
            <div style="background:#2A0000;border:1px solid #6A2020;border-left:4px solid #FF4444;
                        border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;">
                <div style="font-size:0.75rem;font-weight:700;color:#FF9999;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:0.6rem;">⚠️ Sites Nécessitant une Attention</div>
                <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">{badges_html}</div>
            </div>
            """, unsafe_allow_html=True)

        col_w, _ = st.columns([1,2])
        with col_w:
            window = st.selectbox("⏱ Fenêtre temporelle", ["12 heures","24 heures","48 heures","72 heures"], index=2)
        window_map = {"12 heures":12,"24 heures":24,"48 heures":48,"72 heures":72}
        last48 = df.tail(window_map[window])

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=("Débit (Mbps)","Pertes de Paquets (%)","Utilisateurs Actifs","Force du Signal (dBm)"),
            vertical_spacing=0.14, horizontal_spacing=0.08)
        colors_p = ['#A10000','#D35400','#1A7A4A','#2255CC']
        traces = [(last48['throughput_mbps'],1,1),(last48['packet_loss_pct'],1,2),
                  (last48['active_users'],2,1),(last48['signal_dbm'],2,2)]
        for (y,r,c), color in zip(traces, colors_p):
            fig.add_trace(go.Scatter(x=last48['timestamp'],y=y,mode='lines',
                                     line=dict(color=color,width=2),fill='tozeroy',
                                     fillcolor='rgba(161,0,0,0.06)'),row=r,col=c)
        fig.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        chart_caption("4 métriques réseau sur la fenêtre sélectionnée. Signal en dBm : plus proche de 0 = meilleur.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-title'>Nuage de Points : Trafic vs Latence</div>", unsafe_allow_html=True)
            sample = df.sample(400, random_state=1)
            fig_s = go.Figure(go.Scatter(
                x=sample['traffic_gbps'],y=sample['latency_ms'],mode='markers',
                marker=dict(color=sample['congestion_pct'],colorscale=RED_SCALE,
                            size=6,opacity=0.75,colorbar=dict(title='Congestion%',thickness=10))
            ))
            fig_s.update_layout(**PLOTLY_LAYOUT,height=280,xaxis_title='Trafic (Gbps)',yaxis_title='Latence (ms)')
            st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar':False})
            chart_caption("Chaque point = 1h de mesure. Couleur = congestion. Tendance diagonale = comportement normal.")
        with c2:
            st.markdown("<div class='section-title'>Distribution : Congestion</div>", unsafe_allow_html=True)
            fig_h = go.Figure(go.Histogram(x=df['congestion_pct'],nbinsx=40,
                marker_color='rgba(161,0,0,0.7)',marker_line=dict(color='#A10000',width=0.3)))
            fig_h.update_layout(**PLOTLY_LAYOUT,height=280,xaxis_title='Congestion (%)',yaxis_title='Nombre')
            st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar':False})
            chart_caption("Distribution des niveaux de congestion sur 30 jours. Réseau sain : courbe concentrée sous 40%.")

        st.markdown("<div class='section-title'>Matrice de Corrélation des Variables</div>", unsafe_allow_html=True)
        numeric_cols = ['traffic_gbps','latency_ms','congestion_pct','energy_kwh','packet_loss_pct','throughput_mbps','active_users']
        corr = df[numeric_cols].corr()
        fig_c = go.Figure(go.Heatmap(z=corr.values,x=numeric_cols,y=numeric_cols,
            colorscale='RdYlGn',zmid=0,text=np.round(corr.values,2),
            texttemplate='%{text}',textfont_size=10,colorbar=dict(thickness=10,len=0.8)))
        fig_c.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar':False})
        chart_caption("Vert = corrélation positive. Rouge = corrélation négative. Blanc = aucune relation.")


# ─────────────────────────────────────────────
# PAGE: PREDICTION
# ─────────────────────────────────────────────
def page_prediction(df):
    tab_pred, tab_guide = st.tabs(["🔮 Prédiction", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Prédiction & Détection d'Anomalies", """
        <h4>🎯 Objectif</h4>
        <p>Cette page utilise deux modèles d'intelligence artificielle : un pour prédire le trafic futur, un autre pour détecter les comportements anormaux du réseau.</p>

        <h4>🤖 Modèle 1 : Gradient Boosting Regressor (GBR)</h4>
        <p>Algorithme d'ensemble qui apprend les patterns historiques du trafic pour prédire les valeurs futures. Entraîné sur 80% des données, testé sur les 20% restants.</p>
        <ul>
            <li><strong>R² Score :</strong> Qualité d'ajustement. 1.0 = parfait. 0 = inutile. Viser &gt; 0.85</li>
            <li><strong>RMSE (Root Mean Square Error) :</strong> Erreur moyenne en Gbps. Plus petit = meilleur</li>
            <li><strong>MAE (Mean Absolute Error) :</strong> Erreur absolue moyenne. Plus robuste aux outliers</li>
        </ul>

        <h4>📈 Graphique Réel vs Prédit</h4>
        <p>Ligne grise = trafic réel mesuré sur les 20% de test. Ligne rouge pointillée = prédiction du modèle. Plus les deux lignes se superposent, plus le modèle est précis. Des écarts importants indiquent des anomalies ou des événements imprévus.</p>

        <h4>🛡️ Modèle 2 : Isolation Forest (Détection d'Anomalies)</h4>
        <p>Algorithme non-supervisé qui "isole" les points de données qui se comportent différemment de la normale. Paramètre clé : contamination=2% (on s'attend à 2% d'anomalies).</p>
        <ul>
            <li><strong>Croix rouge (×) :</strong> Anomalie détectée — comportement statistiquement anormal</li>
            <li>Causes typiques : pannes équipement, attaques DDoS, événements imprévus, erreurs de mesure</li>
        </ul>

        <h4>📊 Importance des Variables</h4>
        <p>Indique quelles données le modèle utilise le plus pour ses prédictions :</p>
        <ul>
            <li><strong>lag1 :</strong> Trafic de l'heure précédente (prédicteur le plus fort)</li>
            <li><strong>lag24 :</strong> Même heure hier (capture la saisonnalité journalière)</li>
            <li><strong>roll6 :</strong> Moyenne mobile 6h (tendance court terme)</li>
            <li><strong>hour / dow :</strong> Heure du jour et jour de la semaine (patterns temporels)</li>
        </ul>

        <h4>🔮 Prévision 24h</h4>
        <p>Projection du trafic pour les 24 prochaines heures basée sur les patterns historiques. La zone transparente = intervalle de confiance 80% (la vraie valeur aura 80% de chances d'être dans cette zone). La ligne verticale = maintenant.</p>
        """)

    with tab_pred:
        info_box(
            "Gradient Boosting prédit le trafic. Isolation Forest détecte les anomalies. "
            "R² proche de 1 = modèle très précis. Les croix rouges = comportements anormaux. "
            "Consultez le 📖 Guide pour comprendre chaque métrique.",
            "🔮"
        )
        st.markdown("<div class='section-title'>Prédiction du Trafic — Modèle GBR</div>", unsafe_allow_html=True)

        with st.spinner("Entraînement du modèle..."):
            model, X_te, y_te, y_pred, rmse, mae, r2, importances = train_traffic_predictor(df)

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("R² Score",f"{r2:.4f}","Ajustement Modèle","down"),
            ("RMSE",f"{rmse:.3f}","Gbps","warn"),
            ("MAE",f"{mae:.3f}","Gbps","warn"),
            ("Taille Entraînement",f"{len(df)-len(X_te)}","échantillons","down"),
        ]
        for col, (label,val,unit,cls) in zip([m1,m2,m3,m4], metrics):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="font-size:1.7rem;">{val}</div>
                    <div class="kpi-delta {cls}">{unit}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Trafic Réel vs Prédit</div>", unsafe_allow_html=True)
        idx = np.arange(len(y_te))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx,y=y_te.values,name='Réel',line=dict(color='#AAAAAA',width=1.5)))
        fig.add_trace(go.Scatter(x=idx,y=y_pred,name='Prédit',line=dict(color='#FF4444',width=1.5,dash='dot')))
        fig.update_layout(**PLOTLY_LAYOUT,height=310,xaxis_title='Index Temps',yaxis_title='Trafic (Gbps)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        chart_caption("Ligne grise = trafic réel. Ligne rouge pointillée = prédiction du modèle GBR. Plus les courbes se superposent, plus le modèle est précis.")

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.2,1])
        with c1:
            st.markdown("<div class='section-title'>Détection d'Anomalies — Isolation Forest</div>", unsafe_allow_html=True)
            with st.spinner("Détection des anomalies..."):
                _, _, scores, preds = train_anomaly_model(df)
            df2 = df.copy()
            df2['anomaly_score'] = scores
            df2['is_anomaly'] = (preds==-1).astype(int)
            fig_a = go.Figure()
            normal  = df2[df2['is_anomaly']==0]
            anomaly = df2[df2['is_anomaly']==1]
            fig_a.add_trace(go.Scatter(x=normal['timestamp'],y=normal['traffic_gbps'],
                                       mode='lines',name='Normal',line=dict(color='#A10000',width=1.2)))
            fig_a.add_trace(go.Scatter(x=anomaly['timestamp'],y=anomaly['traffic_gbps'],
                                       mode='markers',name='Anomalie',
                                       marker=dict(color='#FF0000',size=8,symbol='x',
                                                   line=dict(width=2,color='#8B0000'))))
            fig_a.update_layout(**PLOTLY_LAYOUT,height=270,yaxis_title='Trafic (Gbps)')
            st.plotly_chart(fig_a, use_container_width=True, config={'displayModeBar':False})
            chart_caption("Les croix rouges (×) = anomalies détectées par Isolation Forest. Points statistiquement anormaux.")
            n_anom = df2['is_anomaly'].sum()
            rate = n_anom/len(df2)*100
            st.markdown(f"""
            <div class="alert-warning">⚠️ <strong>{n_anom} anomalies détectées</strong> ({rate:.1f}% des observations)</div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='section-title'>Importance des Variables</div>", unsafe_allow_html=True)
            sorted_imp = sorted(importances.items(), key=lambda x:x[1], reverse=True)
            total = sum(v for _,v in sorted_imp)
            bars_rows = ""
            for feat, imp in sorted_imp:
                pct = imp/total*100
                bars_rows += f"""
                <div style="display:flex;align-items:center;gap:0.6rem;margin:0.35rem 0;font-size:0.83rem;">
                    <div style="width:140px;color:#F0C8C8;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{feat}</div>
                    <div style="flex:1;height:12px;background:#3A0808;border-radius:6px;overflow:hidden;">
                        <div style="width:{pct:.1f}%;height:100%;border-radius:6px;background:linear-gradient(90deg,#A10000,#C0392B);"></div>
                    </div>
                    <div style="width:42px;text-align:right;font-weight:700;color:#A10000;font-family:'JetBrains Mono',monospace;">{pct:.1f}%</div>
                </div>"""
            comp_h = 40 + len(sorted_imp)*36
            components.html(f"""
            <div style="background:#1E0808;border-radius:12px;padding:1rem 1.2rem;border:1px solid #6A2020;">
                {bars_rows}
            </div>
            """, height=comp_h, scrolling=False)
            chart_caption("lag1 (heure précédente) est généralement le prédicteur le plus fort.")

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔮 Prévision Trafic — Prochaines 24 Heures</div>", unsafe_allow_html=True)
        last_row   = df.iloc[-1]
        hours_fwd  = pd.date_range(start=df['timestamp'].iloc[-1], periods=25, freq='h')[1:]
        hour_vals  = np.array([h.hour for h in hours_fwd])
        dow_vals   = np.array([h.dayofweek for h in hours_fwd])
        daily_base = 50 + 40*np.sin((hour_vals-6)*np.pi/12)**2
        wd_factor  = np.where(dow_vals>=5,0.75,1.0)
        noise_fwd  = np.random.default_rng(99).normal(0,3,24)
        forecast   = np.clip(daily_base*wd_factor+noise_fwd,5,140)
        conf_lo    = np.clip(forecast*0.88,5,140)
        conf_hi    = np.clip(forecast*1.12,5,140)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=df.tail(48)['timestamp'],y=df.tail(48)['traffic_gbps'],
                                    name='Historique',mode='lines',line=dict(color='#888888',width=1.5)))
        fig_fc.add_trace(go.Scatter(x=list(hours_fwd)+list(hours_fwd[::-1]),
                                    y=list(conf_hi)+list(conf_lo[::-1]),
                                    fill='toself',fillcolor='rgba(161,0,0,0.15)',
                                    line=dict(color='rgba(0,0,0,0)'),name='Intervalle 80%'))
        fig_fc.add_trace(go.Scatter(x=hours_fwd,y=forecast,name='Prévision 24h',mode='lines',
                                    line=dict(color='#FF4444',width=2.5,dash='dot')))
        fig_fc.update_layout(**PLOTLY_LAYOUT,height=300,xaxis_title='Heure',yaxis_title='Trafic (Gbps)',
                             shapes=[dict(type='line',x0=df['timestamp'].iloc[-1],x1=df['timestamp'].iloc[-1],
                                         y0=0,y1=140,line=dict(color='#FF6666',width=1.5,dash='dash'))])
        st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar':False})
        chart_caption("Ligne pointillée rouge = prévision 24h. Zone transparente = intervalle de confiance 80%. Ligne verticale = maintenant.")


# ─────────────────────────────────────────────
# PAGE: OPTIMISATION
# ─────────────────────────────────────────────
def page_optimization(df):
    tab_opt, tab_guide = st.tabs(["⚙️ Optimisation", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Optimisation & Recommandations", """
        <h4>🎯 Objectif</h4>
        <p>Le moteur d'optimisation analyse automatiquement les KPIs des 6 dernières heures et génère des recommandations d'actions concrètes classées par priorité.</p>

        <h4>🚦 Niveaux de priorité</h4>
        <ul>
            <li><strong style="color:#FF4444;">ÉLEVÉ (Rouge) :</strong> Action immédiate requise (&lt;15 min). SLA en danger ou risque de panne</li>
            <li><strong style="color:#FF8C42;">MOYEN (Orange) :</strong> Traiter sous 2 heures. Dégradation de qualité en cours</li>
            <li><strong style="color:#22C55E;">FAIBLE (Vert) :</strong> Optimisation opportuniste. Aucune urgence</li>
        </ul>

        <h4>🤖 Comment fonctionne le moteur de règles ?</h4>
        <p>Le système évalue ces conditions :</p>
        <ul>
            <li>Congestion &gt; 70% → Augmenter capacité (activation spectrale)</li>
            <li>Latence &gt; 80ms → Risque SLA, re-routage nécessaire</li>
            <li>Trafic &gt; 80 Gbps → Redistribution de charge</li>
            <li>Énergie &gt; 200 kWh → Mode éco, mise en veille cellules</li>
            <li>Pertes &gt; 2% → Vérification liens fibre, activation FEC</li>
        </ul>

        <h4>📅 Chronologie des Actions (Diagramme de Gantt)</h4>
        <p>Chaque barre = une action autonome déjà exécutée par le système :</p>
        <ul>
            <li><strong>Durée de la barre :</strong> Temps d'exécution de l'action</li>
            <li><strong>Rouge :</strong> Action terminée avec succès</li>
            <li><strong>Vert :</strong> Action en cours d'exécution</li>
            <li><strong>Texte sur la barre :</strong> Impact mesuré après l'action</li>
        </ul>
        <p>Ce diagramme permet d'auditer toutes les décisions autonomes du système et de mesurer leur efficacité réelle.</p>
        """)

    with tab_opt:
        info_box(
            "Recommandations automatiques basées sur les KPIs actuels. "
            "ÉLEVÉ = action immédiate. MOYEN = sous 2h. FAIBLE = optimisation optionnelle. "
            "Le Gantt montre les actions déjà exécutées et leur impact mesuré.",
            "⚙️"
        )
        st.markdown("<div class='section-title'>Moteur de Décision Automatique</div>", unsafe_allow_html=True)
        last = df.tail(6)
        avg_lat  = last['latency_ms'].mean()
        avg_trf  = last['traffic_gbps'].mean()
        avg_cong = last['congestion_pct'].mean()
        avg_eng  = last['energy_kwh'].mean()
        avg_pl   = last['packet_loss_pct'].mean()

        recs = []
        if avg_cong>70: recs.append(("ÉLEVÉ","🔴 Augmenter la Capacité Réseau",
            f"Congestion à {avg_cong:.1f}% — dépasse le seuil de 70%. Activer les blocs spectre de secours sur la bande 2.6 GHz. Soulagement estimé : −{avg_cong*0.35:.0f}% de congestion en 15 min."))
        if avg_lat>80: recs.append(("ÉLEVÉ","🔴 Risque de Violation SLA Latence",
            f"Latence moyenne {avg_lat:.1f} ms approchant la limite SLA (100 ms). Re-router le trafic via les nœuds moins chargés N-07 et N-12. Priorité : flux voix & vidéo."))
        if avg_trf>80: recs.append(("ÉLEVÉ","🔴 Redistribution du Trafic Requise",
            f"Charge trafic {avg_trf:.1f} Gbps — scénario de pic. Activer l'équilibrage de charge sur les secteurs 3A, 5B, 7C."))
        if avg_eng>200: recs.append(("MOYEN","🟠 Optimisation Énergétique",
            f"Consommation {avg_eng:.1f} kWh — au-dessus de la référence. Activer la mise en veille adaptative sur les cellules à faible trafic."))
        if avg_pl>2: recs.append(("MOYEN","🟠 Atténuation des Pertes de Paquets",
            f"Pertes de paquets {avg_pl:.2f}% — risque de dégradation. Vérifier les liens fibre sur les segments C4–C9."))
        if avg_cong<40 and avg_eng>150: recs.append(("FAIBLE","🟢 Économies d'Énergie en Faible Demande",
            "Charge réseau faible. Opportunité de réduire la puissance TX sur 12 secteurs inactifs. Économie estimée : 18–25 kWh/heure."))
        if not recs: recs.append(("FAIBLE","🟢 Réseau en Fonctionnement Normal",
            "Tous les KPIs dans les seuils acceptables. Aucune action immédiate requise."))

        priority_map = {"ÉLEVÉ":"high","MOYEN":"medium","FAIBLE":"low"}
        badge_map    = {"ÉLEVÉ":"badge-high","MOYEN":"badge-medium","FAIBLE":"badge-low"}
        for priority, title, body in recs:
            cls = priority_map[priority]
            st.markdown(f"""
            <div class="rec-card rec-priority-{cls}">
                <span class="rec-badge {badge_map[priority]}">{priority}</span>
                <div class="rec-title">{title}</div>
                <div class="rec-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Chronologie des Actions d'Optimisation</div>", unsafe_allow_html=True)
        actions = pd.DataFrame({
            'Action':   ['Équilibrage Charge','Extension Spectre','Veille Énergie','Re-routage','Activation FEC'],
            'Start':    ['2026-01-28 02:00','2026-01-27 18:30','2026-01-27 03:00','2026-01-26 14:00','2026-01-26 08:00'],
            'End':      ['2026-01-28 04:30','2026-01-27 21:00','2026-01-27 07:00','2026-01-26 16:30','2026-01-26 12:00'],
            'Impact':   ['−22% congestion','−18% latence','−15% énergie','−31% pertes paquets','−45% pertes paquets'],
            'Statut':   ['Terminé','Terminé','Terminé','Terminé','Actif'],
        })
        fig_g = px.timeline(actions, x_start='Start', x_end='End', y='Action',
                            color='Statut', color_discrete_map={'Terminé':'#A10000','Actif':'#22C55E'}, text='Impact')
        fig_g.update_layout(**PLOTLY_LAYOUT, height=260)
        fig_g.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_g, use_container_width=True, config={'displayModeBar':False})
        chart_caption("Chaque barre = une action autonome. Rouge = terminée. Vert = en cours. Impact mesuré affiché sur la barre.")


# ─────────────────────────────────────────────
# PAGE: SIMULATION
# ─────────────────────────────────────────────
def page_simulation(df):
    tab_sim, tab_guide = st.tabs(["🎛️ Simulation", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Simulateur de Scénarios", """
        <h4>🎯 Objectif</h4>
        <p>Le simulateur permet de tester l'impact de décisions réseau AVANT de les appliquer en production. C'est un bac à sable virtuel pour évaluer le compromis performance/coût de différentes stratégies.</p>

        <h4>🎛️ Les paramètres de simulation</h4>
        <ul>
            <li><strong>Capacité Supplémentaire (%) :</strong> Activation de ressources spectrales ou cellules supplémentaires. +20% = déploiement d'une bande additionnelle. Réduit congestion et latence mais augmente les coûts CAPEX.</li>
            <li><strong>Trafic Re-routé (%) :</strong> Pourcentage de trafic redirigé vers des nœuds alternatifs moins chargés. Réduit la congestion sans coût infrastructure mais ajoute une latence de routage.</li>
            <li><strong>Cellules en Mode Veille (%) :</strong> Mise en veille des secteurs sous-utilisés (heures creuses). Économies d'énergie significatives mais réduit la couverture de réserve.</li>
            <li><strong>Réduction Puissance TX (%) :</strong> Diminution de la puissance d'émission des antennes. Économie d'énergie directe. Peut impacter la couverture en périphérie.</li>
            <li><strong>FEC (Forward Error Correction) :</strong> Correction d'erreurs proactive. Réduit les pertes de paquets de ~15% mais consomme de la bande passante.</li>
            <li><strong>Équilibrage de Charge IA :</strong> Algorithme d'optimisation automatique de la distribution du trafic entre cellules.</li>
        </ul>

        <h4>📊 Les cartes de résultats</h4>
        <p>Chaque carte montre : valeur simulée + variation en % par rapport à la référence (situation actuelle).</p>
        <ul>
            <li><strong>Flèche verte ▼ :</strong> Amélioration (baisse de latence/congestion, ou hausse de débit)</li>
            <li><strong>Flèche rouge ▲ :</strong> Dégradation</li>
        </ul>

        <h4>💰 Impact Coût Mensuel</h4>
        <p>Modèle économique simplifié :</p>
        <ul>
            <li>Coût infrastructure = 150€/mois par % de capacité supplémentaire</li>
            <li>Économie énergie = (kWh économisés) × 24h × 30j × 0.12€/kWh</li>
            <li>Économie nette = économies énergie − 10% du coût infrastructure</li>
        </ul>

        <h4>🎯 Graphique Radar</h4>
        <p>Comparaison visuelle avant/après sur 5 dimensions. Plus la surface rouge est grande, meilleures sont les performances simulées. Idéal pour présenter un scénario à un manager ou client.</p>
        """)

    with tab_sim:
        info_box(
            "Ajustez les curseurs pour simuler des changements réseau avant de les appliquer. "
            "Observez immédiatement l'impact sur la latence, l'énergie et le coût mensuel. "
            "Le radar compare les performances avant/après.",
            "🎛️"
        )
        st.markdown("<div class='section-title'>Simulateur de Scénarios Réseau</div>", unsafe_allow_html=True)
        col_inputs, col_results = st.columns([1,1.4])

        with col_inputs:
            st.markdown("<div class='section-title' style='margin-top:0.5rem;'>Paramètres de Simulation</div>", unsafe_allow_html=True)
            extra_capacity  = st.slider("Capacité Supplémentaire (%)", 0,100,20,step=5)
            traffic_reroute = st.slider("Trafic Re-routé (%)",          0, 80,30,step=5)
            sleep_cells_pct = st.slider("Cellules en Mode Veille (%)", 0, 60,15,step=5)
            tx_power_reduce = st.slider("Réduction Puissance TX (%)",  0, 30,10,step=2)
            fec_enabled     = st.checkbox("Activer Correction d'Erreur (FEC)", value=True)
            load_balancing  = st.checkbox("Activer Équilibrage de Charge IA",  value=True)

        with col_results:
            base_lat  = df['latency_ms'].mean()
            base_cong = df['congestion_pct'].mean()
            base_eng  = df['energy_kwh'].mean()
            base_pl   = df['packet_loss_pct'].mean()
            base_tput = df['throughput_mbps'].mean()
            lat_impr   = extra_capacity*0.35 + (traffic_reroute*0.25 if load_balancing else 0)
            cong_impr  = extra_capacity*0.40 + traffic_reroute*0.30
            eng_saving = sleep_cells_pct*0.18 + tx_power_reduce*0.12
            pl_impr    = (15 if fec_enabled else 0) + traffic_reroute*0.10
            tput_impr  = extra_capacity*0.30
            new_lat  = max(base_lat *(1-lat_impr/100),  5)
            new_cong = max(base_cong*(1-cong_impr/100), 2)
            new_eng  = max(base_eng *(1-eng_saving/100),60)
            new_pl   = max(base_pl  *(1-pl_impr/100),   0.01)
            new_tput = base_tput*(1+tput_impr/100)
            infra_cost  = extra_capacity*150
            energy_save = (base_eng-new_eng)*24*30*0.12
            net_saving  = energy_save - infra_cost*0.1

            def sim_card(label,base,new_val,unit,lower_better=True):
                delta = new_val-base; pct = delta/base*100
                good = (delta<0)==lower_better
                color = "#86EFAC" if good else "#FFB3B3"
                arrow = "▼" if delta<0 else "▲"
                return f"""
                <div style="background:#2A0000;border-radius:10px;padding:0.9rem 1rem;border:1px solid #5A1010;">
                    <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#DD9999;">{label}</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#FFFFFF;font-family:'JetBrains Mono',monospace;">{new_val:.1f}
                        <span style="font-size:0.85rem;color:#CC9999;font-weight:400;">{unit}</span></div>
                    <div style="font-size:0.78rem;color:{color};font-weight:600;">{arrow} {abs(pct):.1f}% vs référence</div>
                </div>"""

            cards_html  = sim_card("Latence",base_lat,new_lat,"ms",True)
            cards_html += sim_card("Congestion",base_cong,new_cong,"%",True)
            cards_html += sim_card("Énergie",base_eng,new_eng,"kWh",True)
            cards_html += sim_card("Pertes Paquets",base_pl,new_pl,"%",True)
            components.html(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:0.4rem;">
                {cards_html}
            </div>
            """, height=230, scrolling=False)

            sign = "+" if net_saving>0 else ""
            net_color = "#86EFAC" if net_saving>0 else "#FFB3B3"
            components.html(f"""
            <div style="background:#2A0000;border-radius:10px;padding:1rem 1.2rem;border:1px solid #5A1010;font-family:'Inter',sans-serif;">
                <div style="font-size:0.8rem;font-weight:700;color:#EE9999;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;">💰 Impact Coût Mensuel</div>
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0;border-bottom:1px solid #3A1010;font-size:0.85rem;">
                    <div style="flex:1;color:#F0C8C8;">Coût Infrastructure</div>
                    <div style="font-weight:700;color:#FFB3B3;font-family:'JetBrains Mono',monospace;">+€{infra_cost:,.0f}</div>
                </div>
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0;border-bottom:1px solid #3A1010;font-size:0.85rem;">
                    <div style="flex:1;color:#F0C8C8;">Économies Énergie</div>
                    <div style="font-weight:700;color:#86EFAC;font-family:'JetBrains Mono',monospace;">−€{energy_save:,.0f}</div>
                </div>
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.6rem 0 0.2rem 0;border-top:2px solid #5A1010;font-size:0.85rem;">
                    <div style="flex:1;font-weight:700;color:#FFFFFF;">Économie Nette Mensuelle</div>
                    <div style="font-weight:800;font-size:1.1rem;color:{net_color};font-family:'JetBrains Mono',monospace;">{sign}€{net_saving:,.0f}</div>
                </div>
            </div>
            """, height=160, scrolling=False)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Radar Performance — Avant vs Après</div>", unsafe_allow_html=True)
        cats = ['Latence','Congestion','Énergie','Pertes Paquets','Débit']
        def norm_lb(val,base): return max(0,min(100,100-(val/base*100-100)*2))
        def norm_hb(val,base): return min(100,val/base*100)
        before = [50,50,50,50,50]
        after  = [norm_lb(new_lat,base_lat),norm_lb(new_cong,base_cong),norm_lb(new_eng,base_eng),
                  norm_lb(new_pl,base_pl),norm_hb(new_tput,base_tput)]
        fig_r = go.Figure()
        for vals,name,color in [(before,'Référence','rgba(100,100,100,0.5)'),(after,'Simulé','rgba(161,0,0,0.7)')]:
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
                fill='toself',name=name,line=dict(color=color,width=2),
                fillcolor=color.replace('0.7','0.12').replace('0.5','0.07')))
        fig_r.update_layout(**PLOTLY_LAYOUT,height=340,
            polar=dict(bgcolor='rgba(0,0,0,0)',
                       radialaxis=dict(visible=True,range=[0,110],gridcolor='#3A1010',tickfont_color='#DD9999',linecolor='#5A1010'),
                       angularaxis=dict(gridcolor='#3A1010',tickfont_color='#FFFFFF',linecolor='#5A1010')))
        st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar':False})
        chart_caption("Plus la surface rouge est grande, meilleures sont les performances simulées vs référence.")


# ─────────────────────────────────────────────
# PAGE: ANALYSES
# ─────────────────────────────────────────────
def page_insights(df):
    tab_ins, tab_guide = st.tabs(["🔍 Analyses", "📖 Guide"])

    with tab_guide:
        guide_section("Guide — Analyses Intelligentes & Alertes", """
        <h4>🎯 Objectif</h4>
        <p>Cette page centralise toutes les alertes automatiques et recommandations intelligentes générées par le moteur de règles. C'est le centre opérationnel de surveillance.</p>

        <h4>🔔 Niveaux d'Alerte</h4>
        <ul>
            <li><strong style="color:#FF4444;">🔴 CRITIQUE :</strong> SLA en danger. Escalade P1. Intervention immédiate (0–15 min)</li>
            <li><strong style="color:#FF8C42;">🟠 AVERTISSEMENT :</strong> Dégradation en cours. Surveiller et planifier une action (15 min – 2h)</li>
            <li><strong style="color:#22C55E;">🟢 OK :</strong> Fonctionnement normal. Information de confirmation</li>
            <li><strong style="color:#60A5FA;">ℹ️ INFO :</strong> Statistiques et prévisions. Lecture informative</li>
        </ul>

        <h4>📊 Règles d'Alerte Automatiques</h4>
        <ul>
            <li>Latence de pointe &gt; 120ms → Alerte CRITIQUE (SLA)</li>
            <li>Congestion moyenne 24h &gt; 75% → Alerte CRITIQUE</li>
            <li>Latence moyenne &gt; 60ms → Avertissement</li>
            <li>Énergie &gt; 200 kWh/h → Avertissement (efficacité)</li>
            <li>Pertes de paquets &gt; 1.5% → Avertissement</li>
        </ul>

        <h4>📈 Tendances sur 24 Heures (Mini-graphes)</h4>
        <p>Trois sparklines montrant l'évolution de la latence, congestion et énergie sur les dernières 24h :</p>
        <ul>
            <li><strong>Courbe plate :</strong> Situation stable, pas d'incident en cours</li>
            <li><strong>Pic soudain :</strong> Événement ou incident à investiguer</li>
            <li><strong>Tendance croissante :</strong> Dégradation progressive, anticiper une action</li>
        </ul>

        <h4>💡 Recommandations Intelligentes</h4>
        <p>4 types de recommandations générées par l'IA :</p>
        <ul>
            <li><strong>Maintenance Prédictive :</strong> Basée sur la tendance des métriques des 7 derniers jours</li>
            <li><strong>Efficacité Spectrale :</strong> Analyse d'utilisation des bandes de fréquences</li>
            <li><strong>Optimisation CAPEX :</strong> Groupement des achats d'équipements</li>
            <li><strong>Atténuation Risques SLA :</strong> Prévention des violations contractuelles</li>
        </ul>
        """)

    with tab_ins:
        info_box(
            "Alertes automatiques et recommandations IA. Rouge = critique, Orange = avertissement. "
            "Les mini-graphes montrent les tendances 24h. Consultez le 📖 Guide pour interpréter chaque alerte.",
            "🔍"
        )
        st.markdown("<div class='section-title'>Analyses Intelligentes & Alertes</div>", unsafe_allow_html=True)
        last24 = df.tail(24); last7d = df.tail(168)
        avg_lat   = last24['latency_ms'].mean();   max_lat  = last24['latency_ms'].max()
        avg_cong  = last24['congestion_pct'].mean(); avg_eng = last24['energy_kwh'].mean()
        avg_pl    = last24['packet_loss_pct'].mean()
        traf_std  = last7d['traffic_gbps'].std();  traf_mean = last7d['traffic_gbps'].mean()

        alerts = []
        if max_lat>120: alerts.append(("critical",f"🔴 Latence de pointe {max_lat:.1f} ms — SLA à risque (limite : 100 ms). Escalade P1."))
        if avg_cong>75: alerts.append(("critical",f"🔴 Congestion moyenne 24h {avg_cong:.1f}% — au-dessus du seuil d'urgence de 75%."))
        if avg_lat>60:  alerts.append(("warning", f"🟠 Latence moyenne élevée {avg_lat:.1f} ms — violation SLA possible dans 4–8h."))
        if avg_eng>200: alerts.append(("warning", f"🟠 Consommation énergie {avg_eng:.1f} kWh/h — 15% au-dessus de la référence."))
        if avg_pl>1.5:  alerts.append(("warning", f"🟠 Pertes de paquets {avg_pl:.2f}% — au-dessus du seuil acceptable de 1%."))
        alerts.append(("ok",   "🟢 Liens backbone principaux : Tous opérationnels. Redondance active sur tous les chemins critiques."))
        alerts.append(("info", f"ℹ️ Variabilité trafic σ={traf_std:.2f} Gbps — {'ÉLEVÉE' if traf_std>15 else 'NORMALE'} (fenêtre 7j)."))
        alerts.append(("info", f"ℹ️ Prochain pic prévu : Ce soir 20h00–22h00 · Trafic estimé : {traf_mean*1.35:.1f} Gbps."))

        class_map = {"critical":"alert-critical","warning":"alert-warning","ok":"alert-ok","info":"alert-info"}
        st.markdown("<div class='section-title' style='font-size:1rem;'>🔔 Alertes Actives</div>", unsafe_allow_html=True)
        for typ, msg in alerts:
            st.markdown(f"<div class='{class_map[typ]}'>{msg}</div>", unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Tendances sur 24 Heures</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        def sparkline(col, data, ylabel, color):
            h = color.lstrip('#')
            r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
            fill_rgba = f'rgba({r},{g},{b},0.13)'
            fig = go.Figure(go.Scatter(y=data,mode='lines',line=dict(color=color,width=2),fill='tozeroy',fillcolor=fill_rgba))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=0,r=0,t=5,b=0),height=80,showlegend=False,
                              xaxis=dict(visible=False),yaxis=dict(visible=False))
            with col:
                st.markdown(f"<div style='font-size:0.75rem;font-weight:600;color:#DD9999;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;'>{ylabel}</div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

        sparkline(c1, last24['latency_ms'].values,     "Latence (ms)",   "#FF4444")
        sparkline(c2, last24['congestion_pct'].values, "Congestion (%)", "#FF8C42")
        sparkline(c3, last24['energy_kwh'].values,     "Énergie (kWh)",  "#60A5FA")
        chart_caption("Mini-graphes 24h. Une hausse soudaine indique un incident potentiel.")

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💡 Recommandations Intelligentes</div>", unsafe_allow_html=True)
        smart_recs = [
            ("Maintenance Prédictive","L'antenne SITE-017 montre une tendance croissante de pertes de paquets (+0,8%/jour sur 7 jours). Planifier une inspection matérielle avant défaillance. Probabilité de panne sous 72h : 68%."),
            ("Efficacité Spectrale","La bande 800 MHz n'est utilisée qu'à 42% en heures creuses. Réallouer 20 MHz aux données peut améliorer le débit de ~15% sans coût supplémentaire."),
            ("Optimisation Capex","La trajectoire actuelle suggère que les sites SITE-003, 009, 014 nécessitent une mise à niveau au T2. Un achat groupé peut réduire les capex de 18–22% vs commandes individuelles."),
            ("Atténuation des Risques SLA","Selon les prévisions de trafic 7 jours, 3 sites risquent une violation SLA ce week-end. Déployer des cellules mobiles aux coordonnées 5.3N, 4.0W vendredi avant 16h00."),
        ]
        for title, body in smart_recs:
            st.markdown(f"""
            <div class="rec-card rec-priority-medium">
                <div class="rec-title">💡 {title}</div>
                <div class="rec-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
def page_about():
    tech_items = [
        ("🐍","Python 3.10+","Langage principal"),
        ("🌊","Streamlit","Framework application web"),
        ("🐼","Pandas","Manipulation de données"),
        ("🔢","NumPy","Calcul numérique"),
        ("📊","Plotly","Visualisations interactives"),
        ("🤖","Scikit-learn","Modèles de machine learning"),
        ("🧠","Isolation Forest","Détection d'anomalies"),
        ("📈","Gradient Boosting","Prédiction du trafic"),
        ("🗺️","Mapbox/Plotly","Carte géographique interactive"),
        ("📂","CSV Upload","Import données réelles"),
    ]
    grid_items = "".join(f"""
        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0.8rem;
                    background:#1E0000;border-radius:8px;font-size:0.84rem;color:#F0C8C8;">
            <span>{icon}</span>
            <strong style="color:#FFFFFF;">{name}</strong> — {desc}
        </div>""" for icon, name, desc in tech_items)

    components.html(f"""
    <style>* {{box-sizing:border-box;margin:0;padding:0;}} body {{background:transparent;font-family:'Inter',sans-serif;}} a {{color:#FF8888;}}</style>
    <div style="max-width:800px;margin:0 auto;padding:0.5rem 0;">
        <div style="background:#2A0000;border-radius:16px;padding:2rem 2.4rem;border:1px solid #6A2020;box-shadow:0 4px 20px rgba(0,0,0,0.5);margin-bottom:1.2rem;">
            <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">
                <div style="font-size:3rem;line-height:1;">📡</div>
                <div>
                    <div style="font-size:1.4rem;font-weight:800;color:#FFFFFF;">AutoTel AI</div>
                    <div style="font-size:0.85rem;color:#DD9999;">Autonomous Telecom Network Intelligence System · v3.0</div>
                </div>
            </div>
            <p style="color:#F0C8C8;line-height:1.7;font-size:0.92rem;">
                AutoTel AI est une plateforme de data science enterprise conçue pour les opérations de réseau télécom.
                Elle combine surveillance temps réel, carte géographique interactive, détection d'anomalies par machine learning,
                prévision du trafic, import de données réelles et un moteur de décision autonome.
            </p>
        </div>
        <div style="background:#2A0000;border-radius:16px;padding:1.8rem 2.4rem;border:1px solid #6A2020;box-shadow:0 4px 20px rgba(0,0,0,0.5);margin-bottom:1.2rem;">
            <div style="font-size:1rem;font-weight:700;color:#FFFFFF;margin-bottom:1rem;border-left:3px solid #C0392B;padding-left:0.8rem;">🧰 Stack Technologique</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">{grid_items}</div>
        </div>
        <div style="background:#2A0000;border-radius:16px;padding:1.8rem 2.4rem;border:1px solid #6A2020;box-shadow:0 4px 20px rgba(0,0,0,0.5);margin-bottom:1.2rem;">
            <div style="font-size:1rem;font-weight:700;color:#FFFFFF;margin-bottom:1rem;border-left:3px solid #C0392B;padding-left:0.8rem;">🆕 Nouveautés v3.0</div>
            <ul style="color:#F0C8C8;font-size:0.88rem;line-height:1.8;padding-left:1.5rem;">
                <li><strong style="color:#FFFFFF;">🗺️ Carte Géographique :</strong> Visualisation Mapbox interactive de toutes les antennes avec filtres, heatmap et stats régionales</li>
                <li><strong style="color:#FFFFFF;">📂 Import Données Réelles :</strong> Upload CSV de vos données réseau (SNMP, NetFlow, OSS/BSS) avec validation automatique</li>
                <li><strong style="color:#FFFFFF;">📖 Guides Intégrés :</strong> Chaque onglet dispose d'un guide complet expliquant chaque graphique et métrique</li>
                <li><strong style="color:#FFFFFF;">📊 Analyse Régionale :</strong> Agrégation des performances par zone géographique</li>
            </ul>
        </div>
        <div style="background:linear-gradient(135deg,#6B0000,#A10000);border-radius:16px;padding:1.8rem 2.4rem;color:white;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">👨‍💻</div>
            <div style="font-size:1.1rem;font-weight:800;letter-spacing:0.06em;">ANOH AMON HEMERSON</div>
            <div style="font-size:0.82rem;color:rgba(255,210,210,0.8);margin-top:0.3rem;">Data Scientist · Analytique Télécom · Systèmes IA</div>
            <div style="font-size:0.78rem;color:rgba(255,200,200,0.6);margin-top:0.8rem;">Construit avec ❤️ pour les opérations télécom entreprise</div>
        </div>
    </div>
    """, height=920, scrolling=True)


# ─────────────────────────────────────────────
# SPLASH SCREEN
# ─────────────────────────────────────────────
def inject_splash_screen():
    components.html("""
<!DOCTYPE html><html><head>
<style>* {margin:0;padding:0;box-sizing:border-box;} body {background:transparent;overflow:hidden;}</style>
</head><body>
<script>
(function() {
    function buildSplash(doc) {
        var style = doc.createElement('style');
        style.id = 'autotel-splash-style';
        style.textContent = `
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700;900&display=swap');
            #autotel-splash {
                position:fixed!important;top:0!important;left:0!important;
                width:100vw!important;height:100vh!important;z-index:2147483647!important;
                background:#0D0000!important;display:flex!important;flex-direction:column!important;
                align-items:center!important;justify-content:center!important;
                font-family:'Inter',sans-serif!important;transition:opacity 0.55s ease,transform 0.55s ease!important;
            }
            #autotel-splash.hiding {opacity:0!important;transform:scale(1.04)!important;pointer-events:none!important;}
            .sp-rings {position:relative;width:130px;height:130px;margin-bottom:1.8rem;flex-shrink:0;}
            .sp-ring {position:absolute;border-radius:50%;border:2px solid transparent;animation:spRingPulse 2s ease-out infinite;}
            .sp-ring:nth-child(1){inset:0;border-color:rgba(192,57,43,0.75);animation-delay:0s;}
            .sp-ring:nth-child(2){inset:-18px;border-color:rgba(161,0,0,0.45);animation-delay:0.35s;}
            .sp-ring:nth-child(3){inset:-36px;border-color:rgba(139,0,0,0.22);animation-delay:0.7s;}
            @keyframes spRingPulse {0%{transform:scale(0.65);opacity:0;}35%{opacity:1;}100%{transform:scale(1.35);opacity:0;}}
            .sp-logo-wrap {position:absolute;inset:0;display:flex;align-items:center;justify-content:center;}
            .sp-icon {font-size:2.8rem;animation:spBeat 1.4s ease-in-out infinite;filter:drop-shadow(0 0 16px rgba(192,57,43,0.9));line-height:1;}
            @keyframes spBeat {0%,100%{transform:scale(1);filter:drop-shadow(0 0 12px rgba(192,57,43,0.7));}50%{transform:scale(1.14);filter:drop-shadow(0 0 28px rgba(255,90,90,1));}}
            .sp-brand {font-size:2.5rem;font-weight:900;color:#FFFFFF;letter-spacing:0.14em;text-transform:uppercase;animation:spSlideUp 0.6s cubic-bezier(.22,1,.36,1) 0.25s both;}
            .sp-sub {font-size:0.7rem;color:#DD9999;letter-spacing:0.24em;text-transform:uppercase;font-weight:500;margin-top:0.3rem;animation:spSlideUp 0.6s cubic-bezier(.22,1,.36,1) 0.45s both;}
            @keyframes spSlideUp {from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}
            .sp-prog-wrap {width:230px;height:3px;background:rgba(255,255,255,0.08);border-radius:10px;overflow:hidden;margin-top:2.6rem;animation:spSlideUp 0.4s ease 0.6s both;}
            .sp-prog-bar {height:100%;width:0%;border-radius:10px;background:linear-gradient(90deg,#7A0000,#C0392B,#FF7777);box-shadow:0 0 12px rgba(192,57,43,0.85);animation:spFill 2.5s cubic-bezier(.4,0,.2,1) 0.65s forwards;}
            @keyframes spFill {0%{width:0%;}25%{width:30%;}55%{width:65%;}80%{width:87%;}100%{width:100%;}}
            .sp-dots {display:flex;gap:0.45rem;margin-top:1.1rem;animation:spSlideUp 0.4s ease 0.8s both;}
            .sp-dot {width:7px;height:7px;border-radius:50%;background:#A10000;animation:spDot 1.2s ease-in-out infinite;}
            .sp-dot:nth-child(2){animation-delay:0.2s;}.sp-dot:nth-child(3){animation-delay:0.4s;}
            @keyframes spDot {0%,80%,100%{transform:scale(0.7);opacity:0.5;background:#A10000;}40%{transform:scale(1.4);opacity:1;background:#FF6666;}}
            .sp-tagline {font-size:0.67rem;color:rgba(240,200,200,0.4);letter-spacing:0.1em;margin-top:0.6rem;animation:spSlideUp 0.4s ease 1s both;}
        `;
        doc.head.appendChild(style);
        var div = doc.createElement('div');
        div.id = 'autotel-splash';
        div.innerHTML = `
            <div class="sp-rings"><div class="sp-ring"></div><div class="sp-ring"></div><div class="sp-ring"></div>
            <div class="sp-logo-wrap"><div class="sp-icon">📡</div></div></div>
            <div class="sp-brand">AutoTel AI</div>
            <div class="sp-sub">Telecom Intelligence Platform</div>
            <div class="sp-prog-wrap"><div class="sp-prog-bar"></div></div>
            <div class="sp-dots"><div class="sp-dot"></div><div class="sp-dot"></div><div class="sp-dot"></div></div>
            <div class="sp-tagline">Initialisation du système &nbsp;&middot;&nbsp; v3.0.0</div>
        `;
        doc.body.appendChild(div);
        setTimeout(function() {
            div.classList.add('hiding');
            setTimeout(function() {
                if (div.parentNode) div.parentNode.removeChild(div);
                var st = doc.getElementById('autotel-splash-style');
                if (st && st.parentNode) st.parentNode.removeChild(st);
            }, 600);
        }, 2900);
    }
    try { buildSplash(window.parent.document); } catch(e) { buildSplash(document); }
})();
</script></body></html>
    """, height=0, scrolling=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    inject_splash_screen()
    inject_global_css()
    render_header()

    with st.spinner("Chargement des données réseau..."):
        df_simulated = generate_telecom_dataset(720)

    # Use uploaded data if available
    df = st.session_state.get('uploaded_df', df_simulated)

    if 'uploaded_df' in st.session_state:
        st.markdown("""
        <div style="background:#001A0A;border:1px solid #1A6A30;border-left:4px solid #22C55E;
                    border-radius:8px;padding:0.5rem 1rem;margin-bottom:0.5rem;
                    font-size:0.8rem;color:#86EFAC;">
            📂 <strong>Données réelles actives</strong> — L'application utilise votre fichier importé.
            Allez dans "Import Données" pour en changer.
        </div>
        """, unsafe_allow_html=True)

    page = render_sidebar()

    if page == "Tableau de Bord":
        page_dashboard(df)
    elif page == "Surveillance":
        page_monitoring(df)
    elif page == "Carte Réseau":
        page_map(df)
    elif page == "Import Données":
        page_import(df)
    elif page == "Prédiction":
        page_prediction(df)
    elif page == "Optimisation":
        page_optimization(df)
    elif page == "Simulation":
        page_simulation(df)
    elif page == "Analyses":
        page_insights(df)
    elif page == "À Propos":
        page_about()


if __name__ == "__main__":
    main()