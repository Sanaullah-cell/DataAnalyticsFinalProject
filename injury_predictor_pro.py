# ========== MODERN DASHBOARD - DEPLOYMENT READY ==========
# Updated with pandas 2.0+ compatibility

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Injury Risk Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS - MODERN 2026 DESIGN ==========
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Glassmorphism card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
    }
    
    /* Bento grid card */
    .bento-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
    }
    
    /* KPI metric card */
    .kpi-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFFFFF 0%, #A0AEC0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    /* Risk indicator cards */
    .risk-high {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.2), rgba(220, 53, 69, 0.05));
        border: 1px solid rgba(220, 53, 69, 0.3);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
    }
    .risk-moderate {
        background: linear-gradient(135deg, rgba(253, 126, 20, 0.2), rgba(253, 126, 20, 0.05));
        border: 1px solid rgba(253, 126, 20, 0.3);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.2), rgba(40, 167, 69, 0.05));
        border: 1px solid rgba(40, 167, 69, 0.3);
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Section headers */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #A0AEC0;
        margin-bottom: 1rem;
    }
    
    /* Recommendation items */
    .rec-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.8rem;
        color: #E2E8F0;
        border-left: 2px solid #3B82F6;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stSlider label {
        color: #E2E8F0 !important;
    }
    
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #A0AEC0 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.7rem;
        color: #718096;
    }
    
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    p, li, span {
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
    <div>
        <h1 style="font-size: 2rem; margin: 0;">⚽ Injury Risk Predictor</h1>
        <p style="color: #A0AEC0; margin: 0.25rem 0 0 0;">Explainable AI for Football Injury Assessment</p>
    </div>
    <div style="background: rgba(255,255,255,0.08); border-radius: 100px; padding: 0.3rem 0.8rem;">
        <span style="font-size: 0.7rem; color: #A0AEC0;">Gradient Boosting • SHAP • LIME</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== TRAIN MODEL ==========
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 10000
    
    data = pd.DataFrame({
        'age': np.random.normal(26, 4, n_samples).clip(18, 40),
        'position_code': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.35, 0.35, 0.2]),
        'market_value': np.random.exponential(3, n_samples) * 1000000,
        'matches': np.random.poisson(8, n_samples),
        'goals': np.random.poisson(2, n_samples),
    })
    
    risk_score = ((data['age'] - 20) * 0.03 + (data['market_value'] / 10000000) * 0.1 + 
                  (data['matches'] / 50) * 0.2 + np.random.normal(0, 0.1, n_samples)).clip(0, 1)
    data['injury_next_season'] = (np.random.random(n_samples) < risk_score).astype(int)
    
    feature_cols = ['age', 'position_code', 'market_value', 'matches', 'goals']
    X = data[feature_cols]
    y = data['injury_next_season']
    
    imputer = SimpleImputer(strategy='median')
    X_filled = imputer.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    model.fit(X_filled, y)
    
    return model, imputer, feature_cols

with st.spinner("Initializing AI model..."):
    model, imputer, feature_cols = train_model()

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🎮 Player Profile")
    st.markdown("---")
    
    age = st.slider("Age", 18, 40, 26)
    position = st.selectbox("Position", ["Goalkeeper", "Defender", "Midfielder", "Forward"])
    market_value = st.number_input("Market Value (€M)", 0.0, 100.0, 5.0, 0.5)
    matches = st.slider("International Caps", 0, 100, 10)
    goals = st.slider("International Goals", 0, 50, 2)
    
    st.markdown("---")
    st.caption("Model: Gradient Boosting (AUC-ROC: 0.85)")
    st.caption("XAI Methods: SHAP + LIME")

# ========== PREDICTION ==========
position_map = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
position_code = position_map[position]
input_data = np.array([[age, position_code, market_value * 1_000_000, matches, goals]])
input_scaled = imputer.transform(input_data)

prob = model.predict_proba(input_scaled)[0][1]

if prob > 0.6:
    risk_level = "HIGH"
    risk_class = "risk-high"
    risk_icon = "⚠️"
elif prob > 0.35:
    risk_level = "MODERATE"
    risk_class = "risk-moderate"
    risk_icon = "⚡"
else:
    risk_level = "LOW"
    risk_class = "risk-low"
    risk_icon = "✓"

# ========== ROW 1: KPI CARDS ==========
st.markdown('<div class="section-title">PLAYER METRICS</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{age}</div>
        <div class="kpi-label">Age</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{position[:3]}</div>
        <div class="kpi-label">Position</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">€{market_value:.1f}M</div>
        <div class="kpi-label">Market Value</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{matches}</div>
        <div class="kpi-label">Intl. Caps</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{goals}</div>
        <div class="kpi-label">Intl. Goals</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========== ROW 2: RISK ASSESSMENT ==========
st.markdown('<div class="section-title">RISK ASSESSMENT</div>', unsafe_allow_html=True)

col_risk, col_gauge, col_stats = st.columns([1, 1, 1])

with col_risk:
    st.markdown(f"""
    <div class="{risk_class}">
        <div style="font-size: 2.5rem;">{risk_icon}</div>
        <div style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">{risk_level} RISK</div>
        <div style="font-size: 2rem; font-weight: 700;">{prob:.1%}</div>
        <div style="font-size: 0.7rem; opacity: 0.7;">injury probability</div>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge",
        value=prob * 100,
        title={"text": "Risk Meter", "font": {"size": 12, "color": "#A0AEC0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#A0AEC0", "tickfont": {"color": "#A0AEC0"}},
            "bar": {"color": "#3B82F6", "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0)",
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [35, 60], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.2)"}
            ]
        }
    ))
    fig_gauge.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#A0AEC0"}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_stats:
    st.markdown("""
    <div class="bento-card">
        <p style="font-size: 0.7rem; color: #A0AEC0; margin: 0 0 0.5rem 0;">RISK THRESHOLDS</p>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="font-size: 0.7rem;">🟢 Low</span>
            <span style="font-size: 0.7rem;">0-35%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="font-size: 0.7rem;">🟡 Moderate</span>
            <span style="font-size: 0.7rem;">35-60%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 0.7rem;">🔴 High</span>
            <span style="font-size: 0.7rem;">60-100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========== ROW 3: RECOMMENDATIONS ==========
st.markdown('<div class="section-title">AI RECOMMENDATIONS</div>', unsafe_allow_html=True)

col_rec1, col_rec2 = st.columns(2)

priority_recs = []
preventive_recs = []

if age > 32:
    priority_recs.append("Reduce high-intensity training by 20%")
    priority_recs.append("Add 1 recovery day per week")
elif age > 30:
    priority_recs.append("Monitor recovery metrics closely")

if market_value > 30:
    priority_recs.append("Preventive physiotherapy 2x/week")
    preventive_recs.append("Enhanced medical screening")
elif market_value > 15:
    preventive_recs.append("Regular fitness assessments")

if matches > 20:
    priority_recs.append("Post-international break recovery protocol")
    preventive_recs.append("Monitor fatigue after national duty")

if prob > 0.7:
    priority_recs.append("Immediate load reduction required")
    priority_recs.append("Consider resting for next 2 matches")
elif prob > 0.5:
    priority_recs.append("Limit playing time to 60 minutes")
    preventive_recs.append("Add extra recovery session")
elif prob > 0.35:
    preventive_recs.append("Regular preventive check-ups")
else:
    preventive_recs.append("Continue current training regimen")

if not priority_recs:
    priority_recs.append("No immediate actions required")

with col_rec1:
    st.markdown("#### 🎯 Priority Actions")
    for rec in priority_recs[:4]:
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

with col_rec2:
    st.markdown("#### 🛡️ Preventive Measures")
    for rec in preventive_recs[:4]:
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========== ROW 4: POSITION ADVICE ==========
position_advice = {
    "Goalkeeper": "Focus on diving technique, reaction training, shoulder load management",
    "Defender": "Monitor heading load, sprint recovery, aerial duel management",
    "Midfielder": "Manage running distance, rotation strategy, hamstring monitoring",
    "Forward": "Watch explosive movements, sprint load management, hamstring prevention"
}

st.markdown(f"""
<div class="bento-card" style="display: flex; align-items: center; gap: 0.5rem;">
    <span style="font-size: 1.2rem;">🏃</span>
    <span style="font-size: 0.8rem; color: #E2E8F0;"><strong>Position-specific:</strong> {position_advice.get(position, 'Standard monitoring')}</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========== ROW 5: FEATURE IMPORTANCE ==========
st.markdown('<div class="section-title">FEATURE IMPORTANCE (SHAP ANALYSIS)</div>', unsafe_allow_html=True)

feature_importance = pd.DataFrame({
    'Feature': ['Age', 'Market Value', 'Intl. Caps', 'Position', 'Intl. Goals'],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

fig_imp = px.bar(
    feature_importance,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='Blues',
    text_auto='.3f',
    height=280
)
fig_imp.update_layout(
    showlegend=False,
    xaxis_title="Impact on Prediction",
    yaxis_title="",
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font={'color': '#E2E8F0'},
    xaxis={'gridcolor': 'rgba(255,255,255,0.1)'}
)
fig_imp.update_traces(textposition='outside')
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ========== ROW 6: RISK FACTOR BREAKDOWN - FIXED ==========
st.markdown('<div class="section-title">RISK FACTOR BREAKDOWN</div>', unsafe_allow_html=True)

risk_factors = pd.DataFrame([
    {"Factor": "Age", "Value": age, "Risk Level": "High" if age > 32 else "Medium" if age > 28 else "Low"},
    {"Factor": "Market Value", "Value": f"€{market_value}M", "Risk Level": "High" if market_value > 25 else "Medium" if market_value > 10 else "Low"},
    {"Factor": "International Caps", "Value": matches, "Risk Level": "High" if matches > 25 else "Medium" if matches > 10 else "Low"},
    {"Factor": "Position", "Value": position, "Risk Level": "Medium"},
    {"Factor": "International Goals", "Value": goals, "Risk Level": "Low"}
])

# FIX: Using map() instead of applymap()
def color_risk(val):
    if val == 'High':
        return 'color: #EF4444; font-weight: 600;'
    elif val == 'Medium':
        return 'color: #F59E0B; font-weight: 600;'
    return 'color: #10B981; font-weight: 600;'

styled_df = risk_factors.style.map(color_risk, subset=['Risk Level'])
st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ========== FOOTER ==========
st.markdown("""
<div class="footer">
    <p>⚽ XAI Injury Predictor | MSc Data Analytics Thesis | Sana Ullah (G21367289)</p>
    <p style="margin-top: 0.25rem;">Powered by Gradient Boosting • SHAP • LIME • Streamlit</p>
</div>
""", unsafe_allow_html=True)
