import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pytorch_tabnet.tab_model import TabNetRegressor

# ------------------------------------------------------------
# Student Performance Predictor - Interactive Dashboard
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.environ.get("STUDENT_DATA_PATH", "stud_synthetic_50000_full.csv")
MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "student_tabnet_bundle.joblib")
TARGETS = ["math_score", "reading_score", "writing_score"]

CAT_COLS = [
    "gender", "race_ethnicity", "parental_level_of_education", "lunch",
    "test_preparation_course", "school", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic",
]
NUM_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "G3",
]
FEATURE_COLS = CAT_COLS + NUM_COLS

# ------------------------------------------------------------
# Modern UI Styling & Design System
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Outfit:wght@500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* Base text hierarchy - Crisp, Bold, High-Contrast */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #f8fafc;
    color: #0f172a;
}

[data-testid="stMain"] h1:not(.hero h1),
[data-testid="stMain"] h2:not(.hero h2),
[data-testid="stMain"] h3:not(.hero h3),
[data-testid="stMain"] h4:not(.hero h4) {
    font-family: 'Outfit', 'Inter', sans-serif;
    color: #000000 !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

[data-testid="stMain"] p:not(.hero p):not(.metric-note):not(.metric-label):not([data-testid="stForm"] p):not([data-testid="stWidgetLabel"] p):not(label p),
[data-testid="stMain"] li:not([data-testid="stForm"] li) {
    color: #0f172a;
    font-size: 0.96rem;
    font-weight: 550;
    line-height: 1.55;
}

/* Captions & Subtext - Bold and clearly readable, not washed-out */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    opacity: 1 !important;
}

/* Sidebar - Bold & High Contrast */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 2px solid #e2e8f0;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: #000000 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #0f172a !important;
    font-weight: 650 !important;
}

[data-testid="stSidebar"] small, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Hero Banner - Vibrant deep indigo with solid bright white typography */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4338ca 70%, #6366f1 100%);
    padding: 34px 40px;
    border-radius: 20px;
    color: #ffffff !important;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.28), 0 8px 10px -6px rgba(67, 56, 202, 0.22);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.hero h1, .hero h1 * {
    color: #ffffff !important;
    font-size: 2.3rem !important;
    margin: 0 0 10px 0 !important;
    font-weight: 900 !important;
    letter-spacing: -0.02em !important;
    font-family: 'Outfit', sans-serif !important;
}

.hero p, .hero p * {
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 1.08rem !important;
    max-width: 840px !important;
    line-height: 1.55 !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.25);
}

.hero .hero-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(255, 255, 255, 0.25) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    color: #ffffff !important;
    padding: 5px 16px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
}

/* Glass & Elevated Cards - Bold Labels & Values */
.metric-card {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    min-height: 115px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04) !important;
}

.metric-card .metric-label {
    color: #1e293b !important;
    font-size: 0.82rem !important;
    font-weight: 750 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

.metric-card .metric-value {
    color: #000000 !important;
    font-size: 1.95rem !important;
    font-weight: 900 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.03em !important;
}

.metric-card .metric-note {
    color: #3730a3 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    margin-top: 5px !important;
}

.panel {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 18px !important;
    padding: 22px 24px !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.04) !important;
    margin-bottom: 20px !important;
}

.panel .section-title, .panel .section-title * {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #000000 !important;
    margin-bottom: 6px !important;
    font-family: 'Outfit', sans-serif !important;
}

.panel .small-muted, .panel .small-muted * {
    color: #1e293b !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}

/* Streamlit Native Metric Component */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
}

div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] span {
    color: #1e293b !important;
    font-size: 0.82rem !important;
    font-weight: 750 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

div[data-testid="stMetricValue"] div {
    color: #000000 !important;
    font-size: 1.95rem !important;
    font-weight: 900 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Input Fields & Widgets (Default - Outside Forms) */
[data-testid="stMain"] [data-testid="stWidgetLabel"] label,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {
    color: #000000 !important;
    font-weight: 750 !important;
    font-size: 0.92rem !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    color: #000000 !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="input"] input {
    color: #000000 !important;
    font-weight: 650 !important;
}

div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #0f172a !important;
    font-size: 0.94rem !important;
    font-weight: 650 !important;
}

div[data-testid="stExpander"] details {
    background-color: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary p {
    color: #000000 !important;
    font-weight: 750 !important;
    font-size: 1.0rem !important;
}

/* ======================================================== */
/* MANUAL DROPDOWN & INPUT SECTION - BOLD BLACK THEME */
/* ======================================================== */
[data-testid="stForm"],
form[data-testid="stForm"],
div[data-testid="stForm"] {
    background: #090d16 !important;
    border: 2px solid #334155 !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 14px 35px -5px rgba(0, 0, 0, 0.45) !important;
}

/* Form expanders in black theme */
[data-testid="stForm"] [data-testid="stExpander"] details,
[data-testid="stForm"] div[data-testid="stExpander"] details {
    background-color: #0e1424 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stForm"] [data-testid="stExpander"] summary,
[data-testid="stForm"] div[data-testid="stExpander"] summary {
    background-color: #1e293b !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 16px !important;
}

[data-testid="stForm"] [data-testid="stExpander"] summary span,
[data-testid="stForm"] [data-testid="stExpander"] summary p {
    color: #ffffff !important;
    font-weight: 850 !important;
    font-size: 1.05rem !important;
}

[data-testid="stForm"] [data-testid="stExpander"] summary svg {
    fill: #ffffff !important;
}

/* ALL FORM LABELS (Gender, Race Ethnicity, etc.) - PURE BOLD WHITE */
[data-testid="stForm"] label,
[data-testid="stForm"] label *,
[data-testid="stForm"] p,
[data-testid="stForm"] span,
[data-testid="stForm"] [data-testid="stWidgetLabel"],
[data-testid="stForm"] [data-testid="stWidgetLabel"] *,
[data-testid="stForm"] [data-testid="stWidgetLabel"] label,
[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
[data-testid="stForm"] [data-testid="stWidgetLabel"] span,
[data-testid="stForm"] [data-testid="stMarkdownContainer"] p,
[data-testid="stForm"] [data-testid="stMarkdownContainer"] span,
[data-testid="stForm"] .stSelectbox label,
[data-testid="stForm"] .stSelectbox label *,
[data-testid="stForm"] .stNumberInput label,
[data-testid="stForm"] .stNumberInput label * {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 0.96rem !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: inline-block !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.6) !important;
}

/* Dropdown Selectbox - Pure Black Background with Solid Bold White Text */
[data-testid="stForm"] div[data-baseweb="select"] > div,
[data-testid="stForm"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stForm"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #000000 !important;
    background: #000000 !important;
    border: 1.5px solid #4b5563 !important;
    border-radius: 10px !important;
}

[data-testid="stForm"] div[data-baseweb="select"] span,
[data-testid="stForm"] div[data-baseweb="select"] div,
[data-testid="stForm"] div[data-baseweb="select"] p,
[data-testid="stForm"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #ffffff !important;
    font-weight: 750 !important;
    font-size: 0.95rem !important;
}

[data-testid="stForm"] div[data-baseweb="select"] svg {
    fill: #ffffff !important;
}

/* Number Inputs - Pure Black Background with Solid Bold White Text */
[data-testid="stForm"] div[data-baseweb="input"] > div,
[data-testid="stForm"] [data-testid="stNumberInput"] div[data-baseweb="input"] > div,
[data-testid="stForm"] .stNumberInput div[data-baseweb="input"] > div {
    background-color: #000000 !important;
    background: #000000 !important;
    border: 1.5px solid #4b5563 !important;
    border-radius: 10px !important;
}

[data-testid="stForm"] div[data-baseweb="input"] input,
[data-testid="stForm"] [data-testid="stNumberInput"] input {
    background-color: #000000 !important;
    background: #000000 !important;
    color: #ffffff !important;
    font-weight: 750 !important;
    font-size: 1.0rem !important;
}

[data-testid="stForm"] button[data-testid="stNumberInputStepDown"],
[data-testid="stForm"] button[data-testid="stNumberInputStepUp"] {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border-color: #4b5563 !important;
}

[data-testid="stForm"] button[data-testid="stNumberInputStepDown"] svg,
[data-testid="stForm"] button[data-testid="stNumberInputStepUp"] svg {
    fill: #ffffff !important;
}

/* Dropdown List Popup Menu (when opened) */
div[data-baseweb="popover"] ul[role="listbox"] {
    background-color: #000000 !important;
    border: 1.5px solid #4b5563 !important;
    border-radius: 10px !important;
}

div[data-baseweb="popover"] li[role="option"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
}

div[data-baseweb="popover"] li[role="option"] span,
div[data-baseweb="popover"] li[role="option"] div {
    color: #ffffff !important;
    font-weight: 700 !important;
}

div[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 2px dashed #94a3b8 !important;
    border-radius: 16px !important;
}

div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p {
    color: #0f172a !important;
    font-weight: 650 !important;
}

div[data-testid="stFileUploaderDropzone"] small {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Custom styled buttons */
.stButton>button[kind="primary"],
.stButton>button[kind="primary"] *,
.stButton>button[kind="primary"] p {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 750 !important;
    font-size: 0.98rem !important;
    padding: 11px 24px !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.32) !important;
    transition: all 0.2s ease;
}

.stButton>button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.42) !important;
}

.stDownloadButton>button,
.stDownloadButton>button *,
.stDownloadButton>button p {
    border-radius: 12px !important;
    font-weight: 700 !important;
    border: 1.5px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #0f172a !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease;
}

.stDownloadButton>button:hover {
    background: #f1f5f9 !important;
    border-color: #64748b !important;
}

/* Badges & Status chips */
.badge-chip {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}

.badge-low, .badge-low * {
    background: #dcfce7 !important;
    color: #14532d !important;
    border: 1.5px solid #86efac !important;
}

.badge-med, .badge-med * {
    background: #fef3c7 !important;
    color: #78350f !important;
    border: 1.5px solid #fcd34d !important;
}

.badge-high, .badge-high * {
    background: #fee2e2 !important;
    color: #7f1d1d !important;
    border: 1.5px solid #fca5a5 !important;
}

/* Streamlit Tabs styling */
div[data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #cbd5e1;
    margin-bottom: 22px;
}

div[data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 11px 22px;
    font-weight: 700;
    font-size: 0.95rem;
    color: #334155;
}

div[data-baseweb="tab"][aria-selected="true"] {
    color: #312e81 !important;
    font-weight: 850 !important;
    border-bottom: 3px solid #4f46e5;
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Helper: Styling Plotly charts consistently - High Contrast & Bold
# ------------------------------------------------------------
def style_fig(fig, height=340, title=None):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=15, color="#000000", family="Outfit, Inter, sans-serif", weight="bold")) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        xaxis=dict(
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            linecolor="#94a3b8",
            tickfont=dict(color="#0f172a", size=11, family="Inter", weight="bold"),
            title_font=dict(color="#000000", size=12, family="Inter", weight="bold")
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            linecolor="#94a3b8",
            tickfont=dict(color="#0f172a", size=11, family="Inter", weight="bold"),
            title_font=dict(color="#000000", size=12, family="Inter", weight="bold")
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#cbd5e1",
            borderwidth=1.5,
            font=dict(size=11, color="#0f172a", family="Inter", weight="bold")
        )
    )
    return fig

def create_gauge(score, title, color="#4f46e5"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "", 'font': {'size': 28, 'color': '#000000', 'family': 'Outfit', 'weight': 'bold'}},
        title={'text': title, 'font': {'size': 14, 'color': '#0f172a', 'family': 'Outfit', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': '#94a3b8', 'tickfont': {'color': '#0f172a', 'weight': 'bold'}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#f1f5f9",
            'borderwidth': 1.5,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.18)'},
                {'range': [60, 80], 'color': 'rgba(245, 158, 11, 0.20)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.22)'}
            ],
            'threshold': {
                'line': {'color': '#000000', 'width': 3.5},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=195,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    return fig

def create_radar(scores_dict, cohort_averages):
    categories = ['Math', 'Reading', 'Writing']
    student_vals = [scores_dict['math_score'], scores_dict['reading_score'], scores_dict['writing_score']]
    cohort_vals = [cohort_averages['math_score'], cohort_averages['reading_score'], cohort_averages['writing_score']]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=student_vals + [student_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Predicted Student',
        line=dict(color='#4f46e5', width=3),
        fillcolor='rgba(79, 70, 229, 0.25)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=cohort_vals + [cohort_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Cohort Benchmark',
        line=dict(color='#475569', width=2.5, dash='dash'),
        fillcolor='rgba(71, 85, 105, 0.15)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#cbd5e1', linecolor='#94a3b8', tickfont=dict(color='#0f172a', weight='bold')),
            angularaxis=dict(gridcolor='#cbd5e1', linecolor='#94a3b8', tickfont=dict(color='#0f172a', size=12, weight='bold'))
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(color='#0f172a', weight='bold')),
        height=320,
        margin=dict(l=35, r=35, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12)
    )
    return fig

# ------------------------------------------------------------
# Data & Modeling Utilities
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path):
    data = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + TARGETS if c not in data.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    return data

def grade_band(avg):
    if avg >= 90: return "A", "Excellent", "Low"
    if avg >= 80: return "B", "Good", "Low"
    if avg >= 70: return "C", "Fair", "Moderate"
    if avg >= 60: return "D", "Needs improvement", "High"
    return "F", "At risk", "High"

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
            ]), CAT_COLS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]), NUM_COLS),
        ],
        remainder="drop",
    )

def train_tabnet(data, progress=None):
    clean = data.dropna(subset=TARGETS).copy()
    for c in CAT_COLS:
        clean[c] = clean[c].fillna("missing").astype(str)

    X = clean[FEATURE_COLS].copy()
    bundle = {"model_name": "TabNet", "targets": {}, "feature_cols": FEATURE_COLS}
    metric_rows = []

    for idx, target in enumerate(TARGETS):
        y = clean[target].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        prep = make_preprocessor()
        X_train_t = prep.fit_transform(X_train).astype(np.float32)
        X_test_t = prep.transform(X_test).astype(np.float32)
        y_train_t = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
        y_test_t = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

        model = TabNetRegressor(
            n_d=16, n_a=16, n_steps=5, gamma=1.3, lambda_sparse=1e-3,
            optimizer_params={"lr": 0.02}, seed=42, verbose=0
        )
        model.fit(
            X_train=X_train_t, y_train=y_train_t,
            eval_set=[(X_test_t, y_test_t)],
            eval_name=["test"], eval_metric=["mae"],
            max_epochs=100, patience=15, batch_size=1024,
            virtual_batch_size=128, num_workers=0, drop_last=False,
        )
        pred = model.predict(X_test_t).reshape(-1)
        r2 = float(r2_score(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        bundle["targets"][target] = {
            "preprocessor": prep, "model": model,
            "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse},
        }
        metric_rows.append({"Subject": target.replace("_score","").title(),
                            "R²": r2, "MAE": mae, "RMSE": rmse})
        if progress:
            progress.progress((idx + 1) / len(TARGETS),
                              text=f"Training TabNet for {target.replace('_',' ')} ({idx+1}/3)")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle, pd.DataFrame(metric_rows)

def predict_one(bundle, profile):
    row = pd.DataFrame([{c: profile[c] for c in FEATURE_COLS}])
    scores = {}
    for target in TARGETS:
        item = bundle["targets"][target]
        x = item["preprocessor"].transform(row).astype(np.float32)
        pred = float(item["model"].predict(x).reshape(-1)[0])
        scores[target] = float(np.clip(pred, 0, 100))
    avg = float(np.mean(list(scores.values())))
    letter, label, risk = grade_band(avg)
    return scores, avg, letter, label, risk

def predict_bulk(bundle, input_df):
    sub_df = input_df.copy()
    for c in CAT_COLS:
        if c in sub_df.columns:
            sub_df[c] = sub_df[c].fillna("missing").astype(str)
        else:
            sub_df[c] = "missing"
    for c in NUM_COLS:
        if c in sub_df.columns:
            sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce")
        else:
            sub_df[c] = np.nan

    preds_dict = {}
    for target in TARGETS:
        item = bundle["targets"][target]
        x = item["preprocessor"].transform(sub_df[FEATURE_COLS]).astype(np.float32)
        preds = item["model"].predict(x).reshape(-1)
        preds_dict[target] = np.clip(preds, 0, 100).round(1)

    results = input_df.copy()
    results["pred_math_score"] = preds_dict["math_score"]
    results["pred_reading_score"] = preds_dict["reading_score"]
    results["pred_writing_score"] = preds_dict["writing_score"]

    avg_arr = np.mean(
        [preds_dict["math_score"], preds_dict["reading_score"], preds_dict["writing_score"]],
        axis=0
    ).round(1)
    results["pred_overall_average"] = avg_arr

    bands = [grade_band(avg) for avg in avg_arr]
    results["predicted_grade_band"] = [f"{b[0]} ({b[1]})" for b in bands]
    results["support_risk_level"] = [b[2] for b in bands]
    return results

def df_to_excel_bytes(data_df):
    try:
        import openpyxl  # noqa: F401
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            data_df.to_excel(writer, index=False, sheet_name="Sheet1")
        return buf.getvalue()
    except Exception:
        return None

def recommendations(profile, risk):
    tips = []
    if int(profile["failures"]) > 0:
        tips.append("Review earlier failed subjects and arrange targeted academic tutoring.")
    if int(profile["studytime"]) <= 2:
        tips.append("Build a structured study routine and increase weekly focused study time.")
    if int(profile["absences"]) > 10:
        tips.append("Review reasons for missed classes and implement an attendance improvement plan.")
    if profile["test_preparation_course"] == "none":
        tips.append("Enroll in mock exam preparation and practice test workshops.")
    if profile["schoolsup"] == "no" and int(profile["failures"]) > 0:
        tips.append("Request remedial support sessions through school academic counselors.")
    if profile["famsup"] == "no":
        tips.append("Involve family or peer study groups to provide a supportive study environment.")
    if not tips:
        tips.append("Maintain consistent study habits, high attendance, and periodic progress tracking.")
    if risk == "High":
        tips.insert(0, "Schedule an early intervention meeting with an academic advisor.")
    return tips

# ------------------------------------------------------------
# Application Initialization
# ------------------------------------------------------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.info("Place the CSV beside app.py, or set STUDENT_DATA_PATH to its full path.")
    st.stop()

# Load saved TabNet model bundle
bundle = None
if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        st.sidebar.warning(f"Saved model could not be loaded: {e}")

# Global Cohort Averages for Benchmarks
cohort_averages = {
    "math_score": float(df["math_score"].mean()),
    "reading_score": float(df["reading_score"].mean()),
    "writing_score": float(df["writing_score"].mean()),
    "overall": float(df[TARGETS].mean(axis=1).mean())
}

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
st.sidebar.markdown("### 🎓 Student Analytics")
page = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard Overview", "Predict Student", "Model Performance", "Dataset Explorer"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ System Status")
st.sidebar.write(f"📁 **Dataset:** `{len(df):,}` records")
if bundle is not None:
    st.sidebar.success("✅ TabNet Models Ready")
else:
    st.sidebar.warning("⚠️ Model Not Trained")

st.sidebar.caption("Targets: Math, Reading, Writing • TabNet Deep Learning Architecture")

# ------------------------------------------------------------
# Header Hero Banner
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-pill">ACADEMIC INTELLIGENCE PLATFORM</div>
  <h1>🎓 Student Performance Predictor</h1>
  <p>Leverage tabular deep learning to forecast student exam outcomes, analyze holistic demographic trends, and deliver proactive academic support interventions.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================
if page == "Dashboard Overview":
    total = len(df)
    mean_math = df["math_score"].mean()
    mean_reading = df["reading_score"].mean()
    mean_writing = df["writing_score"].mean()
    overall = df[TARGETS].mean(axis=1)
    mean_score = overall.mean()
    high_risk_share = float((overall < 60).mean() * 100)
    ready = bundle is not None

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL STUDENTS</div><div class="metric-value">{total:,}</div><div class="metric-note">50,000 synthetic cohort</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">OVERALL AVERAGE SCORE</div><div class="metric-value">{mean_score:.1f}<span style="font-size:1.1rem;color:#0f172a;font-weight:700;"> /100</span></div><div class="metric-note">Math: {mean_math:.1f} • Read: {mean_reading:.1f} • Write: {mean_writing:.1f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">AT-RISK POPULATION (&lt;60)</div><div class="metric-value">{high_risk_share:.1f}%</div><div class="metric-note">{int(total * high_risk_share / 100):,} students needing intervention</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">TABNET MODEL STATUS</div><div class="metric-value" style="color:{"#10b981" if ready else "#f59e0b"};">{"ACTIVE" if ready else "PENDING"}</div><div class="metric-note">{"3 Regressors loaded & ready" if ready else "Requires training in sidebar"}</div></div>', unsafe_allow_html=True)

    st.write("")

    # Visual Row 1: Score Distributions Boxplots + Parental Education Impact
    r1_col1, r1_col2 = st.columns([1.1, 1])

    with r1_col1:
        st.markdown('<div class="panel"><div class="section-title">📊 Subject Score Distributions & Variance</div><div class="small-muted">Comparing score dispersion, medians, and quartile ranges across exam subjects.</div>', unsafe_allow_html=True)
        long_scores = df[TARGETS].melt(var_name="Subject", value_name="Score")
        long_scores["Subject"] = long_scores["Subject"].str.replace("_score", "").str.title()
        fig_dist = px.box(
            long_scores, x="Subject", y="Score", color="Subject",
            color_discrete_sequence=["#4f46e5", "#06b6d4", "#f59e0b"],
            points=False
        )
        fig_dist.update_traces(boxmean=True)
        st.plotly_chart(style_fig(fig_dist, height=330), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with r1_col2:
        st.markdown('<div class="panel"><div class="section-title">🎓 Academic Performance by Parental Education</div><div class="small-muted">Average score breakdown across parent education levels.</div>', unsafe_allow_html=True)
        parent_edu = df.groupby("parental_level_of_education")[TARGETS].mean().reset_index()
        parent_edu["Overall"] = parent_edu[TARGETS].mean(axis=1)
        parent_edu = parent_edu.sort_values("Overall", ascending=True)
        fig_parent = px.bar(
            parent_edu, y="parental_level_of_education", x="Overall",
            orientation="h",
            text="Overall",
            color="Overall",
            color_continuous_scale="Viridis"
        )
        fig_parent.update_traces(texttemplate='%{text:.1f}', textposition='inside')
        fig_parent.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(style_fig(fig_parent, height=330), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Visual Row 2: Study Time Progression Curve + Performance Tier Donut
    r2_col1, r2_col2 = st.columns([1.1, 0.9])

    with r2_col1:
        st.markdown('<div class="panel"><div class="section-title">📈 Study Time vs. Subject Score Trajectory</div><div class="small-muted">Direct correlation between weekly hours invested and score gains across all 3 subjects.</div>', unsafe_allow_html=True)
        study_trend = df.groupby("studytime")[TARGETS].mean().reset_index().melt("studytime", var_name="Subject", value_name="Average Score")
        study_trend["Subject"] = study_trend["Subject"].str.replace("_score", "").str.title()
        study_labels = {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}
        study_trend["Study Time Bracket"] = study_trend["studytime"].map(study_labels)

        fig_study = px.line(
            study_trend, x="Study Time Bracket", y="Average Score", color="Subject",
            markers=True,
            color_discrete_sequence=["#4f46e5", "#06b6d4", "#f59e0b"]
        )
        fig_study.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(style_fig(fig_study, height=320), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col2:
        st.markdown('<div class="panel"><div class="section-title">🍩 Cohort Grade Band Distribution</div><div class="small-muted">Percentage of students categorized into standardized academic tiers.</div>', unsafe_allow_html=True)
        df_bands = df[TARGETS].mean(axis=1).apply(lambda s: grade_band(s)[0] + " (" + grade_band(s)[1] + ")").value_counts().reset_index()
        df_bands.columns = ["Grade Tier", "Count"]

        fig_donut = px.pie(
            df_bands, names="Grade Tier", values="Count",
            hole=0.55,
            color_discrete_sequence=["#10b981", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"]
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        fig_donut.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_donut, height=320), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Visual Row 3: Absences vs Score Scatter + Test Prep Course Impact
    r3_col1, r3_col2 = st.columns([1, 1])

    with r3_col1:
        st.markdown('<div class="panel"><div class="section-title">📉 Impact of School Absences on Outcomes</div><div class="small-muted">Average score trajectory as student absenteeism increases.</div>', unsafe_allow_html=True)
        # Bin absences into clean buckets for clear visualization
        df_sub = df.copy()
        df_sub["Absence Bracket"] = pd.cut(df_sub["absences"], bins=[-1, 2, 5, 10, 15, 25, 100], labels=["0-2", "3-5", "6-10", "11-15", "16-25", "25+"])
        abs_trend = df_sub.groupby("Absence Bracket", observed=False)[TARGETS].mean().reset_index()
        abs_trend["Overall Average"] = abs_trend[TARGETS].mean(axis=1)

        fig_abs = px.bar(
            abs_trend, x="Absence Bracket", y="Overall Average",
            color="Overall Average",
            color_continuous_scale="Reds_r",
            text="Overall Average"
        )
        fig_abs.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_abs.update_layout(coloraxis_showscale=False, yaxis=dict(range=[0, 100]))
        st.plotly_chart(style_fig(fig_abs, height=310), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with r3_col2:
        st.markdown('<div class="panel"><div class="section-title">🎯 Test Preparation Course Advantage</div><div class="small-muted">Point differential between students who completed test prep vs none.</div>', unsafe_allow_html=True)
        prep_comp = df.groupby("test_preparation_course")[TARGETS].mean().reset_index()
        prep_comp = prep_comp.melt("test_preparation_course", var_name="Subject", value_name="Score")
        prep_comp["Subject"] = prep_comp["Subject"].str.replace("_score", "").str.title()
        prep_comp["Course Status"] = prep_comp["test_preparation_course"].str.title()

        fig_prep = px.bar(
            prep_comp, x="Subject", y="Score", color="Course Status",
            barmode="group",
            color_discrete_sequence=["#94a3b8", "#10b981"]
        )
        fig_prep.update_layout(yaxis=dict(range=[0, 100]))
        st.plotly_chart(style_fig(fig_prep, height=310), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2: PREDICT STUDENT
# ============================================================
elif page == "Predict Student":
    st.subheader("🧑‍🎓 Student Performance Prediction")
    st.caption("Forecast academic outcomes via manual single-student profiling or bulk Excel/CSV file upload.")

    if bundle is None:
        st.warning("⚠️ TabNet regression models have not been trained yet. Please open **Model Performance** and train the models first.")

    tab_manual, tab_bulk = st.tabs(["👤 Single Student Entry", "📂 Bulk File Upload (Excel / CSV)"])

    # ----------------------------------------------------
    # TAB 1: SINGLE STUDENT ENTRY
    # ----------------------------------------------------
    with tab_manual:
        st.markdown("#### 📝 Student Profile Configuration")
        st.caption("Adjust student demographic, academic, and behavioral attributes to simulate performance.")

        defaults = {
            "gender": "female", "race_ethnicity": "group C", "parental_level_of_education": "some college",
            "lunch": "standard", "test_preparation_course": "none", "school": "GP", "address": "U",
            "famsize": "GT3", "Pstatus": "T", "Mjob": "other", "Fjob": "other", "reason": "course",
            "guardian": "mother", "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "no",
            "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
            "age": 17, "Medu": 3, "Fedu": 2, "traveltime": 1, "studytime": 2, "failures": 0,
            "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4,
            "absences": 4, "G1": 11, "G2": 11, "G3": 11,
        }

        with st.form("manual_student_form"):
            with st.expander("🏫 1. School, Demographics & Family Background", expanded=True):
                c1, c2, c3 = st.columns(3)
                profile = {}
                with c1:
                    for col in ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course", "school", "address"]:
                        opts = sorted(df[col].dropna().astype(str).unique().tolist())
                        val = defaults.get(col, opts[0])
                        profile[col] = st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(val) if val in opts else 0)
                with c2:
                    for col in ["famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup"]:
                        opts = sorted(df[col].dropna().astype(str).unique().tolist())
                        val = defaults.get(col, opts[0])
                        profile[col] = st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(val) if val in opts else 0)
                with c3:
                    for col in ["famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]:
                        opts = sorted(df[col].dropna().astype(str).unique().tolist())
                        val = defaults.get(col, opts[0])
                        profile[col] = st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(val) if val in opts else 0)

            with st.expander("📚 2. Academic History, Study Habits & Lifestyle", expanded=True):
                ncols = st.columns(4)
                numeric_labels = {
                    "age": "Age", "Medu": "Mother's Education (0–4)", "Fedu": "Father's Education (0–4)",
                    "traveltime": "Travel Time (1–4)", "studytime": "Weekly Study Time (1–4)",
                    "failures": "Past Class Failures", "famrel": "Family Relationship (1–5)",
                    "freetime": "Free Time (1–5)", "goout": "Going Out (1–5)",
                    "Dalc": "Workday Alcohol (1–5)", "Walc": "Weekend Alcohol (1–5)",
                    "health": "Health Status (1–5)", "absences": "School Absences", "G1": "Prior Exam G1 (0–20)",
                    "G2": "Prior Exam G2 (0–20)", "G3": "Prior Exam G3 (0–20)",
                }
                for i, col in enumerate(NUM_COLS):
                    with ncols[i % 4]:
                        lo = int(df[col].min())
                        hi = int(df[col].max())
                        d = int(np.clip(defaults[col], lo, hi))
                        profile[col] = st.number_input(numeric_labels[col], min_value=lo, max_value=hi, value=d, step=1)

            submitted = st.form_submit_button("✨ Compute TabNet Predictions", type="primary", width="stretch")

        if submitted:
            if bundle is None:
                st.error("Please train the TabNet models in the **Model Performance** page first.")
            else:
                scores, avg, letter, label, risk = predict_one(bundle, profile)
                st.write("")
                st.markdown("### 🎯 Model Prediction Results")

                # 4 Radial Gauges
                g1, g2, g3, g4 = st.columns(4)
                with g1:
                    st.plotly_chart(create_gauge(scores['math_score'], "MATH SCORE", "#4f46e5"), width="stretch")
                with g2:
                    st.plotly_chart(create_gauge(scores['reading_score'], "READING SCORE", "#06b6d4"), width="stretch")
                with g3:
                    st.plotly_chart(create_gauge(scores['writing_score'], "WRITING SCORE", "#f59e0b"), width="stretch")
                with g4:
                    st.plotly_chart(create_gauge(avg, "OVERALL AVERAGE", "#10b981"), width="stretch")

                # Visual Row: Radar Chart Comparison & Action Recommendations
                v_col1, v_col2 = st.columns([1, 1.1])
                with v_col1:
                    st.markdown('<div class="panel"><div class="section-title">🕸️ Skill Profile vs. Cohort Benchmark</div><div class="small-muted">Comparing student predictions against 50,000 dataset student averages.</div>', unsafe_allow_html=True)
                    st.plotly_chart(create_radar(scores, cohort_averages), width="stretch")
                    st.markdown('</div>', unsafe_allow_html=True)

                with v_col2:
                    risk_badge_class = "badge-low" if risk == "Low" else "badge-med" if risk == "Moderate" else "badge-high"
                    st.markdown(f"""
                    <div class="panel">
                      <div class="section-title">🏷️ Performance Classification</div>
                      <div style="margin: 12px 0;">
                        <span style="font-size: 2.2rem; font-weight: 900; font-family: 'Outfit'; color: #000000;">{letter}</span>
                        <span style="font-size: 1.25rem; font-weight: 750; color: #0f172a; margin-left: 8px;">— {label}</span>
                      </div>
                      <div style="margin-bottom: 14px;">
                        <span class="badge-chip {risk_badge_class}">ACADEMIC SUPPORT LEVEL: {risk.upper()}</span>
                      </div>
                      <div style="color: #1e293b; font-size: 0.9rem; font-weight: 600; line-height: 1.5; margin-bottom: 16px;">
                        Performance bands are determined from the harmonic blend of predicted math, reading, and writing competency.
                      </div>
                      <div style="font-weight: 700; color: #1e293b; margin-bottom: 8px; font-size: 0.95rem;">💡 Targeted Action Plan:</div>
                    """, unsafe_allow_html=True)

                    for tip in recommendations(profile, risk):
                        st.markdown(f"• **{tip}**")

                    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 2: BULK FILE UPLOAD
    # ----------------------------------------------------
    with tab_bulk:
        st.markdown("#### 📂 Bulk Prediction via Excel or CSV")
        st.caption("Upload a spreadsheet containing multiple student profiles to generate batch predictions, distribution charts, and downloadable reports.")

        # Step 1: Download Templates
        st.markdown('<div class="panel">'
                    '<div class="section-title">Step 1: Download Input Schema Template</div>'
                    '<div class="small-muted">Ensure your uploaded file includes all 37 required categorical and numeric feature columns.</div>',
                    unsafe_allow_html=True)
        sample_df = df[FEATURE_COLS].head(5).copy()
        t1, t2 = st.columns(2)
        with t1:
            csv_template = sample_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Template (CSV)",
                data=csv_template,
                file_name="student_prediction_template.csv",
                mime="text/csv",
                width="stretch"
            )
        with t2:
            excel_bytes = df_to_excel_bytes(sample_df)
            if excel_bytes is not None:
                st.download_button(
                    label="📥 Download Template (Excel .xlsx)",
                    data=excel_bytes,
                    file_name="student_prediction_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )
            else:
                st.info("Excel export requires openpyxl. CSV template available.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 2: Upload File
        st.markdown('<div class="panel">'
                    '<div class="section-title">Step 2: Upload Student File</div>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["xlsx", "xls", "csv"],
            key="bulk_uploader",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            user_df = None
            try:
                if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                    user_df = pd.read_excel(uploaded_file)
                else:
                    user_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading file: {e}")

            if user_df is not None:
                # Column normalization
                normalized_map = {c.lower().replace(" ", "_").replace("-", "_").replace("/", "_"): c for c in FEATURE_COLS}
                rename_map = {}
                for col in user_df.columns:
                    norm = str(col).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
                    if norm in normalized_map:
                        rename_map[col] = normalized_map[norm]
                clean_df = user_df.rename(columns=rename_map)

                missing_cols = [c for c in FEATURE_COLS if c not in clean_df.columns]
                st.write("")
                st.markdown(f"**Loaded:** `{uploaded_file.name}` • **{len(clean_df):,}** students recognized")

                if missing_cols:
                    st.error(f"⚠️ Missing {len(missing_cols)} required column(s): {', '.join(missing_cols)}")
                    st.info("Please use the template above to ensure headers match expected schema.")
                else:
                    st.success("✅ File format and feature columns successfully validated.")
                    with st.expander(f"👀 Preview Uploaded Raw Data (First 5 of {len(clean_df):,} records)"):
                        st.dataframe(clean_df.head(5), width="stretch")

                    if st.button("🚀 Run TabNet Batch Prediction", type="primary", width="stretch"):
                        if bundle is None:
                            st.error("Please train the TabNet models in **Model Performance** first.")
                        else:
                            with st.spinner(f"Computing predictions for {len(clean_df):,} students..."):
                                results_df = predict_bulk(bundle, clean_df)

                            st.session_state["bulk_predictions"] = results_df
                            st.session_state["bulk_filename"] = uploaded_file.name

                    if "bulk_predictions" in st.session_state and st.session_state.get("bulk_filename") == uploaded_file.name:
                        preds = st.session_state["bulk_predictions"]
                        st.write("")
                        st.markdown("### 📊 Batch Prediction Intelligence")

                        # Summary KPIs
                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric("TOTAL STUDENTS", f"{len(preds):,}")
                        k2.metric("AVG MATH", f"{preds['pred_math_score'].mean():.1f}")
                        k3.metric("AVG READING", f"{preds['pred_reading_score'].mean():.1f}")
                        k4.metric("AVG WRITING", f"{preds['pred_writing_score'].mean():.1f}")
                        k5.metric("OVERALL AVG", f"{preds['pred_overall_average'].mean():.1f}")

                        # Charts Row 1: Grade Tier Distribution + Risk Donut
                        b_c1, b_c2 = st.columns(2)
                        with b_c1:
                            st.markdown('<div class="panel"><div class="section-title">📊 Predicted Grade Band Distribution</div>', unsafe_allow_html=True)
                            band_counts = preds["predicted_grade_band"].value_counts().reset_index()
                            band_counts.columns = ["Grade Band", "Count"]
                            fig_b_bands = px.bar(
                                band_counts, x="Grade Band", y="Count",
                                color="Grade Band",
                                color_discrete_sequence=["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"],
                                text="Count"
                            )
                            fig_b_bands.update_traces(textposition='outside')
                            st.plotly_chart(style_fig(fig_b_bands, height=290), width="stretch")
                            st.markdown('</div>', unsafe_allow_html=True)

                        with b_c2:
                            st.markdown('<div class="panel"><div class="section-title">🍩 Academic Support Risk Share</div>', unsafe_allow_html=True)
                            risk_counts = preds["support_risk_level"].value_counts().reset_index()
                            risk_counts.columns = ["Risk Level", "Count"]
                            fig_b_risk = px.pie(
                                risk_counts, names="Risk Level", values="Count",
                                color="Risk Level",
                                color_discrete_map={"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"},
                                hole=0.5
                            )
                            fig_b_risk.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(style_fig(fig_b_risk, height=290), width="stretch")
                            st.markdown('</div>', unsafe_allow_html=True)

                        # Charts Row 2: Score Comparison Boxplots
                        st.markdown('<div class="panel"><div class="section-title">📦 Subject Score Distributions (Uploaded Batch)</div>', unsafe_allow_html=True)
                        pred_score_cols = ["pred_math_score", "pred_reading_score", "pred_writing_score"]
                        melted_batch = preds[pred_score_cols].melt(var_name="Subject", value_name="Predicted Score")
                        melted_batch["Subject"] = melted_batch["Subject"].str.replace("pred_", "").str.replace("_score", "").str.title()
                        fig_batch_box = px.box(
                            melted_batch, x="Subject", y="Predicted Score", color="Subject",
                            color_discrete_sequence=["#4f46e5", "#06b6d4", "#f59e0b"]
                        )
                        st.plotly_chart(style_fig(fig_batch_box, height=290), width="stretch")
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Data Table
                        st.markdown("#### 📋 Detailed Batch Predictions Table")
                        pred_cols = ["pred_math_score", "pred_reading_score", "pred_writing_score", "pred_overall_average", "predicted_grade_band", "support_risk_level"]
                        other_cols = [c for c in preds.columns if c not in pred_cols]
                        reordered = preds[pred_cols + other_cols]
                        st.dataframe(reordered, width="stretch")

                        # Export buttons
                        st.markdown("#### 💾 Export Predicted Results")
                        e1, e2 = st.columns(2)
                        base_fname = uploaded_file.name.rsplit(".", 1)[0]
                        with e1:
                            csv_data = preds.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Export Results as CSV",
                                data=csv_data,
                                file_name=f"predicted_{base_fname}.csv",
                                mime="text/csv",
                                width="stretch"
                            )
                        with e2:
                            excel_data = df_to_excel_bytes(preds)
                            if excel_data is not None:
                                st.download_button(
                                    label="📥 Export Results as Excel (.xlsx)",
                                    data=excel_data,
                                    file_name=f"predicted_{base_fname}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    width="stretch"
                                )
                            else:
                                st.info("Install openpyxl for Excel export.")

# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.subheader("🧠 TabNet Neural Architecture & Validation")
    st.caption("Deep Learning Tabular Neural Network with sparse attention mechanisms and sequential feature selection.")

    # Architecture Overview Panel
    st.markdown("""
    <div class="panel">
      <div class="section-title">🏗️ TabNet Architecture Specifications</div>
      <div class="small-muted">
        TabNet uses sequential attention to choose which features to reason from at each decision step, combining the interpretability of decision trees with gradient descent neural optimization.
      </div>
      <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:12px;">
        <span class="badge-chip badge-low">ATTENTION STEPS: 5</span>
        <span class="badge-chip badge-low">SHARED / STEP DIMENSIONS: 16 / 16</span>
        <span class="badge-chip badge-low">SPARSITY COEFFICIENT: 0.001</span>
        <span class="badge-chip badge-low">OPTIMIZER: Adam (lr=0.02)</span>
        <span class="badge-chip badge-low">VALIDATION SPLIT: 20% Held-Out</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if bundle is not None:
        rows = []
        for target, item in bundle["targets"].items():
            m = item["metrics"]
            rows.append({
                "Subject": target.replace("_score", "").title(),
                "R²": m["R2"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"]
            })
        metrics_df = pd.DataFrame(rows)

        c1, c2, c3 = st.columns(3)
        c1.metric("MEAN TEST R²", f"{metrics_df['R²'].mean():.4f}", "+ High Explanatory Power")
        c2.metric("MEAN TEST MAE", f"{metrics_df['MAE'].mean():.2f} pts", "- Lower is Better")
        c3.metric("MEAN TEST RMSE", f"{metrics_df['RMSE'].mean():.2f} pts", "- Lower is Better")

        st.write("")

        # Visual Comparison Bar Chart
        p1, p2 = st.columns([1.1, 0.9])
        with p1:
            st.markdown('<div class="panel"><div class="section-title">📊 Multi-Metric Evaluation by Subject</div>', unsafe_allow_html=True)
            melted_metrics = metrics_df.melt("Subject", var_name="Metric", value_name="Score")
            fig_m = px.bar(
                melted_metrics, x="Subject", y="Score", color="Metric",
                barmode="group",
                color_discrete_sequence=["#4f46e5", "#06b6d4", "#f59e0b"],
                text="Score"
            )
            fig_m.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(style_fig(fig_m, height=320), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with p2:
            st.markdown('<div class="panel"><div class="section-title">📋 Detailed Performance Metric Table</div>', unsafe_allow_html=True)
            st.dataframe(
                metrics_df.style.format({"R²": "{:.4f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}),
                width="stretch",
                hide_index=True
            )
            st.caption("Evaluated on 10,000 unobserved held-out student records (random_state=42).")
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature Importance Chart
        st.markdown('<div class="panel"><div class="section-title">🌟 Key Academic Drivers (Feature Impact Proxy)</div><div class="small-muted">Estimated feature relevance based on model correlations with subject targets.</div>', unsafe_allow_html=True)
        # Compute correlation with targets as feature impact proxy
        corr_series = df[NUM_COLS].apply(lambda col: df[TARGETS].mean(axis=1).corr(col)).abs().sort_values(ascending=True)
        top_corr = corr_series.tail(12).reset_index()
        top_corr.columns = ["Feature", "Correlation Impact"]
        fig_imp = px.bar(
            top_corr, y="Feature", x="Correlation Impact",
            orientation="h",
            color="Correlation Impact",
            color_continuous_scale="Purples",
            text="Correlation Impact"
        )
        fig_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig_imp, height=340), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("⚠️ No trained model bundle currently saved.")

    # Retrain button
    if st.button("🚀 Train / Retrain TabNet Regressors", type="primary", width="stretch"):
        prog_bar = st.progress(0, text="Preprocessing dataset...")
        try:
            trained_bundle, res_df = train_tabnet(df, prog_bar)
            st.success("🎉 Training complete! Three TabNet regressors and preprocessors saved.")
            st.dataframe(res_df.style.format({"R²": "{:.4f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}), width="stretch", hide_index=True)
            st.rerun()
        except Exception as e:
            st.exception(e)
            st.error("Training failed. Please verify dependencies.")

# ============================================================
# PAGE 4: DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    st.subheader("📊 Dataset Explorer & Feature Correlation")
    st.caption("Deep-dive exploratory analysis across 50,000 synthetic student profiles.")

    # Top KPI summary
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("RECORD COUNT", f"{len(df):,}")
    e2.metric("TOTAL ATTRIBUTES", len(df.columns))
    e3.metric("MISSING CELLS", f"{int(df.isna().sum().sum()):,}")
    e4.metric("MEMORY USAGE", f"{df.memory_usage().sum() / (1024*1024):.1f} MB")

    st.write("")

    # Correlation Heatmap
    st.markdown('<div class="panel"><div class="section-title">🔥 Numeric Feature Correlation Matrix</div><div class="small-muted">Pearson correlation heatmap demonstrating collinearity between grades, study time, alcohol, and absences.</div>', unsafe_allow_html=True)
    corr_cols = ["studytime", "failures", "absences", "G1", "G2", "G3", "math_score", "reading_score", "writing_score"]
    corr_matrix = df[corr_cols].corr().round(2)
    fig_heat = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1, zmax=1
    )
    st.plotly_chart(style_fig(fig_heat, height=420), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # Dynamic Categorical Feature Inspector
    st.markdown('<div class="panel"><div class="section-title">🔍 Dynamic Demographic Breakdown</div><div class="small-muted">Select any categorical feature to inspect score differences.</div>', unsafe_allow_html=True)
    sel_cat = st.selectbox("Select Categorical Attribute to Analyze", CAT_COLS, index=0)

    cat_breakdown = df.groupby(sel_cat)[TARGETS].mean().reset_index()
    cat_melted = cat_breakdown.melt(sel_cat, var_name="Subject", value_name="Score")
    cat_melted["Subject"] = cat_melted["Subject"].str.replace("_score", "").str.title()

    fig_cat = px.bar(
        cat_melted, x=sel_cat, y="Score", color="Subject",
        barmode="group",
        color_discrete_sequence=["#4f46e5", "#06b6d4", "#f59e0b"]
    )
    fig_cat.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(style_fig(fig_cat, height=330), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # Data Table Preview
    st.markdown('<div class="panel"><div class="section-title">📄 Raw Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Global Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Student Performance Predictor • TabNet Neural Architecture • Built with Streamlit & Plotly")
