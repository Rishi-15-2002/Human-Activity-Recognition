import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ==========================================================
# HUMAN ACTIVITY RECOGNITION WEB APPLICATION
# Multi-tab Streamlit dashboard
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom styling: black + navy blue theme ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');
    :root { --navy:#0b1f3a; --blue:#123e70; --bright:#2d6ca8; --ink:#050a12; --muted:#b7c7dc; }
    .stApp { background: linear-gradient(145deg, #050a12 0%, #0a1930 52%, #102e53 100%); color:#edf4ff; }
    [data-testid="stHeader"] { background:rgba(0,0,0,0); }
    .main .block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1450px; }
    h1,h2,h3,h4 { font-family:'DM Serif Display', Georgia, serif !important; font-style:italic !important; color:#f4f7ff !important; }
    p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { color:#d8e4f3; }
    .hero { padding:1.8rem 2rem; border-radius:22px; background:linear-gradient(115deg,#050a12 0%,#0b1f3a 52%,#174b7e 100%); color:white; margin-bottom:1.2rem; border:1px solid #294c72; box-shadow:0 14px 35px rgba(0,0,0,.28); }
    .hero h1 { color:white; margin:0; font-size:2.35rem; font-style:italic; }
    .hero p { color:#d9e8fb; margin:.55rem 0 0; font-size:1.02rem; }
    .section-card { background:linear-gradient(145deg,rgba(15,42,73,.96),rgba(7,20,38,.96)); border:1px solid #31577e; border-radius:18px; padding:1.25rem 1.35rem; margin-bottom:.9rem; box-shadow:0 9px 24px rgba(0,0,0,.22); color:#edf4ff; }
    .section-card b { color:#a9d5ff; }
    .image-note { color:#9fb8d3; font-size:.82rem; font-style:italic; margin-top:-.35rem; }
    .kicker { color:#82bfff; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; }
    .callout { border-left:4px solid #63aef2; padding:.8rem 1rem; background:rgba(35,91,143,.18); border-radius:0 12px 12px 0; margin:.7rem 0 1rem; }
    .small-muted { color:#afc2d9; font-size:.9rem; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#0b1f3a,#12365d); border:1px solid #31577e; padding:14px 16px; border-radius:15px; box-shadow:0 5px 16px rgba(0,0,0,.2); }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#f4f7ff !important; }


    /* =========================================================
       HIGH-VISIBILITY 3D TAB NAVIGATION
       Broad BaseWeb selectors ensure styles reach Streamlit tabs.
       ========================================================= */
    div[data-testid="stTabs"] {
        position:relative !important;
        padding:16px 18px 20px !important;
        margin:8px 0 26px !important;
        border:1px solid rgba(123,190,245,.38) !important;
        border-radius:22px !important;
        background:linear-gradient(160deg,rgba(15,43,72,.96),rgba(3,12,24,.98)) !important;
        box-shadow:0 18px 28px rgba(0,0,0,.42),
                   inset 0 2px 0 rgba(255,255,255,.13),
                   inset 0 -7px 14px rgba(0,0,0,.38) !important;
        perspective:1200px !important;
        overflow:visible !important;
    }
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display:flex !important;
        align-items:center !important;
        gap:14px !important;
        flex-wrap:wrap !important;
        border:0 !important;
        background:transparent !important;
        overflow:visible !important;
        transform-style:preserve-3d !important;
    }
    /* Base tab surface: raised block with a visible dark extrusion below it. */
    div[data-testid="stTabs"] [role="tab"],
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        appearance:none !important;
        -webkit-appearance:none !important;
        position:relative !important;
        display:inline-flex !important;
        align-items:center !important;
        justify-content:center !important;
        min-height:48px !important;
        padding:11px 19px 12px !important;
        margin:0 0 8px !important;
        border:1px solid #477ba8 !important;
        border-bottom:1px solid #477ba8 !important;
        border-radius:13px !important;
        background:linear-gradient(145deg,#1b4267 0%,#102b49 52%,#091a2e 100%) !important;
        color:#eaf5ff !important;
        font-size:14px !important;
        font-weight:750 !important;
        line-height:1.2 !important;
        white-space:nowrap !important;
        text-decoration:none !important;
        transform:translate3d(0,0,0) perspective(900px) rotateX(0deg) !important;
        transform-origin:center bottom !important;
        box-shadow:
            0 7px 0 #04101d,
            0 10px 0 #020912,
            0 16px 22px rgba(0,0,0,.48),
            inset 0 1px 0 rgba(255,255,255,.24),
            inset 0 -3px 7px rgba(0,0,0,.28) !important;
        transition:transform .22s ease, box-shadow .22s ease,
                   background .22s ease, border-color .22s ease,
                   filter .22s ease !important;
        cursor:pointer !important;
        overflow:visible !important;
        z-index:1 !important;
    }
    /* Top bevel highlight gives the tab a physical glossy surface. */
    div[data-testid="stTabs"] [role="tab"]::before,
    div[data-testid="stTabs"] [data-baseweb="tab"]::before {
        content:"" !important;
        position:absolute !important;
        left:8px !important;
        right:8px !important;
        top:3px !important;
        height:1px !important;
        border-radius:99px !important;
        background:rgba(255,255,255,.42) !important;
        opacity:.9 !important;
        pointer-events:none !important;
    }
    /* Hover: tab visibly rises, tilts and casts a longer shadow. */
    div[data-testid="stTabs"] [role="tab"]:hover,
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        transform:translate3d(0,-9px,18px) perspective(900px) rotateX(12deg) scale(1.055) !important;
        background:linear-gradient(145deg,#49a8e8 0%,#2478b8 48%,#155080 100%) !important;
        color:#ffffff !important;
        border-color:#c4e8ff !important;
        filter:brightness(1.12) saturate(1.18) !important;
        box-shadow:
            0 11px 0 #0a2946,
            0 16px 0 #041322,
            0 27px 34px rgba(0,0,0,.56),
            0 0 30px rgba(64,169,245,.48),
            inset 0 2px 0 rgba(255,255,255,.42),
            inset 0 -4px 8px rgba(0,0,0,.18) !important;
        z-index:20 !important;
    }
    /* Selected tab stays lifted so users can see the active section. */
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        transform:translate3d(0,-5px,10px) perspective(900px) rotateX(7deg) !important;
        background:linear-gradient(145deg,#56b5f2 0%,#287fbd 48%,#164c79 100%) !important;
        color:#ffffff !important;
        border:1px solid #d0edff !important;
        border-bottom:1px solid #d0edff !important;
        box-shadow:
            0 9px 0 #082440,
            0 14px 0 #041321,
            0 23px 30px rgba(0,0,0,.52),
            0 0 0 3px rgba(111,193,250,.16),
            0 0 32px rgba(66,165,235,.42),
            inset 0 2px 0 rgba(255,255,255,.46),
            inset 0 -4px 8px rgba(0,0,0,.18) !important;
        z-index:12 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display:none !important;
        opacity:0 !important;
    }
    div[data-testid="stTabs"] [role="tab"]:active,
    div[data-testid="stTabs"] [data-baseweb="tab"]:active {
        transform:translate3d(0,4px,0) perspective(900px) rotateX(-4deg) scale(.98) !important;
        box-shadow:0 2px 0 #04101d, 0 5px 8px rgba(0,0,0,.36) !important;
    }
    @media (prefers-reduced-motion:reduce) {
        div[data-testid="stTabs"] [role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            transition:none !important;
        }
    }
    .pill { display:inline-block; background:#dcecff; color:#0b2a4c; padding:6px 12px; border-radius:30px; font-size:.82rem; font-weight:700; margin-right:6px; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#050a12,#0b1f3a 70%,#123e70); }
    [data-testid="stSidebar"] * { color:#edf4ff !important; }
    .landing-shell { margin:-1.2rem -1rem 0; color:#f5f8ff; }
    .landing-cover { min-height:76vh; position:relative; overflow:hidden; display:flex; align-items:center; padding:5rem 6%; border-radius:0 0 28px 28px; background:linear-gradient(132deg,#05080f 0%,#05080f 48.8%,#0b1f3a 49%,#174b7e 100%); border:1px solid #1b3b60; box-shadow:0 18px 45px rgba(0,0,0,.28); }
    .landing-cover:after { content:''; position:absolute; inset:0; background-image:linear-gradient(90deg,rgba(5,8,15,.92),rgba(5,8,15,.40)),url('https://www.mdpi.com/healthcare/healthcare-10-01084/article_deploy/html/images/healthcare-10-01084-g001.png'); background-size:cover; background-position:center right; opacity:.30; mix-blend-mode:screen; pointer-events:none; }
    .identity { max-width:780px; position:relative; z-index:1; }
    .eyebrow { letter-spacing:.22em; text-transform:uppercase; color:#91bce8; font:600 .78rem Inter,sans-serif; }
    .landing-name { font:italic  clamp(2.8rem,6vw,5.6rem) 'DM Serif Display',Georgia,serif; line-height:1.02; margin:1.1rem 0; color:#fff; }
    .landing-rule { width:95px; height:4px; background:#70b5f5; border-radius:8px; margin:1.6rem 0; }
    .identity-grid { display:grid; grid-template-columns:repeat(2,minmax(190px,1fr)); gap:1rem 2rem; max-width:680px; margin-top:2rem; }
    .identity-item { border-left:2px solid #477eae; padding-left:14px; }
    .identity-label { display:block; color:#91bce8; text-transform:uppercase; letter-spacing:.12em; font:600 .7rem Inter,sans-serif; margin-bottom:5px; }
    .identity-value { color:#f5f8ff; font:italic 1.05rem Georgia,serif; }
    .project-panel { min-height:58vh; display:flex; flex-direction:column; justify-content:center; padding:5rem 8%; margin-top:1.2rem; border-radius:24px; background:linear-gradient(135deg,#0b1f3a,#123e70 65%,#174f82); border:1px solid #31577e; }
    .project-title { font:italic clamp(2rem,4vw,3.8rem) 'DM Serif Display',Georgia,serif; line-height:1.12; max-width:900px; margin:.8rem 0 2rem; color:#fff; }
    .mentor { font:italic 1.35rem Georgia,serif; color:#d5e8ff; }
    .enter-wrap { text-align:center; padding:2rem 0 1rem; }
    div.stButton > button {
        background:#f7faff !important; color:#050a12 !important;
        border:1px solid #d9e8f8 !important; border-radius:999px !important;
        padding:.8rem 2.2rem !important; font-weight:800 !important;
        letter-spacing:.06em; min-height:3.2rem; transition:all .2s ease-in-out;
    }
    div.stButton > button p { color:#050a12 !important; }
    div.stButton > button:hover {
        background:#dcecff !important; color:#000000 !important;
        border-color:#ffffff !important; transform:translateY(-2px);
        box-shadow:0 7px 18px rgba(0,0,0,.24);
    }
    .stAlert { border-radius:14px; }
    code { color:#dcecff; }
    @media(max-width:700px) { .landing-cover{padding:3rem 7%;min-height:70vh;} .identity-grid{grid-template-columns:1fr;} .landing-shell{margin-left:-.5rem;margin-right:-.5rem;} }

    /* CONSOLIDATED 3D GLASS NAVIGATION */
    div[data-testid="stTabs"] {
        position:relative !important;
        padding:16px 18px 22px !important;
        margin:10px 0 28px !important;
        border:1px solid rgba(123,190,245,.42) !important;
        border-radius:24px !important;
        background:linear-gradient(155deg,rgba(16,47,78,.97),rgba(3,11,23,.98)) !important;
        box-shadow:0 18px 30px rgba(0,0,0,.48),
                   inset 0 2px 0 rgba(255,255,255,.14),
                   inset 0 -8px 15px rgba(0,0,0,.42) !important;
        perspective:1400px !important;
        overflow:visible !important;
    }
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display:flex !important;
        align-items:center !important;
        gap:14px !important;
        flex-wrap:wrap !important;
        border:0 !important;
        background:transparent !important;
        overflow:visible !important;
        transform-style:preserve-3d !important;
    }
    div[data-testid="stTabs"] [role="tab"],
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        appearance:none !important;
        position:relative !important;
        min-height:49px !important;
        padding:12px 20px !important;
        margin:0 0 9px !important;
        border:1px solid #5689b5 !important;
        border-radius:14px !important;
        background:linear-gradient(145deg,#204c73 0%,#123454 55%,#081a2c 100%) !important;
        color:#eaf5ff !important;
        font-size:14px !important;
        font-weight:750 !important;
        white-space:nowrap !important;
        transform:translate3d(0,0,0) perspective(900px) rotateX(0deg) !important;
        transform-origin:center bottom !important;
        box-shadow:0 8px 0 #04101d,0 12px 0 #020912,
                   0 17px 24px rgba(0,0,0,.5),
                   inset 0 2px 0 rgba(255,255,255,.25),
                   inset 0 -4px 8px rgba(0,0,0,.32) !important;
        transition:transform .25s ease,box-shadow .25s ease,
                   background .25s ease,border-color .25s ease !important;
        cursor:pointer !important;
        z-index:1 !important;
    }
    div[data-testid="stTabs"] [role="tab"]::before,
    div[data-testid="stTabs"] [data-baseweb="tab"]::before {
        content:"" !important;
        position:absolute !important;
        top:4px !important; left:10px !important; right:10px !important;
        height:2px !important; border-radius:99px !important;
        background:rgba(255,255,255,.48) !important;
        pointer-events:none !important;
    }
    div[data-testid="stTabs"] [role="tab"]:hover,
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        transform:translate3d(0,-10px,22px) perspective(900px) rotateX(13deg) scale(1.06) !important;
        background:linear-gradient(145deg,#56b8f4 0%,#287fbd 50%,#164c79 100%) !important;
        color:#fff !important;
        border-color:#d0edff !important;
        box-shadow:0 12px 0 #0a2946,0 18px 0 #041322,
                   0 30px 38px rgba(0,0,0,.58),
                   0 0 32px rgba(64,169,245,.52),
                   inset 0 2px 0 rgba(255,255,255,.48) !important;
        z-index:20 !important;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        transform:translate3d(0,-5px,12px) perspective(900px) rotateX(7deg) !important;
        background:linear-gradient(145deg,#63c0fa 0%,#2b85c4 50%,#164d7a 100%) !important;
        color:#fff !important;
        border:1px solid #d5efff !important;
        box-shadow:0 10px 0 #082440,0 15px 0 #041321,
                   0 24px 32px rgba(0,0,0,.54),
                   0 0 0 3px rgba(111,193,250,.18),
                   0 0 34px rgba(66,165,235,.46),
                   inset 0 2px 0 rgba(255,255,255,.5) !important;
        z-index:12 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display:none !important;
        opacity:0 !important;
    }

    /* SCROLL-REVEAL COLOUR + SLOW ANIMATED AURORA */
    .stApp {
        background:
            radial-gradient(ellipse at 12% 12%,rgba(35,121,194,.26) 0%,transparent 35%),
            radial-gradient(ellipse at 88% 58%,rgba(29,105,177,.25) 0%,transparent 39%),
            linear-gradient(180deg,#030810 0%,#07172b 32%,#0b2441 68%,#123e70 100%) !important;
        background-size:180% 180%,190% 190%,100% 240% !important;
        background-position:0% 0%,100% 100%,center top !important;
        background-attachment:fixed !important;
        animation:har-aurora-scroll 26s ease-in-out infinite alternate;
    }
    @keyframes har-aurora-scroll {
        0% {background-position:0% 0%,100% 100%,center top;}
        50% {background-position:70% 30%,25% 70%,center 45%;}
        100% {background-position:100% 100%,0% 0%,center bottom;}
    }
    section.main {background:transparent !important;}

    /* Lift cards and metric panels; soften other cards while one is hovered. */
    .section-card, div[data-testid="stMetric"] {
        transform:translateZ(0);
        transition:transform .28s ease,box-shadow .28s ease,
                   filter .28s ease,opacity .28s ease,border-color .28s ease !important;
        transform-style:preserve-3d;
    }
    .section-card:hover, div[data-testid="stMetric"]:hover {
        transform:perspective(900px) translateY(-9px) rotateX(2deg) scale(1.025) !important;
        border-color:#9bd8ff !important;
        box-shadow:0 24px 42px rgba(0,0,0,.5),
                   0 0 28px rgba(65,157,225,.28),
                   inset 0 2px 0 rgba(255,255,255,.12) !important;
        filter:brightness(1.13);
        z-index:8;
    }
    .main .block-container:has(.section-card:hover) .section-card:not(:hover),
    .main .block-container:has(div[data-testid="stMetric"]:hover) div[data-testid="stMetric"]:not(:hover) {
        filter:blur(1.6px) brightness(.72);
        opacity:.72;
    }
    div.stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#2e78b8,#174b7e) !important;
        color:#fff !important;
        border:1px solid #9bd0ff !important;
        border-radius:14px !important;
        box-shadow:0 10px 25px rgba(0,0,0,.34),0 0 20px rgba(65,150,230,.24) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform:translateY(-4px) scale(1.025) !important;
        box-shadow:0 17px 34px rgba(0,0,0,.42),0 0 30px rgba(80,170,245,.44) !important;
    }
    @media (prefers-reduced-motion:reduce) {
        .stApp {animation:none !important;}
        div[data-testid="stTabs"] [role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"],
        .section-card,div[data-testid="stMetric"] {transition:none !important;}
    }

</style>
""", unsafe_allow_html=True)

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}
ACTIVITY_LABELS = list(ACTIVITY_NAMES.values())

@st.cache_resource
def load_artifacts():
    required = [
        ARTIFACT_DIR / "knn_har_model.pkl",
        ARTIFACT_DIR / "decision_tree_har_model.pkl",
        ARTIFACT_DIR / "random_forest_har_model.pkl",
        ARTIFACT_DIR / "metadata.json",
        ARTIFACT_DIR / "X_test.csv",
        ARTIFACT_DIR / "y_test.csv",
    ]
    missing = [str(p.relative_to(APP_DIR)) for p in required if not p.exists()]
    if missing:
        return None, missing

    models = {
        "KNN": joblib.load(ARTIFACT_DIR / "knn_har_model.pkl"),
        "Decision Tree": joblib.load(ARTIFACT_DIR / "decision_tree_har_model.pkl"),
        "Random Forest": joblib.load(ARTIFACT_DIR / "random_forest_har_model.pkl"),
    }
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    X_test = pd.read_csv(ARTIFACT_DIR / "X_test.csv")
    y_test = pd.read_csv(ARTIFACT_DIR / "y_test.csv")["activity_id"]
    return (models, metadata, X_test, y_test), []

loaded, missing_files = load_artifacts()

# ---------- PAGE NAVIGATION: INTRODUCTION / DASHBOARD ----------
if "app_page" not in st.session_state:
    st.session_state.app_page = "introduction"

# ---------- FIRST PAGE: STUDENT INTRODUCTION ----------
if st.session_state.app_page == "introduction":
    st.markdown("""
    <div class="landing-shell">
      <section class="landing-cover">
        <div class="identity">
          <div class="eyebrow">Academic project · Data Science</div>
          <div class="landing-name">Hrishikesh<br>Chandra</div>
          <div class="landing-rule"></div>
          <div class="identity-grid">
            <div class="identity-item"><span class="identity-label">Student ID</span><span class="identity-value">26CSM1R05</span></div>
            <div class="identity-item"><span class="identity-label">Programme</span><span class="identity-value">M.Tech · Computer Science and Engineering</span></div>
            <div class="identity-item"><span class="identity-label">Batch</span><span class="identity-value">2026–2028</span></div>
            <div class="identity-item"><span class="identity-label">Institution</span><span class="identity-value">NIT Warangal</span></div>
          </div>
        </div>
      </section>
      <section class="project-panel">
        <div class="eyebrow">Project title</div>
        <div class="project-title">Human Activity Recognition Using Smartphone Sensors</div>
        <div class="eyebrow">Project mentor</div>
        <div class="mentor">Prof. Radha Krishna</div>
      </section>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="enter-wrap">', unsafe_allow_html=True)
    if st.button("ENTER HAR DASHBOARD  →", key="enter_har_dashboard", use_container_width=True):
        st.session_state.app_page = "dashboard"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


top_nav_left, top_nav_right = st.columns([5, 1.7])
with top_nav_right:
    if st.button("← BACK TO INTRODUCTION", key="top_return_to_introduction", use_container_width=True, type="primary"):
        st.session_state.app_page = "introduction"
        st.rerun()

st.markdown("""
<div class="hero">
  <div><span class="pill">DATA SCIENCE PROJECT</span><span class="pill">M.TECH · CSE · 2026–2028</span></div>
  <h1>Human Activity Recognition Using Smartphone Sensors</h1>
  <p>A study of six human activities using machine-learning models trained on smartphone accelerometer and gyroscope features.</p>
</div>
""", unsafe_allow_html=True)

if loaded is None:
    st.error("Model artifacts are not ready yet.")
    st.markdown("Place the exported files in the **artifacts/** folder next to `app.py`.")
    st.markdown("Missing files:")
    for f in missing_files:
        st.write(f"- `{f}`")
    st.info("Return to your training notebook and run the Web App Export cell provided with this project. Then restart the app.")
    st.stop()

models, metadata, X_test, y_test = loaded
feature_names = metadata["feature_names"]
activity_map = {int(k): v for k, v in metadata["activity_map"].items()}
activity_labels = [activity_map[k] for k in sorted(activity_map)]
results = metadata.get("results", {})
best_model_name = metadata.get("best_model", "Random Forest")
best_accuracy = results.get(best_model_name, {}).get("Accuracy", 0.0)

# Sidebar
with st.sidebar:
    st.title("📱 HAR Dashboard")
    st.caption("Hrishikesh Chandra · 26CSM1R05")
    st.caption("M.Tech CSE | 2026–2028")
    st.caption("Mentor: Prof. Pisipati Radha Krishna")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.write("UCI HAR Smartphone Dataset")
    st.write("10,299 observations · 561 features · 6 activities")
    st.markdown("---")
    st.markdown("**Models**")
    st.write("• K-Nearest Neighbors")
    st.write("• Decision Tree")
    st.write("• Random Forest")
    st.markdown("---")
    if st.button("← INTRODUCTION", key="return_to_introduction", use_container_width=True):
        st.session_state.app_page = "introduction"
        st.rerun()
    st.caption("Evaluation uses the official held-out test partition.")

tabs = st.tabs([
    "Overview",
    "Problem Statement",
    "Solution Approach",
    "Model Architecture",
    "Results",
    "Accuracy Test",
    "Feature Insights"
])

# ---------- TAB 1: OVERVIEW ----------
with tabs[0]:
    st.subheader("Project at a glance")
    st.markdown("A smartphone can act as a compact motion-sensing device. This project studies whether its inertial measurements can distinguish everyday human activities using classical machine-learning methods.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", "10,299")
    c2.metric("Input features", "561")
    c3.metric("Activity classes", "6")
    c4.metric("Best model accuracy", f"{best_accuracy*100:.2f}%")

    st.markdown("#### From phone movement to an activity label")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section-card"><div class="kicker">Project objective</div><h3>Recognise everyday movement</h3><p>Use preprocessed accelerometer and gyroscope feature vectors to predict one of six activity classes. The notebook trains three supervised classifiers and evaluates them on the official held-out test partition.</p><p><b>What the dashboard provides</b></p><ul><li>Project background and problem definition</li><li>Dataset and model workflow</li><li>Measured performance comparison</li><li>Test-set evaluation and uploaded-row prediction</li><li>Feature-importance interpretation</li></ul></div>', unsafe_allow_html=True)
    with right:
        st.image("https://www.mdpi.com/healthcare/healthcare-10-01084/article_deploy/html/images/healthcare-10-01084-g001.png", use_container_width=True)
        st.markdown('<div class="image-note">Illustration: smartphone and wearable inertial-sensor placement. Source: MDPI, open-access article on sensor-fusion HAR.</div>', unsafe_allow_html=True)

    st.markdown("#### The six activities in the dataset")
    activity_descriptions = [
        ("🚶", "WALKING", "Periodic body movement during level walking."),
        ("↗️", "WALKING_UPSTAIRS", "Walking with an upward change in elevation."),
        ("↘️", "WALKING_DOWNSTAIRS", "Walking while descending a staircase."),
        ("🪑", "SITTING", "A seated, mostly stationary posture."),
        ("🧍", "STANDING", "An upright, mostly stationary posture."),
        ("🛏️", "LAYING", "A lying, mostly stationary posture."),
    ]
    activity_cols = st.columns(3)
    for idx, (emoji, title, desc) in enumerate(activity_descriptions):
        with activity_cols[idx % 3]:
            st.markdown(f'<div class="section-card"><div style="font-size:1.8rem">{emoji}</div><b>{title}</b><p class="small-muted">{desc}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Dataset protocol:</b> 30 volunteers performed six activities while carrying a waist-mounted smartphone. The inertial sensors recorded at 50 Hz; the published dataset provides a subject-separated training and test split.</div>', unsafe_allow_html=True)
    st.markdown("Source and dataset documentation: [UCI Machine Learning Repository – Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)")

# ---------- TAB 2: PROBLEM STATEMENT ----------
with tabs[1]:
    st.subheader("Problem statement")
    st.markdown("#### Why activity recognition from smartphones?")
    st.write("Modern smartphones contain inertial sensors that measure linear acceleration and angular velocity. Their motion patterns change according to the user's activity. The challenge is to interpret these measurements reliably and map them to meaningful activity categories.")
    st.markdown('<div class="section-card"><div class="kicker">Core research problem</div><h3>Can engineered motion features distinguish six daily activities?</h3><p>Develop a supervised classification workflow that uses the 561 feature variables supplied with the UCI HAR dataset to recognise WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING and LAYING.</p><p>The task is a six-class classification problem. Each input observation is represented by a fixed-length numerical feature vector, and the model returns one predicted activity label.</p></div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="section-card"><b>Why this matters</b><ul><li>Activity-aware fitness and wellness applications</li><li>Context-aware mobile services</li><li>Daily movement summaries</li><li>Potential support for monitoring mobility patterns</li></ul></div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="section-card"><b>Technical challenges</b><ul><li>Different activities can produce overlapping sensor patterns</li><li>Walking and stair movement are related but not identical</li><li>Sitting and standing can have low-motion signatures</li><li>Phone placement and individual movement styles can affect signals</li></ul></div>', unsafe_allow_html=True)

    st.image("https://www.mdpi.com/sensors/sensors-20-06559/article_deploy/html/images/sensors-20-06559-g002-550.jpg", use_container_width=True)
    st.markdown('<div class="image-note">Example accelerometer/gyroscope time-series patterns across locomotion activities. Image source: Sensors (MDPI), smartphone locomotion activity study.</div>', unsafe_allow_html=True)

# ---------- TAB 3: SOLUTION APPROACH ----------
with tabs[2]:
    st.subheader("Solution approach")
    st.write("The workflow starts with the published UCI HAR feature files, preserves the supplied subject-independent split, trains three classical classifiers, and evaluates the resulting models on held-out observations.")
    stages = [
        ("01", "Dataset acquisition", "Load the UCI HAR training and test files, activity labels and the 561 feature names."),
        ("02", "Exploratory analysis", "Inspect dataset dimensions, class counts, feature ranges and missing-value status."),
        ("03", "Feature preparation", "Align feature columns and labels. Apply scaling within the KNN pipeline; tree-based methods use their native split rules."),
        ("04", "Model training", "Fit K-Nearest Neighbors, Decision Tree and Random Forest using the training partition."),
        ("05", "Performance evaluation", "Predict the held-out test set and calculate accuracy, macro precision, macro recall and macro F1-score."),
        ("06", "Model interpretation", "Compare the three classifiers, inspect confusion matrices and review Random Forest feature importance.")
    ]
    for num, title, desc in stages:
        st.markdown(f'<div class="section-card"><span style="color:#79bdff;font-weight:900;font-size:1.05rem">{num}</span> &nbsp; <b>{title}</b><br><span class="small-muted">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("#### What goes into the model?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card"><b>Sensor sources</b><ul><li>Three-axis body acceleration</li><li>Three-axis gravity acceleration</li><li>Three-axis gyroscope angular velocity</li></ul><p class="small-muted">The dataset contains preprocessed features derived from inertial sensor signals.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card"><b>Model input and output</b><p><b>Input:</b> 561 numerical features for one observation.</p><p><b>Output:</b> one of six activity labels.</p><p class="small-muted">The application expects the same feature names and column order used during model training.</p></div>', unsafe_allow_html=True)

    st.image("https://www.mdpi.com/sensors/sensors-18-04132/article_deploy/html/images/sensors-18-04132-g003.png", use_container_width=True)
    st.markdown('<div class="image-note">Illustrative HAR processing pipeline from a related smartphone inertial-sensor study. The diagram is a conceptual reference; the implementation here uses the already engineered 561-feature UCI files.</div>', unsafe_allow_html=True)
    st.markdown("#### Dataset acquisition details")
    d1, d2, d3 = st.columns(3)
    d1.metric("Participants", "30")
    d2.metric("Sampling rate", "50 Hz")
    d3.metric("Window", "2.56 s")
    st.caption("UCI describes 128 readings per fixed-width window with 50% overlap and a 70%/30% subject-based training/test division. The supplied feature dataset contains 7,352 training and 2,947 test observations.")

# ---------- TAB 4: MODEL ARCHITECTURE ----------
with tabs[3]:
    st.subheader("Model training and selection architecture")
    st.graphviz_chart("""
    digraph HAR {
      rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", color="#b8cbd8", fillcolor="#f5fbff", fontname="Arial"];
      edge [color="#5b8295"];
      A [label="UCI HAR Dataset\\n10,299 observations"];
      B [label="Official train/test split\\n7,352 train | 2,947 test"];
      C [label="561 engineered features\\nAccelerometer + Gyroscope"];
      D [label="Preprocessing\\nKNN: StandardScaler\\nTrees: no scaling"];
      E1 [label="KNN (k=5)"];
      E2 [label="Decision Tree"];
      E3 [label="Random Forest (200 trees)"];
      F [label="Evaluation\\nAccuracy | Precision | Recall | F1"];
      G [label="Model comparison + feature importance"];
      A -> B -> C -> D;
      D -> E1; D -> E2; D -> E3;
      E1 -> F; E2 -> F; E3 -> F;
      F -> G;
    }
    """)
    st.markdown("#### Model selection")
    st.write("The selected model is the classifier with the highest macro F1-score in the notebook's measured results.")
    st.dataframe(pd.DataFrame([
        {"Model": name, "Accuracy": vals["Accuracy"], "Macro Precision": vals["Precision"], "Macro Recall": vals["Recall"], "Macro F1": vals["F1-score"]}
        for name, vals in results.items()
    ]).set_index("Model").style.format("{:.4f}"), use_container_width=True)
    st.caption("The KNN scaler is stored inside its trained pipeline, so the same transformation is applied during inference.")
    st.markdown("#### How the three classifiers make decisions")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="section-card"><b>K-Nearest Neighbors</b><p>Compares a new feature vector with nearby training examples and assigns the most common class among its neighbours. Feature scaling is important because distance calculations are sensitive to feature magnitude.</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="section-card"><b>Decision Tree</b><p>Uses a sequence of feature-based splits to divide the training observations into increasingly homogeneous groups. It is relatively interpretable but can overfit if not controlled.</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="section-card"><b>Random Forest</b><p>Combines predictions from multiple decision trees trained with randomness in samples/features. Aggregation can reduce the variance of a single tree and improve generalisation.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Data partition used")
    st.write("The project uses the dataset's predefined training and test files rather than randomly splitting the full dataset again. This preserves the original subject-separated evaluation protocol described by UCI.")

# ---------- TAB 5: RESULTS ----------
with tabs[4]:
    st.subheader("Measured model results")
    metric_df = pd.DataFrame(results).T
    metric_df.index.name = "Model"
    metric_df = metric_df.reset_index()
    c1, c2, c3 = st.columns(3)
    for col, model_name in zip([c1, c2, c3], ["KNN", "Decision Tree", "Random Forest"]):
        col.metric(model_name, f"{results[model_name]['Accuracy']*100:.2f}% accuracy",
                   f"Macro F1 {results[model_name]['F1-score']:.4f}")
    plot_df = metric_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
                             var_name="Metric", value_name="Score")
    fig = px.bar(plot_df, x="Model", y="Score", color="Metric", barmode="group",
                 range_y=[0, 1], title="Comparison of classifier performance",
                 color_discrete_sequence=["#176b87", "#36a3a5", "#f4a261", "#6c63a8"])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Results table")
    st.dataframe(metric_df.set_index("Model").style.format("{:.4f}"), use_container_width=True)

    st.markdown("#### Comparison and improvement within this project")
    st.write(
        f"Random Forest is the best-performing model in the current experiment, with "
        f"{results['Random Forest']['Accuracy']*100:.2f}% test accuracy and "
        f"{results['Random Forest']['F1-score']:.4f} macro F1-score. "
        f"Compared with KNN, its accuracy is higher by "
        f"{(results['Random Forest']['Accuracy']-results['KNN']['Accuracy'])*100:.2f} percentage points; "
        f"compared with Decision Tree, it is higher by "
        f"{(results['Random Forest']['Accuracy']-results['Decision Tree']['Accuracy'])*100:.2f} percentage points."
    )
    st.markdown("#### Reading the evaluation metrics")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="section-card"><b>Accuracy</b><p>Fraction of all test observations that received the correct activity label.</p><b>Precision</b><p>For a class, the fraction of predictions for that class that were correct. Macro precision averages class-wise precision equally.</p></div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="section-card"><b>Recall</b><p>For a class, the fraction of its actual observations correctly identified. Macro recall gives each activity equal weight.</p><b>F1-score</b><p>Harmonic mean of precision and recall. Macro F1 averages class-wise F1 values equally.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Visual interpretation")
    st.write("The confusion matrix shows where the model's predictions agree with the true labels and which activities are most often confused. Strong diagonal values indicate correct classifications; off-diagonal values identify class pairs that may need closer analysis.")

    chosen = st.selectbox("View confusion matrix for", list(models.keys()), index=2)
    preds = models[chosen].predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=sorted(activity_map))
    cm_df = pd.DataFrame(cm, index=activity_labels, columns=activity_labels)
    fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                       labels=dict(x="Predicted activity", y="Actual activity", color="Samples"),
                       title=f"Confusion matrix — {chosen}")
    st.plotly_chart(fig_cm, use_container_width=True)

# ---------- TAB 6: ACCURACY TEST ----------
with tabs[5]:
    st.subheader("Accuracy test and live prediction")
    st.markdown("The first test below evaluates the saved models on the official 2,947-row test partition. You can also upload a CSV containing 561 feature columns and an optional `activity_id` column.")
    st.markdown('<div class="callout"><b>How to use this page:</b> choose a classifier, run the official evaluation to reproduce its test metrics, or upload a correctly formatted feature CSV to obtain predictions. If the uploaded file includes true activity IDs, the app also calculates accuracy for those rows.</div>', unsafe_allow_html=True)
    st.markdown("#### Input format")
    st.code("feature_1, feature_2, ..., feature_561\n... numerical values ...\n\n# Optional target column for evaluation\nactivity_id  # values 1 to 6", language="text")
    model_choice = st.selectbox("Choose model", list(models.keys()), key="accuracy_model")
    if st.button("Run official test-set evaluation", type="primary"):
        with st.spinner("Evaluating model on the official test set..."):
            y_pred = models[model_choice].predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc*100:.2f}%")
        m2.metric("Macro precision", f"{prec:.4f}")
        m3.metric("Macro recall", f"{rec:.4f}")
        m4.metric("Macro F1-score", f"{f1:.4f}")
        report = classification_report(y_test, y_pred, labels=sorted(activity_map),
                                       target_names=[activity_map[k] for k in sorted(activity_map)],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

# ---------- TAB 7: FEATURE INSIGHTS ----------
with tabs[6]:
    st.subheader("Feature insights")
    importance_path = ARTIFACT_DIR / "feature_importance.csv"
    if importance_path.exists():
        # Show the 20 highest-ranked features, strongest at the top.
        imp = (
            pd.read_csv(importance_path)
            .sort_values("Importance", ascending=False)
            .head(20)
            .sort_values("Importance", ascending=True)
        )

        fig = go.Figure()
        bar_colors = [
            f"rgba(48,{int(132 + 92*i/max(len(imp)-1, 1))},{int(190 + 55*i/max(len(imp)-1, 1))},0.98)"
            for i in range(len(imp))
        ]
        fig.add_trace(go.Bar(
            x=imp["Importance"],
            y=imp["Feature"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#C6EDFF", width=0.8)),
            text=[f"{v:.4f}" for v in imp["Importance"]],
            textposition="outside",
            textfont=dict(color="#F4FAFF", size=11),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>"
        ))
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text="TOP 20 RANDOM FOREST FEATURES",
                x=0.02, xanchor="left",
                font=dict(family="Arial, sans-serif", size=20, color="#F3F8FF")
            ),
            height=760,
            autosize=True,
            margin=dict(l=30, r=95, t=90, b=75),
            paper_bgcolor="#071426",
            plot_bgcolor="#0B1D33",
            font=dict(family="Arial, sans-serif", size=13, color="#EAF4FF"),
            showlegend=False,
            xaxis=dict(
                title=dict(text="Feature importance score", font=dict(size=14, color="#D8ECFF")),
                tickfont=dict(size=12, color="#DCEBFA"),
                gridcolor="rgba(172,207,237,0.18)",
                zerolinecolor="rgba(172,207,237,0.35)",
                showline=True, linecolor="rgba(172,207,237,0.35)",
                ticks="outside", tickcolor="#9FC8E8"
            ),
            yaxis=dict(
                title=dict(text="Input feature", font=dict(size=14, color="#D8ECFF")),
                tickfont=dict(size=12, color="#FFFFFF"),
                automargin=True,
                gridcolor="rgba(0,0,0,0)",
                showline=False,
                categoryorder="array",
                categoryarray=imp["Feature"].tolist()
            ),
            hoverlabel=dict(
                bgcolor="#153B5D", bordercolor="#8FD5FF",
                font=dict(color="#FFFFFF", size=13)
            ),
            bargap=0.28
        )
        st.plotly_chart(
            fig, use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False, "responsive": True}
        )
        st.caption("Feature importance indicates how much the fitted Random Forest used a feature for its splits. It does not establish causation.")
        st.markdown("#### How to interpret the ranking")
        st.write("Features near the top contributed more to the Random Forest's split decisions in this fitted model. Importance scores are model-specific: they describe how the forest used the available variables, not whether a feature independently causes or physically determines an activity.")
    else:
        st.info("Feature importance file not found. Run the notebook export cell to generate it.")

    st.markdown("#### Sensor signal context")
    st.image("https://www.mdpi.com/sensors/sensors-20-06559/article_deploy/html/images/sensors-20-06559-g002-550.jpg", use_container_width=True)
    st.markdown('<div class="image-note">Example multi-axis sensor signal patterns during different locomotion activities. Source: Sensors (MDPI).</div>', unsafe_allow_html=True)
    st.markdown("#### Interpretation and limitations")
    st.write("- The dataset features were engineered from accelerometer and gyroscope signals.")
    st.write("- The current web app accepts feature vectors, not raw phone sensor streams.")
    st.write("- Results may vary with phone placement, sensor hardware and people not represented in the dataset.")
    st.write("- A live smartphone application would require a separate sensor-capture and feature-extraction pipeline.")

st.markdown("---")
st.caption("Human Activity Recognition | Data Science Fundamentals | Results are based on the trained notebook models and the official UCI HAR test partition.")
