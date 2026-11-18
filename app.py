import copy
import base64
import html
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

import analytics
import database
from auth import login_or_create, InvalidInputError, WrongPasswordError

# ----------------------------- Theme ---------------------------------------
BG = "#F5F0E8"          # warm editorial cream
NAVY = "#193B3D"        # deep teal
BLUE = "#285D61"        # muted teal
GREEN = "#26764E"       # sage green
AMBER = "#B87B25"       # ochre
RED = "#B95045"         # terracotta
MUTED = "#766F66"       # warm gray
BORDER = "#373B36"      # fine ink borders

STAR_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path fill="#D8AB65" d="M32 3 38 23 58 13 47 32 61 43 40 40 32 61 24 40 3 43 17 32 6 13 26 23Z"/></svg>'
STAR_FAVICON = "data:image/svg+xml;base64," + base64.b64encode(STAR_SVG.encode()).decode()
st.set_page_config(page_title="✳ WattWise", page_icon=STAR_FAVICON, layout="wide")

def ensure_database_ready():
    """Ensure database tables exist, displaying a clear alert if unavailable."""
    try:
        database.init_db()
        return True
    except Exception as e:
        st.error(f"⚠️ Database connection unavailable: {e}")
        st.caption("Please ensure DATABASE_URL is configured and PostgreSQL / Supabase is reachable.")
        return False

db_ready = ensure_database_ready()


DEFAULTS = {
    "logged_in": False,
    "user_id": None,
    "email": "",
    "page_nav": "Home",
    "ocr_values": None,
    "ocr_confidences": None,
    "upload_name": None,
    "saved_upload_name": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,0,0&display=swap');
:root {{ --cream:{BG}; --ink:#1D2927; --teal:{NAVY}; --border:{BORDER}; }}
html, body, [class*="css"], .stApp {{ font-family:'DM Sans', sans-serif; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{ background:{BG} !important; color:#1D2927; }}
[data-testid="stHeader"] {{ background:{BG} !important; }}
[data-testid="stDecoration"] {{ display:none !important; }}
[data-testid="stAppViewContainer"] > .main {{ background:{BG} !important; }}
.main .block-container {{ padding-top:1.5rem; padding-bottom:3rem; max-width:1450px; }}
h1,h2,h3 {{ color:#1D2927 !important; letter-spacing:-.035em; font-weight:800 !important; }}
h1 {{ font-size:clamp(2rem,3vw,3rem) !important; }}
h2 {{ font-size:1.55rem !important; }}
h3 {{ font-size:1.2rem !important; }}
p, label {{ color:#333A35; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color:#746D62 !important; }}
hr {{ border-color:#C9C1B4 !important; }}
/* Editorial bordered panels */
[data-testid="stVerticalBlockBorderWrapper"] {{ background:#F8F4ED !important; border:1px solid #3A403B !important; border-radius:9px !important; box-shadow:none !important; }}
[data-testid="stMetric"] {{ background:#E5EEE7; border:1px solid #343B36; padding:20px 18px; border-radius:8px; min-height:130px; }}
[data-testid="stMetricLabel"] p {{ font-family:'IBM Plex Mono',monospace !important; text-transform:uppercase; font-size:11px !important; letter-spacing:.055em; font-weight:600; color:#44604F; }}
[data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace !important; font-size:clamp(20px,2vw,31px) !important; font-weight:600; letter-spacing:-.065em; color:#162623; }}
/* Forms and tables */
[data-testid="stForm"] {{ border:1px solid #464942 !important; background:#F9F5EF !important; border-radius:8px !important; padding:20px !important; }}
[data-baseweb="input"] > div, [data-baseweb="select"] > div, textarea {{ background:#FFFDF8 !important; border:1px solid #B7B1A5 !important; border-radius:6px !important; }}
[data-testid="stDataFrame"] {{ border:1px solid #B7B1A5; border-radius:7px; overflow:hidden; }}
.stButton > button, [data-testid="stFormSubmitButton"] button {{ border:1px solid #242B28 !important; border-radius:6px !important; background:#F5F0E8 !important; color:#1E2A27 !important; font-weight:700 !important; min-height:42px; box-shadow:none !important; }}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{ background:#DDEAE1 !important; color:#17372E !important; border-color:#17372E !important; }}
[data-testid="stFormSubmitButton"] button {{ background:#203F40 !important; color:#FFFFFF !important; }}
[data-testid="stFormSubmitButton"] button:hover {{ background:#315B57 !important; color:white !important; }}

/* Ensure text inside buttons is visible regardless of Streamlit theme */
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] button *,
[data-testid="stFormSubmitButton"] button p,
[data-testid="stFormSubmitButton"] button span {{ color:#FFFFFF !important; opacity:1 !important; -webkit-text-fill-color:#FFFFFF !important; }}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover * {{ color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }}
.stButton > button,
.stButton > button *,
.stButton > button p,
.stButton > button span {{ color:#1E2A27 !important; opacity:1 !important; -webkit-text-fill-color:#1E2A27 !important; }}


/* Strong contrast for all text, including labels and disabled controls */
[data-testid="stForm"] label p, [data-testid="stForm"] label span,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {{ color:#25332F !important; opacity:1 !important; }}
[data-testid="stForm"] input, [data-testid="stForm"] textarea,
[data-baseweb="input"] input {{ color:#1D2927 !important; -webkit-text-fill-color:#1D2927 !important; }}
[data-testid="stForm"] input::placeholder {{ color:#726D64 !important; opacity:1 !important; }}
[data-testid="stFormSubmitButton"] button:disabled,
[data-testid="stFormSubmitButton"] button:disabled * {{ color:#FFFFFF !important; opacity:1 !important; -webkit-text-fill-color:#FFFFFF !important; }}
[data-testid="stAlert"] p, [data-testid="stAlert"] span {{ opacity:1 !important; }}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ color:#EDF1E8 !important; }}
/* Sidebar */
section[data-testid="stSidebar"] {{ background:#193B3D !important; border-right:1px solid #10292A; min-width:282px !important; max-width:282px !important; }}
section[data-testid="stSidebar"] > div {{ background:#193B3D !important; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap:.55rem; }}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] span {{ color:#EDF1E8; }}
section[data-testid="stSidebar"] hr {{ border-color:#557273 !important; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color:#B6CDC7 !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button {{ justify-content:flex-start !important; gap:12px; font-size:15px !important; font-weight:650 !important; padding:12px 15px !important; min-height:49px !important; border-radius:8px !important; transition:background .15s ease; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {{ background:transparent !important; border-color:transparent !important; color:#EDF1E8 !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] * {{ color:#EDF1E8 !important; -webkit-text-fill-color:#EDF1E8 !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {{ background:#2B5352 !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {{ background:#E3EEE6 !important; border-color:#E3EEE6 !important; color:#193B3D !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] * {{ color:#193B3D !important; -webkit-text-fill-color:#193B3D !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {{ background:#D5E5D7 !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] [data-testid="stIconMaterial"], section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] [data-testid="stIconMaterial"] {{ font-size:20px !important; }}
section[data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible {{ outline:2px solid #D8AB65 !important; outline-offset:2px; }}
section[data-testid="stSidebar"] .st-key-sign_out button {{ background:#F5F0E8 !important; color:#193B3D !important; border-color:#F5F0E8 !important; }}
section[data-testid="stSidebar"] .st-key-sign_out button * {{ color:#193B3D !important; -webkit-text-fill-color:#193B3D !important; }}
section[data-testid="stSidebar"] a {{ color:#D6EBDD !important; overflow-wrap:anywhere; }}
.ww-brand {{ font-size:27px; font-weight:800; letter-spacing:-.05em; color:#F9F5EA; padding-top:18px; }}
.ww-brand span {{ color:#D8AB65 !important; }}
.ww-tagline {{ font-size:12px; letter-spacing:.035em; color:#B8D1C9; margin:4px 0 18px; }}
.ww-nav-heading {{ display:block; margin:18px 0 18px !important; padding:0 0 7px !important; line-height:1.6; position:relative; z-index:1; font-family:'IBM Plex Mono',monospace; font-size:10px; color:#A9C6C0; letter-spacing:.14em; padding-top:5px; }}
.ww-profile-identity {{ display:flex; align-items:center; gap:20px; padding:20px 22px; margin:8px 0 20px; border:1px solid #AFA99F; border-radius:9px; background:#E6EEE7; }}
.ww-avatar {{ width:84px; height:84px; flex-shrink:0; }}
.ww-avatar svg {{ width:100%; height:100%; display:block; }}
.ww-profile-title {{ font-size:22px; font-weight:800; color:#1D302E; }}
.ww-profile-description {{ font-size:13px; color:#67766D; margin-top:4px; }}
.ww-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:600; letter-spacing:.12em; color:#817363; text-transform:uppercase; margin:4px 0 0; }}
.ww-heading {{ font-size:clamp(28px,3vw,42px); font-weight:800; color:#1A2826; letter-spacing:-.055em; margin:0 0 8px; line-height:1.12; }}
.ww-subtitle {{ color:#766D63; font-size:14px; margin:0 0 18px; }}
.ww-topbar {{ background:#204849; color:#E9F0E8; padding:10px 15px; border-radius:6px; display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:.06em; margin-bottom:22px; }}
.ww-topbar b {{ color:#D8B574; }}
.logo {{ text-align:center; color:#214A43; font-size:42px; font-weight:800; margin-bottom:4px; }}
.tagline {{ text-align:center; color:#766D63; font-size:15px; margin-bottom:30px; }}
.welcome {{ text-align:center; color:#1D302E; font-size:32px; font-weight:800; }}
.subtitle {{ text-align:center; color:#766D63; font-size:15px; margin-bottom:24px; }}
@media (max-width:850px) {{ section[data-testid="stSidebar"] {{ min-width:260px !important; max-width:260px !important; }} }}

/* Compact editorial layout and accessible interactive controls */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stAppViewContainer"] > header {{ background-color:{BG} !important; }}
[data-testid="stHeader"] button, [data-testid="stHeader"] svg {{ color:#193B3D !important; fill:currentColor; }}
.main .block-container {{ padding-top:1.15rem !important; padding-bottom:2rem !important; }}
[data-testid="stVerticalBlock"] {{ gap:1rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ padding:0 !important; }}
[data-testid="stVerticalBlockBorderWrapper"] > div {{ padding:0.1rem !important; }}
[data-testid="stPlotlyChart"] {{ max-width:100%; }}
[data-testid="stPlotlyChart"] > div {{ max-width:100%; }}
[data-testid="stWidgetLabel"] p, label, [data-testid="stMarkdownContainer"] p {{ overflow-wrap:anywhere; }}
[data-baseweb="input"] input, [data-baseweb="select"] input, textarea, [data-baseweb="select"] span {{ color:#1D302E !important; -webkit-text-fill-color:#1D302E !important; }}
input::placeholder, textarea::placeholder {{ color:#766F66 !important; opacity:1 !important; }}
input:disabled, textarea:disabled {{ color:#4C5B55 !important; -webkit-text-fill-color:#4C5B55 !important; opacity:1 !important; }}
[data-baseweb="popover"] *, [role="listbox"] *, [role="option"] {{ color:#1D302E !important; }}
[data-baseweb="popover"], [role="listbox"] {{ background:#FFFDF8 !important; }}
[data-testid="stAlert"] {{ color:#1D302E !important; }}
[data-testid="stAlert"] p, [data-testid="stAlert"] span {{ color:#1D302E !important; }}
.stButton > button *, [data-testid="stFormSubmitButton"] button * {{ opacity:1 !important; visibility:visible !important; }}
.stButton > button:disabled, .stButton > button:disabled * {{ color:#59645E !important; -webkit-text-fill-color:#59645E !important; opacity:1 !important; }}
[data-testid="stFormSubmitButton"] button:disabled, [data-testid="stFormSubmitButton"] button:disabled * {{ color:#F5F0E8 !important; -webkit-text-fill-color:#F5F0E8 !important; }}
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible {{ outline:3px solid #D8AB65 !important; outline-offset:2px; }}
section[data-testid="stSidebar"] {{ width:290px !important; min-width:290px !important; max-width:290px !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top:1.4rem; }}
.ww-nav-heading {{ margin:10px 0 6px !important; font-size:11px; }}
.ww-topbar {{ margin-bottom:14px; }}
.ww-subtitle {{ margin-bottom:12px; }}
.ww-summary-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:10px 0 18px; }}
.ww-summary-card {{ min-width:0; min-height:150px; padding:19px 18px; border:1px solid #39443E; border-radius:10px; display:flex; flex-direction:column; justify-content:space-between; gap:14px; }}
.ww-summary-label {{ font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:700; letter-spacing:.065em; color:#41574F; }}
.ww-summary-value {{ font-family:'IBM Plex Mono',monospace; font-size:clamp(19px,1.8vw,28px); font-weight:700; letter-spacing:-.055em; line-height:1.25; color:#1D302E; overflow-wrap:anywhere; }}
.ww-summary-detail {{ font-size:12px; color:#56635B; margin-top:5px; }}
.ww-profile-identity {{ margin:6px 0 14px; }}
.ww-account-detail {{ padding:7px 0; color:#1D302E; overflow-wrap:anywhere; }}
.ww-account-detail strong {{ font-weight:700; }}
.ww-account-label {{ color:#766F66; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}

 .ww-account-grid {{ display:grid; grid-template-columns:minmax(0,1.5fr) minmax(220px,1fr); gap:16px; margin:16px 0 8px; }}
 .ww-account-tile {{ background:#F5F0E8; border:1px solid #CFC5B7; border-radius:10px; padding:19px 21px; min-width:0; }}
 .ww-account-tile--email {{ background:#E3EEE6; margin-top:12px; }}
 .ww-account-tile--consumer {{ background:#DCE9E9; }}
 .ww-account-tile--count {{ background:#EEE2CC; }}
 .ww-account-tile-label {{ color:#766F66; font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; margin-bottom:12px; }}
 .ww-account-tile-value {{ color:#1D302E; font-size:clamp(16px,1.5vw,21px); font-weight:750; line-height:1.4; overflow-wrap:anywhere; }}
 .ww-account-tile--count .ww-account-tile-value {{ font-size:30px; }}
 .ww-account-note {{ color:#766F66; font-size:12px; line-height:1.5; margin-top:10px; }}
 @media(max-width:750px) {{ .ww-account-grid {{ grid-template-columns:1fr; }} .ww-account-tile {{ padding:16px; }} }}
@media(max-width:1150px) {{ .ww-summary-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
@media(max-width:650px) {{ .ww-summary-grid {{ grid-template-columns:1fr; }} .ww-summary-card {{ min-height:118px; }} .ww-profile-identity {{ padding:14px; gap:12px; }} .ww-avatar {{ width:60px; height:60px; }} .main .block-container {{ padding-left:1rem; padding-right:1rem; }} section[data-testid="stSidebar"] {{ min-width:260px !important; max-width:260px !important; }} }}


/* 1. Smooth page entrance */
@keyframes wwPageEnter {{
  from {{
    opacity: 0;
    transform: translateY(8px);
  }}
  to {{
    opacity: 1;
    transform: translateY(0);
  }}
}}

[data-testid="stMainBlockContainer"] {{
  animation: wwPageEnter 0.32s ease-out both;
}}

/* 2. Smooth sidebar hover */
section[data-testid="stSidebar"] [data-testid="stButton"] button {{
  transition:
    background-color 0.2s ease,
    border-color 0.2s ease,
    transform 0.2s ease !important;
}}

section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
  transform: translateX(3px);
}}

/* 3. Staggered summary cards */
@keyframes wwCardEnter {{
  from {{
    opacity: 0;
    transform: translateY(10px);
  }}
  to {{
    opacity: 1;
    transform: translateY(0);
  }}
}}

.ww-summary-card {{
  animation: wwCardEnter 0.4s ease-out both;
}}

.ww-summary-card:nth-child(2) {{
  animation-delay: 0.06s;
}}

.ww-summary-card:nth-child(3) {{
  animation-delay: 0.12s;
}}

.ww-summary-card:nth-child(4) {{
  animation-delay: 0.18s;
}}

/* 4. Accessibility */
@media (prefers-reduced-motion: reduce) {{
  [data-testid="stMainBlockContainer"],
  .ww-summary-card {{
    animation: none !important;
  }}

  section[data-testid="stSidebar"] [data-testid="stButton"] button {{
    transition: none !important;
    transform: none !important;
  }}
}}

</style>
""", unsafe_allow_html=True)


def editorial_header(section, title, subtitle):
    st.markdown(
        f'<div class="ww-topbar"><span><b style="color:#D8AB65">✳</b> <b>WATTWISE</b> &nbsp; · &nbsp; ENERGY INTELLIGENCE</span>'
        f'<span>{section.upper()} &nbsp; · &nbsp; BILL ANALYTICS</span></div>'
        f'<div class="ww-eyebrow">{section}</div>'
        f'<div class="ww-heading">{title}</div>'
        f'<div class="ww-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def login_page():
    st.markdown('<div class="logo"><span style="color:#D8AB65">✳</span> WattWise</div><div class="tagline">Smarter Bills, Brighter Tomorrow.</div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown('<div class="welcome">Welcome to WattWise</div><div class="subtitle">Enter your email and password to continue.</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com", max_chars=254)
            password = st.text_input("Password", placeholder="At least 6 characters", type="password", max_chars=128)
            submitted = st.form_submit_button("Continue", use_container_width=True)
        if submitted:
            try:
                result = login_or_create(email, password, db=database)
                st.session_state.logged_in = True
                st.session_state.user_id = result["user_id"]
                st.session_state.email = result["email"]
                st.session_state.page_nav = "Home"
                st.success("Account created!" if result["is_new"] else "Welcome back!")
                st.rerun()
            except WrongPasswordError:
                st.error("Incorrect password. Please try again.")
            except InvalidInputError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not sign in: {error}")


def user_bills(consumer_no=None):
    """Read only the logged-in user's bills, optionally for one connection."""
    return database.get_all_bills(int(st.session_state.user_id), consumer_no)


def consumer_selector(bills, key, label="Select Consumer Number", allow_all=False):
    """Select a connection from the current user's saved bills."""
    consumers = sorted({str(value).strip() for value in bills.get("consumer_no", pd.Series(dtype=str)).dropna()
                        if str(value).strip()})
    if not consumers:
        return None
    options = (["All consumers"] if allow_all else []) + consumers
    selected = st.selectbox(label, options, key=key,
                            help="Each consumer number represents a separate electricity connection.")
    return None if selected == "All consumers" else selected


def valid_days(value):
    """Convert a database day value to a positive int, or return None."""
    try:
        if value is None or pd.isna(value):
            return None
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def show_slab_status(units, billing_days):
    days = valid_days(billing_days)
    if days is None:
        st.info("Slab projection needs valid billing-day information.")
        return
    try:
        # The project estimates a standard 30-day cycle from the bill's daily rate.
        elapsed = min(days, 30)
        projected = analytics.project_units(float(units), elapsed, 30)
        status = analytics.slab_alert(projected)
    except (ValueError, TypeError) as error:
        st.info(f"Slab projection is unavailable: {error}")
        return

    level = status["level"]
    title = {"SAFE": "Slab Status", "WARNING": "Slab Warning", "CRITICAL": "Slab Critical"}.get(level, "Slab Status")
    color = {"SAFE": GREEN, "WARNING": AMBER, "CRITICAL": RED}.get(level, BLUE)
    boundary = status["boundary"]
    headroom = status["headroom"]

    with st.container(border=True):
        st.subheader(title)
        if level == "CRITICAL":
            st.write(f"Projected usage is **{projected:.1f} units** — 500+ units, highest slab.")
            right_label = "500+ units, highest slab"
            fill = 1.0
        else:
            st.write(f"Projected usage is **{projected:.1f} units**; the next slab boundary is **{boundary} units** (headroom: {headroom:.1f} units).")
            right_label = f"{projected:.0f} / {boundary} units"
            fill = min(max(projected / float(boundary), 0.0), 1.0)
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;gap:12px;font-size:13px;color:{MUTED};margin:8px 0 6px;">
              <span>Current usage</span><span>{right_label}</span>
            </div>
            <div style="height:12px;background:#E5EDF5;border-radius:20px;overflow:hidden;">
              <div style="height:12px;width:{fill * 100:.1f}%;background:{color};border-radius:20px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(status["message"])
    st.caption("Projection is a linear estimate. Slab thresholds and tariff rates are project reference values; verify the current MSEDCL tariff before relying on cost estimates.")


def show_daily_chart(latest):
    days = valid_days(latest.get("billing_days"))
    if days is None:
        st.info("Daily estimate is unavailable because this bill has no valid billing-day count.")
        return
    try:
        units = float(latest["units_consumed"])
        month = str(latest["billing_month"])
        daily = analytics.estimate_daily_load(units, month + "-01", days=days)
        refs = analytics.daily_reference_levels(units, days)["slab_daily_kwh"]
        fig = px.bar(
            daily,
            x="date",
            y="kwh",
            color=daily["is_weekend"].map({False: "Weekday", True: "Weekend"}),
            color_discrete_map={"Weekend": AMBER, "Weekday": BLUE},
            labels={"date": "Date", "kwh": "Estimated consumption (kWh)", "color": "Day type"},
            title="Daily Consumption (Estimated)",
        )
        for boundary, daily_level in refs.items():
            fig.add_hline(
                y=daily_level,
                line_dash="dash",
                line_color=RED if boundary == 500 else MUTED,
                annotation_text=f"{boundary}-unit reference",
                annotation_position="top left",
            )
        fig.update_layout(legend_title_text="Day type", margin=dict(t=55, b=20))
        fig.update_layout(paper_bgcolor="#F5F0E8", plot_bgcolor="#F5F0E8",
                          font=dict(family="DM Sans", color="#243733"),
                          margin=dict(l=10, r=10, t=55, b=15))
        fig.update_xaxes(showgrid=False, linecolor="#B7B1A5")
        fig.update_yaxes(gridcolor="#E3DCD0", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This daily profile is modelled from the bill total and billing period; it is not smart-meter data.")
    except (ValueError, TypeError) as error:
        st.info(f"Daily consumption estimate is unavailable: {error}")


def appliance_optimizer(units):
    with st.container(border=True):
        st.subheader("Appliance Usage Optimizer")
        st.caption("Illustrative estimates based on rated appliance power and assumed usage hours—not measured appliance-level data.")
        catalogue = analytics.get_default_appliances()
        by_name = {item["name"]: item for item in catalogue}
        selected_names = st.multiselect(
            "Select the appliances you own",
            options=list(by_name.keys()),
            default=[],
            key="owned_appliances",
        )
        selected = []
        if selected_names:
            st.markdown("**Your typical daily usage**")
            cols = st.columns(2)
            for index, name in enumerate(selected_names):
                appliance = copy.deepcopy(by_name[name])
                with cols[index % 2]:
                    appliance["hours_day"] = st.number_input(
                        f"{name} — hours/day",
                        min_value=0.0,
                        max_value=24.0,
                        value=float(appliance["hours_day"]),
                        step=0.5,
                        key=f"hours_{name}",
                    )
                selected.append(appliance)

        target = st.number_input(
            "Target monthly consumption (kWh)",
            min_value=0.0,
            max_value=100000.0,
            value=float(max(0, units) * 0.9),
            step=10.0,
            key="optimizer_target",
        )
        if st.button("Optimize selected appliances", disabled=not selected, key="run_optimizer"):
            try:
                recommendations = analytics.optimize_appliances(float(units), float(target), selected)
                summary = analytics.summarize_optimization(float(units), float(target), recommendations)
                st.info(summary["message"])
                recommended_names = {item["name"] for item in recommendations}
                recommendation_by_name = {item["name"]: item for item in recommendations}
                rows = []
                for appliance in selected:
                    recommendation = recommendation_by_name.get(appliance["name"])
                    rows.append({
                        "Appliance": appliance["name"],
                        "Hours/day": round(appliance["hours_day"], 1),
                        "Estimated monthly kWh": round(analytics.appliance_monthly_kwh(appliance), 2),
                        "Optimizer recommendation": (
                            f"Reduce by {recommendation['reduce_hours_per_day']:.2f} h/day"
                            if recommendation else "No reduction recommended"
                        ),
                        "Cut usage?": "Recommended" if appliance["name"] in recommended_names else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if recommendations:
                    st.caption(f"Estimated energy-charge saving: ₹{summary['total_rupee_saving']:.2f}. This excludes fixed charges, duty and taxes.")
            except (ValueError, TypeError) as error:
                st.error(f"Could not calculate appliance recommendations: {error}")


def summary_card(label, value, background, detail=""):
    """Render a responsive metric card without Streamlit metric text clipping."""
    return (f'<div class="ww-summary-card" style="background:{background}">'
            f'<div class="ww-summary-label">{html.escape(str(label))}</div>'
            f'<div><div class="ww-summary-value">{html.escape(str(value))}</div>'
            f'<div class="ww-summary-detail">{html.escape(str(detail))}</div></div></div>')


def go_to_bills():
    st.session_state.page_nav = "Bills"


def home_page():
    editorial_header("Overview", "Electricity Overview", "Your electricity consumption, trends and savings in one place.")
    all_bills = user_bills()
    selected_consumer = consumer_selector(all_bills, "overview_consumer", "Select Consumer Number")
    bills = user_bills(selected_consumer) if selected_consumer else all_bills
    if selected_consumer:
        st.caption(f"Showing analytics only for consumer {selected_consumer}.")
    st.subheader("Usage Summary")
    if bills.empty:
        cards = [summary_card("TOTAL CONSUMPTION", "—", "#E3EEE6"),
                 summary_card("LATEST BILL", "—", "#EEE2CC"),
                 summary_card("DAILY AVERAGE", "—", "#DCE9E9"),
                 summary_card("BILL CHANGE", "Not enough data", "#EBD2C7")]
    else:
        latest = bills.iloc[-1]
        previous = bills.iloc[-2] if len(bills) > 1 else None
        days = valid_days(latest.get("billing_days"))
        avg = float(latest.units_consumed) / days if days else None
        if previous is None:
            change_value, change_detail = "Not enough data", "Upload another bill to compare"
        else:
            difference = float(latest.total_amount) - float(previous.total_amount)
            old_amount = float(previous.total_amount)
            if old_amount == 0:
                change_value = "N/A (₹0 baseline)"
            else:
                pct = abs(difference) / abs(old_amount) * 100
                change_value = ("↑ " if difference > 0 else "↓ " if difference < 0 else "→ ") + f"{pct:.1f}%"
            change_detail = f"{'+' if difference > 0 else '−' if difference < 0 else ''}₹{abs(difference):,.2f} vs previous bill"
        cards = [summary_card("TOTAL CONSUMPTION", f"{float(bills.units_consumed.sum()):,.0f} kWh", "#E3EEE6", "Across saved bills"),
                 summary_card("LATEST BILL", f"₹{float(latest.total_amount):,.2f}", "#EEE2CC", str(latest.billing_month)),
                 summary_card("DAILY AVERAGE", f"{avg:.2f} kWh/day" if avg is not None else "Unavailable", "#DCE9E9", "Latest billing cycle"),
                 summary_card("BILL CHANGE", change_value, "#EBD2C7", change_detail)]
    st.markdown('<div class="ww-summary-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
    st.button("+ Upload New Bill", key="quick_upload", on_click=go_to_bills, use_container_width=False)
    if bills.empty:
        st.info("Upload your first electricity bill to see dashboard analytics.")
        return

    latest = bills.iloc[-1]
    st.divider()
    with st.container(border=True):
        st.subheader("Monthly Electricity Consumption")
        monthly_fig = px.bar(
            bills, x="billing_month", y="units_consumed",
            labels={"billing_month": "Month", "units_consumed": "Units (kWh)"},
            color_discrete_sequence=[BLUE],
        )
        monthly_fig.update_layout(paper_bgcolor="#F5F0E8", plot_bgcolor="#F5F0E8",
                                  font=dict(family="DM Sans", color="#243733"))
        monthly_fig.update_xaxes(showgrid=False)
        monthly_fig.update_yaxes(gridcolor="#E3DCD0", zeroline=False)
        st.plotly_chart(monthly_fig, use_container_width=True)
    with st.container(border=True):
        st.subheader("Average Daily Consumption")
        daily_months = bills[pd.to_numeric(bills.billing_days, errors="coerce") > 0].copy()
        if daily_months.empty:
            st.info("Valid billing-day details are not available.")
        else:
            daily_months["daily_average"] = daily_months.units_consumed / daily_months.billing_days
            daily_fig = px.bar(
                daily_months, x="billing_month", y="daily_average",
                labels={"billing_month": "Month", "daily_average": "Units per day (kWh)"},
                color_discrete_sequence=[GREEN],
            )
            daily_fig.update_layout(paper_bgcolor="#F5F0E8", plot_bgcolor="#F5F0E8",
                                    font=dict(family="DM Sans", color="#243733"))
            daily_fig.update_xaxes(showgrid=False)
            daily_fig.update_yaxes(gridcolor="#E3DCD0", zeroline=False)
            st.plotly_chart(daily_fig, use_container_width=True)
    with st.container(border=True):
        st.subheader("Daily Consumption (Estimated)")
        show_daily_chart(latest)
    st.divider()
    show_slab_status(latest.units_consumed, latest.billing_days)
    st.divider()
    appliance_optimizer(float(latest.units_consumed))
    st.divider()


def bills_page():
    editorial_header("Bill ledger", "Bill History", "Upload a clear MSEDCL electricity bill and verify the extracted details before saving.")
    uploaded = st.file_uploader("Choose bill image", type=["jpg", "jpeg", "png"], key="bill_upload")
    if uploaded is None:
        # Allow the same file to be uploaded again after it was saved/removed.
        st.session_state.saved_upload_name = None
    if uploaded is not None and st.session_state.saved_upload_name != uploaded.name:
        if st.session_state.upload_name != uploaded.name:
            st.session_state.upload_name = uploaded.name
            st.session_state.ocr_values = None
            st.session_state.ocr_confidences = None
        if st.button("Read bill with OCR", icon=":material/document_scanner:"):
            try:
                import ocr_engine
                with st.spinner("Reading bill image… this may take a little time the first time."):
                    extracted = ocr_engine.extract_bill(uploaded.getvalue())
                st.session_state.ocr_values = {key: value[0] for key, value in extracted.items()}
                st.session_state.ocr_confidences = {key: value[1] for key, value in extracted.items()}
                st.success("Bill scanned. Please verify every field below.")
            except Exception as error:
                st.session_state.ocr_values = {}
                st.session_state.ocr_confidences = {}
                st.error(f"OCR could not read this image: {error}")
                st.info("You can still enter the bill details manually below.")

    values = st.session_state.ocr_values or {}
    confidence = st.session_state.ocr_confidences or {}
    if (uploaded is not None and st.session_state.saved_upload_name != uploaded.name) or values:
        with st.container(border=True):
            st.subheader("Review and confirm bill details")
            if confidence:
                low = [name for name, score in confidence.items() if score is None or score < 0.65]
                if low:
                    st.warning("Please carefully check fields with low or missing OCR confidence: " + ", ".join(low))
            # Outside the form so the date picker appears immediately when ticked or when OCR extracted a due date.
            default_has_due = bool(values.get("due_date"))
            has_due_date = st.checkbox("Add bill due date", value=default_has_due, key="has_due_date")
            with st.form("save_bill_form"):
                c1, c2 = st.columns(2)
                with c1:
                    consumer = st.text_input("Consumer number", value=str(values.get("consumer_no") or ""))
                    month_default = str(values.get("bill_month") or date.today().strftime("%Y-%m")).strip()
                    try:
                        month_date = date.fromisoformat(month_default + "-01")
                    except (TypeError, ValueError):
                        month_date = date.today().replace(day=1)
                    billing_month = st.text_input("Billing month (YYYY-MM)", value=month_date.strftime("%Y-%m"), help="Use format YYYY-MM, for example 2026-10.")
                    units = st.number_input("Units consumed (kWh)", min_value=0, value=int(values.get("units") or 0), step=1)
                    billing_days = st.number_input("Billing days", min_value=1, max_value=366, value=30, step=1)
                with c2:
                    prev = st.number_input("Previous meter reading", min_value=0, value=int(values.get("prev_reading") or 0), step=1)
                    curr = st.number_input("Current meter reading", min_value=0, value=int(values.get("curr_reading") or 0), step=1)
                    amount = st.number_input("Total bill amount (₹)", min_value=0.0, value=float(values.get("total_amount") or 0.0), step=10.0)
                    default_due_val = date.today()
                    if values.get("due_date"):
                        try:
                            default_due_val = date.fromisoformat(str(values["due_date"]))
                        except (ValueError, TypeError):
                            default_due_val = date.today()
                    due_date = st.date_input("Bill due date", value=default_due_val) if has_due_date else None
                confirmed = st.form_submit_button("Confirm & Save Bill", use_container_width=True)
            try:
                estimated_energy = analytics.energy_charge(float(units))["total"]
                st.caption(f"Estimated energy charge (excludes fixed charges, duty, taxes): ₹{estimated_energy:,.2f}")
            except (ValueError, TypeError):
                st.caption("Estimated energy charge (excludes fixed charges, duty, taxes) will appear after valid units are entered.")
            if confirmed:
                try:
                    billing_month = billing_month.strip()
                    parsed_month = date.fromisoformat(billing_month + "-01")
                    if len(billing_month) != 7:
                        raise ValueError("Billing month must be in YYYY-MM format.")
                    energy = analytics.energy_charge(float(units))
                    found_conf = [v for v in confidence.values() if v]
                    bill = {
                        "consumer_no": consumer,
                        "billing_month": parsed_month.strftime("%Y-%m"),
                        "prev_reading": int(prev) if prev else None,
                        "curr_reading": int(curr) if curr else None,
                        "units_consumed": int(units),
                        "billing_days": int(billing_days),
                        "total_amount": float(amount),
                        "energy_charge": energy["total"],
                        "slab": analytics.current_slab(float(units)),
                        "ocr_confidence": min(found_conf) if found_conf else None,
                        "due_date": due_date.isoformat() if due_date else None,
                    }
                    if not consumer.strip() or amount <= 0:
                        raise ValueError("Enter a consumer number and a valid total bill amount.")
                    if curr and prev and curr < prev:
                        raise ValueError("Current meter reading cannot be smaller than previous reading.")
                    database.insert_bill(bill, int(st.session_state.user_id))
                    st.success("Bill saved successfully! Your dashboard has been updated.")
                    st.session_state.ocr_values = None
                    st.session_state.ocr_confidences = None
                    st.session_state.saved_upload_name = uploaded.name if uploaded is not None else None
                    st.session_state.upload_name = None
                    st.rerun()
                except database.DuplicateBillError as error:
                    st.error(str(error))
                except (database.InvalidBillError, ValueError, TypeError) as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Could not save bill: {error}")

    st.divider()
    with st.container(border=True):
        st.subheader("Bill History")
        bills = user_bills()
        if bills.empty:
            st.info("You have not saved any bills yet.")
        else:
            selected_consumer = consumer_selector(bills, "history_consumer", "Filter by consumer number", allow_all=True)
            consumer_bills = (bills[bills["consumer_no"].astype(str) == selected_consumer]
                              if selected_consumer else bills)
            months = sorted(consumer_bills["billing_month"].dropna().astype(str).unique().tolist(), reverse=True)
            selected_month = st.selectbox("Filter by billing month", ["All months"] + months, key="history_month_filter")
            filtered = (consumer_bills if selected_month == "All months"
                        else consumer_bills[consumer_bills["billing_month"].astype(str) == selected_month])
            columns = ["billing_month", "consumer_no", "units_consumed", "total_amount"]
            if "due_date" in filtered.columns:
                columns.append("due_date")
            display = filtered[columns].copy()
            display.rename(columns={"billing_month": "Billing Month", "consumer_no": "Consumer Number",
                                    "units_consumed": "Units (kWh)", "total_amount": "Amount (₹)",
                                    "due_date": "Due Date"}, inplace=True)
            st.dataframe(display, use_container_width=True, hide_index=True)
            export_columns = ["billing_month", "consumer_no", "units_consumed", "total_amount"]
            if "due_date" in filtered.columns:
                export_columns.append("due_date")
            csv_data = filtered[export_columns].to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download CSV", data=csv_data, file_name="wattwise_bill_history.csv",
                               mime="text/csv", use_container_width=False, key="download_bill_history")
            options = {f"{row.billing_month} | {row.consumer_no} | ₹{row.total_amount:.2f}": int(row.id)
                       for row in filtered.itertuples()}
            if options:
                selected = st.selectbox("Select a bill to delete", list(options.keys()))
                if st.button("Delete selected bill", icon=":material/delete:"):
                    database.delete_bill(options[selected], int(st.session_state.user_id))
                    st.success("Bill deleted.")
                    st.rerun()
            else:
                st.info("No bills match these filters.")


def notifications_page():
    editorial_header("Alerts", "Notifications", "Due dates and usage changes from your saved bills.")
    user_id = int(st.session_state.user_id)
    prefs = database.get_user_prefs(user_id)
    if not prefs["notifications_enabled"]:
        st.info("In-app notifications and usage insights are turned off. You can turn them on from My Profile.")
        return

    all_bills = user_bills()
    if all_bills.empty:
        st.info("Upload a bill to receive due-date reminders and usage insights here.")
        return
    selected_consumer = consumer_selector(all_bills, "notifications_consumer", "Select Consumer Number")
    bills = user_bills(selected_consumer) if selected_consumer else all_bills
    if selected_consumer:
        st.caption(f"Showing reminders and insights only for consumer {selected_consumer}.")

    st.subheader("Bill Due-Date Reminders")
    due_bills = bills[bills["due_date"].notna() & (bills["due_date"].astype(str).str.strip() != "")].copy()
    if due_bills.empty:
        st.info("No due dates have been added yet. Add a due date when saving a bill in the Bills section.")
    else:
        due_bills["due_date"] = pd.to_datetime(due_bills["due_date"], errors="coerce").dt.date
        due_bills = due_bills[due_bills["due_date"].notna()].sort_values("due_date")
        today = date.today()
        for row in due_bills.itertuples():
            due = row.due_date
            label = f"{row.billing_month} bill — due {due.strftime('%d %b %Y')}"
            if due < today:
                st.error(f"{label} (overdue)")
            elif due == today:
                st.warning(f"{label} (due today)")
            else:
                days_left = (due - today).days
                st.info(f"{label} (due in {days_left} day{'s' if days_left != 1 else ''})")

    st.divider()
    st.subheader("Consumption Warnings & Usage Insights")
    latest = bills.iloc[-1]
    if len(bills) > 1:
        previous = bills.iloc[-2]
        change = float(latest.units_consumed) - float(previous.units_consumed)
        if change > 0:
            st.warning(f"Your latest usage is {change:.0f} units higher than the previous bill.")
        elif change < 0:
            st.success(f"Your latest usage is {abs(change):.0f} units lower than the previous bill.")
        else:
            st.info("Your usage is unchanged from the previous bill.")
    else:
        st.info("Add another month's bill to compare your consumption.")

    days = valid_days(latest.get("billing_days"))
    if days:
        try:
            projection = analytics.project_units(float(latest.units_consumed), min(days, 30), 30)
            result = analytics.slab_alert(projection)
            if result["level"] == "WARNING":
                st.warning(result["message"])
            elif result["level"] == "CRITICAL":
                st.error(result["message"])
            else:
                st.success(result["message"])
        except (ValueError, TypeError):
            st.info("Slab insight is unavailable because billing-day information is invalid.")
    st.caption("These are in-app reminders and insights. Automatic reminders while the app is closed are not configured.")


def profile_page():
    editorial_header("Account", "My Profile", "Your account and notification preferences.")
    st.markdown('''<div class="ww-profile-identity">
      <div class="ww-avatar" aria-label="Profile avatar">
        <svg viewBox="0 0 100 100" role="img" aria-label="Person" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50" cy="50" r="49" fill="#416F78"/>
          <circle cx="50" cy="34" r="14" fill="#F5F0E8"/>
          <path d="M23 76c0-17 12-27 27-27s27 10 27 27c0 4-3 7-7 7H30c-4 0-7-3-7-7Z" fill="#F5F0E8"/>
        </svg>
      </div>
      <div><div class="ww-profile-title">Your account</div><div class="ww-profile-description">Manage your WattWise preferences</div></div>
    </div>''', unsafe_allow_html=True)
    user_id = int(st.session_state.user_id)
    bills = user_bills()
    with st.container(border=True):
        st.subheader("Account Information")
        if bills.empty or "consumer_no" not in bills.columns:
            consumers = []
        else:
            consumers = sorted({str(v).strip() for v in bills["consumer_no"].dropna() if str(v).strip()})
        consumer_display = ", ".join(consumers) if consumers else "No consumer number linked yet."
        email_display = html.escape(str(st.session_state.email))
        consumer_display_safe = html.escape(consumer_display)
        st.markdown(
            f'<div class="ww-account-tile ww-account-tile--email">'
            f'<div class="ww-account-tile-label">Registered email</div>'
            f'<div class="ww-account-tile-value">{email_display}</div></div>'
            f'<div class="ww-account-grid">'
            f'<div class="ww-account-tile ww-account-tile--consumer">'
            f'<div class="ww-account-tile-label">Consumer number(s)</div>'
            f'<div class="ww-account-tile-value">{consumer_display_safe}</div></div>'
            f'<div class="ww-account-tile ww-account-tile--count">'
            f'<div class="ww-account-tile-label">Bills uploaded</div>'
            f'<div class="ww-account-tile-value">{len(bills)}</div></div></div>',
            unsafe_allow_html=True,
        )
        if len(consumers) > 1:
            st.caption("Multiple consumer numbers are associated with this account; they may represent different connections.")
    with st.container(border=True):
        st.subheader("Preferences")
        prefs = database.get_user_prefs(user_id)
        notifications_enabled = st.toggle(
            "Enable in-app notifications and usage insights",
            value=prefs["notifications_enabled"],
            key="notifications_enabled_toggle",
        )
        if notifications_enabled != prefs["notifications_enabled"]:
            try:
                database.set_notifications_enabled(user_id, notifications_enabled)
                st.success("Notification preference saved.")
            except (database.InvalidUserError, RuntimeError) as error:
                st.error(f"Could not save notification preference: {error}")


def main():
    if not st.session_state.logged_in:
        login_page()
        return
    with st.sidebar:
        st.markdown('<div class="ww-brand"><span>✳</span> WattWise</div>'
                    '<div class="ww-tagline">SMARTER BILLS, BRIGHTER TOMORROW.</div>',
                    unsafe_allow_html=True)
        st.divider()
        pages = ["Home", "Notifications", "Bills", "Profile"]
        if st.session_state.page_nav not in pages:
            st.session_state.page_nav = "Home"
        st.markdown('<div class="ww-nav-heading">NAVIGATION</div>', unsafe_allow_html=True)
        nav_items = [
            ("Home", "Overview", ":material/dashboard:"),
            ("Notifications", "Notifications", ":material/notifications:"),
            ("Bills", "Bill Ledger", ":material/receipt_long:"),
            ("Profile", "My Profile", ":material/person:"),
        ]
        for target, label, nav_icon in nav_items:
            st.button(label, icon=nav_icon, key=f"nav_{target}", use_container_width=True,
                      type="primary" if st.session_state.page_nav == target else "secondary",
                      on_click=lambda page=target: st.session_state.update(page_nav=page))
        st.divider()
        st.caption("SIGNED IN")
        st.write(st.session_state.email)
        if st.button("Sign Out", use_container_width=True, icon=":material/logout:", key="sign_out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    page = st.session_state.page_nav
    if page == "Home":
        home_page()
    elif page == "Bills":
        bills_page()
    elif page == "Notifications":
        notifications_page()
    elif page == "Profile":
        profile_page()


if __name__ == "__main__":
    main()