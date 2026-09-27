"""
Khmer Social Media Monitor & Content Moderation Dashboard
Production Streamlit Application with Dual Monitoring Modes:
- Mode A: My Channel / Page Monitor (Authenticated with Red Alert Moderation Queue)
- Mode B: Public Page & Content Explorer (External Tracking & Community Intelligence)
Features:
- Live YouTube & Facebook Connectors
- Khmer NLP Syllable Tokenization & Contextual Sentiment Analysis
- Content Moderation & Toxicity Detection (Profanity, Hate Speech, Threats, Scams)
- Review Actions & Actionable CSV Export with UTF-8-SIG (BOM)
- Multi-User Authentication & Session Management
"""

from __future__ import annotations
import os
import sys
from datetime import datetime
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup path so local modules import cleanly
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from auth.user_manager import UserManager
from connectors.youtube_connector import YouTubeConnector
from connectors.facebook_connector import FacebookConnector
from connectors.page_explorer import PageExplorer
from moderation.toxicity_detector import KhmerToxicityDetector
from nlp.preprocessor import KhmerPreprocessor
from nlp.sentiment_engine import KhmerSentimentEngine
from nlp.sample_data import MOCK_YOUTUBE_COMMENTS, MOCK_FACEBOOK_COMMENTS

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Khmer Social Media Monitor & Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling & Typography (Kantumruy Pro + Inter)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Kantumruy+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Kantumruy Pro', sans-serif;
    }

    .khmer-font {
        font-family: 'Kantumruy Pro', sans-serif !important;
        line-height: 1.6;
    }

    /* Auth Card */
    .auth-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
    }

    .auth-header {
        text-align: center;
        margin-bottom: 25px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-subtext {
        font-size: 0.8rem;
        margin-top: 2px;
        font-weight: 500;
    }

    /* Sentiment & Moderation Colors */
    .pos-color { color: #10B981 !important; }
    .neg-color { color: #EF4444 !important; }
    .neu-color { color: #94A3B8 !important; }
    .total-color { color: #3B82F6 !important; }
    .nss-color { color: #8B5CF6 !important; }
    .flagged-color { color: #DC2626 !important; }
    .health-color { color: #059669 !important; }

    /* Pill Badges */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.16);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.16);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-neutral {
        background-color: rgba(148, 163, 184, 0.16);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .badge-flagged {
        background-color: rgba(220, 38, 38, 0.2);
        color: #DC2626;
        border: 1px solid rgba(220, 38, 38, 0.4);
    }
    .badge-clean {
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Red Alert Banner */
    .alert-banner {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(239, 68, 68, 0.08) 100%);
        border: 1px solid rgba(220, 38, 38, 0.4);
        border-radius: 12px;
        padding: 14px 20px;
        margin: 15px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* User Profile in Sidebar */
    .user-profile-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 15px;
    }

    /* Header Container */
    .dashboard-header {
        padding: 1.4rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 1.2rem;
    }

    .public-info-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Caching & Service Initialization
# ---------------------------------------------------------
@st.cache_resource
def get_user_manager():
    return UserManager()


@st.cache_resource
def get_sentiment_engine():
    return KhmerSentimentEngine()


@st.cache_resource
def get_preprocessor():
    return KhmerPreprocessor()


@st.cache_resource
def get_toxicity_detector():
    return KhmerToxicityDetector()


user_manager = get_user_manager()
sentiment_engine = get_sentiment_engine()
preprocessor = get_preprocessor()
toxicity_detector = get_toxicity_detector()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "monitoring_mode" not in st.session_state:
    st.session_state.monitoring_mode = "Mode A: 🛡️ My Channel / Page Monitor"
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "public_result" not in st.session_state:
    st.session_state.public_result = None
if "reviewed_comment_ids" not in st.session_state:
    st.session_state.reviewed_comment_ids = set()
if "current_platform" not in st.session_state:
    st.session_state.current_platform = "YouTube"


# ---------------------------------------------------------
# Authentication Landing Page View
# ---------------------------------------------------------
def render_authentication_view():
    """Renders the Login and Sign-Up authentication views when user is not logged in."""
    hero_col1, hero_col2 = st.columns([1.1, 1])

    with hero_col1:
        st.markdown("""
        <div style="padding: 2rem 0;">
            <h1 style="font-size: 2.6rem; font-weight: 800; line-height: 1.2;">
                🛡️ Khmer Social Media Monitor & Content Moderation
            </h1>
            <p style="font-size: 1.15rem; opacity: 0.85; margin: 1rem 0;">
                Real-time AI Sentiment Intelligence and Automated Content Moderation for <b>Facebook Pages</b> and <b>YouTube Channels</b>.
            </p>
            <div style="margin-top: 1.5rem; display: flex; flex-direction: column; gap: 0.6rem;">
                <div>🚨 <b>Toxicity & Harm Detection:</b> Scans for vulgarity, harassment, hate speech, threats, and scams in Khmer script.</div>
                <div>⚡ <b>Red Alert Moderation Queue:</b> Immediate triage of flagged comments with review workflows and actionable CSV exports.</div>
                <div>🌐 <b>Public Page & Content Explorer:</b> External community tracking and brand reputation analysis.</div>
                <div>🧠 <b>Hybrid Khmer NLP:</b> Syllable segmentation, contextual negator windowing, and Scikit-learn TF-IDF models.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="auth-header">
            <h3 style="margin: 0; font-weight: 700;">🔐 User Authentication</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #888;">Sign in to your account or register to access the dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        auth_tab_login, auth_tab_signup = st.tabs(["🔑 Sign In", "📝 Create New Account"])

        with auth_tab_login:
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input("Username or Email", placeholder="e.g. admin@khmer.ai")
                login_password = st.text_input("Password", type="password", placeholder="••••••••")
                login_submit = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")

                if login_submit:
                    success, result = user_manager.authenticate_user(login_username, login_password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = result
                        st.success(f"Welcome back, {result.get('full_name')}!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")

            st.markdown("---")
            st.caption("💡 Quick Demo Credentials: `admin@khmer.ai` / `Password123!`")
            if st.button("⚡ Instant Demo Login (1-Click)", use_container_width=True):
                success, result = user_manager.authenticate_user("admin@khmer.ai", "Password123!")
                if success:
                    st.session_state.authenticated = True
                    st.session_state.current_user = result
                    st.rerun()

        with auth_tab_signup:
            with st.form("signup_form", clear_on_submit=True):
                signup_fullname = st.text_input("Full Name", placeholder="e.g. Bopha Chan")
                signup_username = st.text_input("Username or Email", placeholder="e.g. bopha@example.com")
                signup_password = st.text_input("Password (min 6 characters)", type="password", placeholder="••••••••")
                signup_confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                signup_submit = st.form_submit_button("📝 Register Account", use_container_width=True, type="primary")

                if signup_submit:
                    success, msg = user_manager.register_user(
                        full_name=signup_fullname,
                        username=signup_username,
                        password=signup_password,
                        confirm_password=signup_confirm
                    )
                    if success:
                        st.success(f"✅ {msg}")
                        st.info("👉 Switch to the 'Sign In' tab above to log in.")
                    else:
                        st.error(f"⚠️ {msg}")

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions: Visualizations & Metrics
# ---------------------------------------------------------
COLOR_MAP_SENTIMENT = {
    "Positive": "#10B981",  # Vibrant Emerald
    "Negative": "#EF4444",  # Crimson Red
    "Neutral": "#94A3B8"   # Slate Blue
}

COLOR_MAP_CATEGORY = {
    "Profanity / Vulgarity": "#F59E0B",
    "Hate Speech / Harassment": "#EF4444",
    "Threat / Violence": "#991B1B",
    "Scam / Fraud": "#8B5CF6",
    "Hostile Sentiment": "#EC4899",
    "None": "#10B981"
}


def render_sentiment_and_toxicity_donuts(df: pd.DataFrame):
    """Renders side-by-side Plotly donut charts for sentiment share and content safety."""
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 🍩 Sentiment Distribution")
        counts = df["Sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]

        fig_sent = px.pie(
            counts,
            names="Sentiment",
            values="Count",
            hole=0.55,
            color="Sentiment",
            color_discrete_map=COLOR_MAP_SENTIMENT
        )
        fig_sent.update_traces(textinfo="percent+label", hoverinfo="label+value+percent")
        fig_sent.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            showlegend=False,
            annotations=[dict(text=f"<b>{len(df)}</b><br>Total", x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with c2:
        st.markdown("##### 🛡️ Content Safety & Risk Breakdown")
        
        # Categorize into Clean vs Violation Categories
        cat_counts = df["Violation_Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig_tox = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            hole=0.55,
            color="Category",
            color_discrete_map=COLOR_MAP_CATEGORY
        )
        flagged_count = int(df["Is_Flagged"].sum())
        fig_tox.update_traces(textinfo="percent+label", hoverinfo="label+value+percent")
        fig_tox.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            showlegend=False,
            annotations=[dict(text=f"<b>{flagged_count}</b><br>Flagged", x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig_tox, use_container_width=True)


def render_keyword_frequency_chart(df: pd.DataFrame):
    """Computes and renders frequent Khmer keywords extracted from comments."""
    all_tokens = []
    stopwords = {"នេះ", "នោះ", "ដែល", "បាន", "និង", "ជា", "មាន", "ទៅ", "មក", "មួយ", "ក្នុង", "ឲ្យ", "ឱ្យ", "ថា", "សុំ", "តើ", "បង", "admin"}
    
    for tokens_str in df.get("Cleaned_Tokens", []):
        tokens = str(tokens_str).split()
        for t in tokens:
            t = t.strip()
            if len(t) >= 2 and t not in stopwords and not t.isdigit() and not t in "!@#$%^&*()_+-=~?,.":
                all_tokens.append(t)

    if not all_tokens:
        st.info("No frequent tokens found.")
        return

    top_tokens = Counter(all_tokens).most_common(12)
    kw_df = pd.DataFrame(top_tokens, columns=["Word", "Frequency"])

    fig_kw = px.bar(
        kw_df,
        x="Frequency",
        y="Word",
        orientation="h",
        color="Frequency",
        color_continuous_scale="Blues",
        labels={"Word": "Khmer Keyword", "Frequency": "Mentions"}
    )
    fig_kw.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_kw, use_container_width=True)


# ---------------------------------------------------------
# Mode A: My Channel / Page Monitor
# ---------------------------------------------------------
def render_channel_monitor_mode():
    """Renders Mode A: Connected Page & Channel Monitor with Red Alert Moderation Queue."""
    df = st.session_state.analysis_df

    if df is None or df.empty:
        st.info("👋 Welcome to **My Channel / Page Monitor**. Configure your credentials in the sidebar and click **'🚀 Ingest & Moderate Comments'** to start.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            ### 🛡️ What this Mode Does:
            - **Connects directly** to your YouTube video or Facebook page post.
            - **Scans incoming comments** for profanity, harassment, hate speech, threats, and scams.
            - **Calculates Toxicity Score** and Net Sentiment Score (NSS).
            - **Populates the Red Alert Moderation Queue** for rapid review and action.
            """)
        with c2:
            st.markdown("""
            ### ⚡ Instant Demo:
            - Leave mode set to **'🧪 Demo / Sample Data'** in the sidebar.
            - Click **'🚀 Ingest & Moderate Comments'**.
            - Observe how harmful comments are immediately caught and flagged in the Red Alert tab!
            """)
        return

    # Update moderation status if user reviewed comments
    reviewed_ids = st.session_state.reviewed_comment_ids
    df["Moderation_Status"] = df.apply(
        lambda r: "Reviewed / Approved" if r.get("comment_id") in reviewed_ids 
        else ("Pending Review" if r.get("Is_Flagged") else "Clean"),
        axis=1
    )

    # Summary Metrics
    total_count = len(df)
    pos_count = (df["Sentiment"] == "Positive").sum()
    neg_count = (df["Sentiment"] == "Negative").sum()
    neu_count = (df["Sentiment"] == "Neutral").sum()
    flagged_count = (df["Is_Flagged"] == True).sum()
    pending_count = (df["Moderation_Status"] == "Pending Review").sum()

    pos_pct = (pos_count / total_count * 100) if total_count > 0 else 0
    neg_pct = (neg_count / total_count * 100) if total_count > 0 else 0
    neu_pct = (neu_count / total_count * 100) if total_count > 0 else 0
    flagged_pct = (flagged_count / total_count * 100) if total_count > 0 else 0

    # Healthy ratio (Clean comments %)
    health_ratio = 100.0 - flagged_pct

    # 5 Key Metrics Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Ingested</div>
            <div class="metric-value total-color">{total_count}</div>
            <div class="metric-subtext">Live comments</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Positive</div>
            <div class="metric-value pos-color">{pos_count}</div>
            <div class="metric-subtext pos-color">{pos_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Negative</div>
            <div class="metric-value neg-color">{neg_count}</div>
            <div class="metric-subtext neg-color">{neg_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Neutral</div>
            <div class="metric-value neu-color">{neu_count}</div>
            <div class="metric-subtext neu-color">{neu_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        f_class = "flagged-color" if flagged_count > 0 else "health-color"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚨 Flagged Toxic</div>
            <div class="metric-value {f_class}">{flagged_count}</div>
            <div class="metric-subtext {f_class}">{flagged_pct:.1f}% flagged</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        h_color = "pos-color" if health_ratio >= 80 else ("neg-color" if health_ratio < 60 else "neu-color")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🛡️ Community Health</div>
            <div class="metric-value {h_color}">{health_ratio:.1f}%</div>
            <div class="metric-subtext">Safe content index</div>
        </div>
        """, unsafe_allow_html=True)

    # Red Alert Banner if pending harmful comments exist
    if pending_count > 0:
        st.markdown(f"""
        <div class="alert-banner">
            <div>
                <span style="font-size: 1.25rem; font-weight: 700; color: #DC2626;">🚨 Red Alert: {pending_count} Flagged Inappropriate Comments Require Attention!</span>
                <div style="font-size: 0.9rem; color: #888; margin-top: 3px;">Harmful, vulgar, threat, or scam content detected. Review them in the Moderation Queue below.</div>
            </div>
            <div>
                <span class="badge badge-flagged" style="font-size: 0.95rem; padding: 6px 14px;">ACTION REQUIRED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Main Tabs for Channel Monitor
    tab_overview, tab_queue, tab_explorer, tab_deepdive = st.tabs([
        "📊 Sentiment & Safety Overview",
        f"🚨 Red Alert Moderation Queue ({pending_count})",
        "📋 All Comments Explorer",
        "🔎 Linguistic Inspector"
    ])

    # ---------------------------------------------------------
    # TAB 1: OVERVIEW
    # ---------------------------------------------------------
    with tab_overview:
        render_sentiment_and_toxicity_donuts(df)

        st.markdown("---")
        t_col1, t_col2 = st.columns([1.2, 1])

        with t_col1:
            st.markdown("##### 📈 Comment Activity & Reaction Timeline")
            if "published_at" in df.columns:
                try:
                    temp_df = df.copy()
                    temp_df["dt"] = pd.to_datetime(temp_df["published_at"])
                    temp_df["hour_or_day"] = temp_df["dt"].dt.strftime("%Y-%m-%d %H:00")
                    timeline_data = temp_df.groupby(["hour_or_day", "Sentiment"]).size().reset_index(name="Count")
                    
                    fig_timeline = px.bar(
                        timeline_data,
                        x="hour_or_day",
                        y="Count",
                        color="Sentiment",
                        color_discrete_map=COLOR_MAP_SENTIMENT,
                        barmode="stack",
                        labels={"hour_or_day": "Timestamp", "Count": "Volume"}
                    )
                    fig_timeline.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                except Exception:
                    st.info("Timeline rendering unavailable for current timestamps.")

        with t_col2:
            st.markdown("##### 🔤 Most Frequent Khmer Keywords")
            render_keyword_frequency_chart(df)

    # ---------------------------------------------------------
    # TAB 2: RED ALERT MODERATION QUEUE
    # ---------------------------------------------------------
    with tab_queue:
        st.markdown("### 🚨 Inappropriate & Toxic Content Moderation Queue")
        st.caption("Comments flagged by rule-based blocklists (profanity, hate speech, threats, scams) and hostile sentiment scoring.")

        flagged_df = df[df["Is_Flagged"] == True].copy()

        if flagged_df.empty:
            st.success("🎉 Great news! Zero inappropriate or toxic comments detected in this stream.")
        else:
            # Actionable Export for Legal/Moderation Action
            exp_csv = flagged_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 Export Flagged Comments Report (CSV utf-8-sig)",
                data=exp_csv,
                file_name=f"flagged_moderation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Display individual flagged comment cards with review actions
            for idx, row in flagged_df.iterrows():
                cid = row.get("comment_id", f"c_{idx}")
                is_reviewed = cid in st.session_state.reviewed_comment_ids
                
                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(220, 38, 38, 0.05); border: 1px solid rgba(220, 38, 38, 0.3); border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <b>👤 Author:</b> <code>{row.get('author', 'Anonymous')}</code> &nbsp;|&nbsp; 
                                <b>Category:</b> <span class="badge badge-flagged">{row.get('Violation_Category', 'Flagged')}</span> &nbsp;|&nbsp;
                                <b>Severity:</b> <b>{row.get('Severity', 'N/A')}</b> &nbsp;|&nbsp;
                                <b>Toxicity Score:</b> <code>{row.get('Toxicity_Score', 0.0):.2f}</code>
                            </div>
                            <div>
                                <span class="badge {'badge-clean' if is_reviewed else 'badge-flagged'}">
                                    {'✅ Reviewed' if is_reviewed else '⚠️ Pending Review'}
                                </span>
                            </div>
                        </div>
                        <div style="margin: 10px 0; font-size: 1.05rem;" class="khmer-font">
                            💬 {row.get('Highlighted_Comment', row.get('comment', ''))}
                        </div>
                        <div style="font-size: 0.85rem; color: #888;">
                            <b>Detected Trigger Words:</b> <span style="color: #DC2626; font-weight: 600;">{row.get('Flagged_Keywords') or 'Hostile Pattern'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2, _ = st.columns([1.2, 1.2, 3])
                    with btn_c1:
                        if not is_reviewed:
                            if st.button("✅ Mark Reviewed", key=f"rev_{cid}", use_container_width=True):
                                st.session_state.reviewed_comment_ids.add(cid)
                                st.rerun()
                        else:
                            if st.button("↩️ Re-flag", key=f"unrev_{cid}", use_container_width=True):
                                st.session_state.reviewed_comment_ids.discard(cid)
                                st.rerun()
                    with btn_c2:
                        st.caption(f"ID: {cid}")

    # ---------------------------------------------------------
    # TAB 3: ALL COMMENTS EXPLORER
    # ---------------------------------------------------------
    with tab_explorer:
        st.markdown("### 📋 Filtered Comments Explorer")

        f1, f2, f3 = st.columns([1.5, 1.2, 1])
        with f1:
            search_query = st.text_input("🔍 Search keyword or author:", placeholder="Search Khmer text or username...")
        with f2:
            view_filter = st.selectbox(
                "Filter View:",
                ["All Comments", "🚨 Flagged / Inappropriate Only", "Positive", "Negative", "Neutral"]
            )
        with f3:
            min_conf = st.slider("Min Confidence:", 0.50, 1.00, 0.50, 0.05)

        # Apply Filters
        filtered = df.copy()
        if view_filter == "🚨 Flagged / Inappropriate Only":
            filtered = filtered[filtered["Is_Flagged"] == True]
        elif view_filter in ["Positive", "Negative", "Neutral"]:
            filtered = filtered[filtered["Sentiment"] == view_filter]

        if search_query.strip():
            q = search_query.strip().lower()
            filtered = filtered[
                filtered["comment"].str.lower().str.contains(q, na=False) |
                filtered["author"].str.lower().str.contains(q, na=False)
            ]

        filtered = filtered[filtered["Confidence"] >= min_conf]

        # Display Detected Keyword Hotlist
        hotlist = []
        for kw_str in filtered.get("Flagged_Keywords", []):
            if kw_str and str(kw_str).strip():
                for kw in str(kw_str).split(","):
                    if kw.strip():
                        hotlist.append(kw.strip())

        if hotlist:
            top_hotlist = [k for k, _ in Counter(hotlist).most_common(8)]
            st.markdown(f"**🔥 Active Sensitive Word Hotlist:** " + " ".join([f"<span class='badge badge-flagged'>{w}</span>" for w in top_hotlist]), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        display_cols = [
            "platform", "author", "comment", "Sentiment", "Is_Flagged",
            "Violation_Category", "Toxicity_Score", "Moderation_Status", "like_count"
        ]
        valid_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[valid_cols],
            column_config={
                "platform": st.column_config.TextColumn("Platform", width="small"),
                "author": st.column_config.TextColumn("Author", width="medium"),
                "comment": st.column_config.TextColumn("Khmer Comment", width="large"),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Is_Flagged": st.column_config.CheckboxColumn("🚨 Flagged", width="small"),
                "Violation_Category": st.column_config.TextColumn("Violation", width="medium"),
                "Toxicity_Score": st.column_config.ProgressColumn("Toxicity", min_value=0.0, max_value=1.0, format="%.2f", width="small"),
                "Moderation_Status": st.column_config.TextColumn("Status", width="small"),
                "like_count": st.column_config.NumberColumn("Likes", width="small"),
            },
            use_container_width=True,
            hide_index=True,
            height=380
        )

        # Export All
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            csv_all = filtered.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 Download All (CSV utf-8-sig)",
                data=csv_all,
                file_name=f"all_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            json_all = filtered.to_json(orient="records", force_ascii=False, indent=2)
            st.download_button(
                "📥 Download All (JSON)",
                data=json_all,
                file_name=f"all_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    # ---------------------------------------------------------
    # TAB 4: LINGUISTIC INSPECTOR
    # ---------------------------------------------------------
    with tab_deepdive:
        st.markdown("### 🔎 Deep-Dive Comment Linguistic Diagnostics")
        if not df.empty:
            sel_comment = st.selectbox("Select comment to inspect:", df["comment"].tolist(), key="mode_a_inspect")
            c_row = df[df["comment"] == sel_comment].iloc[0]

            col_l1, col_l2 = st.columns([2, 1])
            with col_l1:
                st.markdown(f"**Original Text:**")
                st.info(f"💬 {c_row['comment']}")
                st.markdown(f"**Segmented Syllable / Word Tokens:**")
                st.code(c_row.get("Cleaned_Tokens", ""))
            with col_l2:
                badge_class = f"badge-{c_row['Sentiment'].lower()}"
                st.markdown(f"**Sentiment:** <span class='badge {badge_class}'>{c_row['Sentiment']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{c_row['Confidence'] * 100:.1f}%`")
                st.markdown(f"**Polarity Score:** `{c_row.get('Sentiment_Score', 0.0)}`")
                st.markdown(f"**Toxicity Score:** `{c_row.get('Toxicity_Score', 0.0):.2f}`")
                st.markdown(f"**Violation Category:** `{c_row.get('Violation_Category', 'None')}`")
                if c_row.get("Flagged_Keywords"):
                    st.markdown(f"**Trigger Sensitive Words:** `{c_row['Flagged_Keywords']}`")


# ---------------------------------------------------------
# Mode B: Public Page & Content Explorer
# ---------------------------------------------------------
def render_public_explorer_mode():
    """Renders Mode B: Public Page and External Channel Tracker."""
    st.markdown("### 🌐 Public Page & Content Explorer")
    st.caption("Track external public Facebook Pages and YouTube Channels, analyze community sentiment trends, and inspect public discussions.")

    exp_col1, exp_col2 = st.columns([2.5, 1])
    with exp_col1:
        public_query = st.text_input(
            "Enter Public Channel / Page / Video Link or Topic:",
            placeholder="e.g. Fresh News Daily, Cambodia Tech, or YouTube/Facebook URL...",
            help="Type a channel name, topic keyword, or public URL."
        )
    with exp_col2:
        pub_platform = st.selectbox("Platform:", ["YouTube", "Facebook"])

    yt_key_val = st.session_state.get("public_yt_key", "")
    with st.expander("⚙️ Optional: Connect YouTube Data API v3 Key for Live Search"):
        yt_key_input = st.text_input(
            "Google API Key (YouTube Data API v3):",
            value=yt_key_val,
            type="password",
            placeholder="AIzaSy...",
            help="If provided, queries will fetch live public YouTube search results and comment threads."
        )
        st.session_state.public_yt_key = yt_key_input

    track_button = st.button("🔎 Track & Analyze Public Page", type="primary", use_container_width=True)

    if track_button or st.session_state.public_result is not None:
        if track_button:
            with st.spinner("Retrieving public page data, recent posts, and community comments..."):
                explorer = PageExplorer(youtube_api_key=st.session_state.get("public_yt_key", ""))
                res = explorer.explore_public_channel(
                    query_or_url=public_query or "Fresh News Daily",
                    platform=pub_platform,
                    max_comments=60
                )
                
                # Analyze comments with sentiment & moderation
                raw_df = pd.DataFrame(res["comments"])
                if not raw_df.empty:
                    analyzed = sentiment_engine.analyze_dataframe(
                        raw_df,
                        text_column="comment",
                        mode="hybrid",
                        include_moderation=True
                    )
                    # Defensive check: ensure moderation columns are present
                    if "Is_Flagged" not in analyzed.columns:
                        detector = get_toxicity_detector()
                        analyzed = detector.scan_dataframe(analyzed, text_col="comment")
                    res["analysis_df"] = analyzed
                else:
                    res["analysis_df"] = pd.DataFrame()

                st.session_state.public_result = res
                st.success(f"🎉 Successfully analyzed public content for '{res['channel_info']['title']}'!")

        res = st.session_state.public_result
        if not res:
            return

        c_info = res["channel_info"]
        pub_df = res.get("analysis_df", pd.DataFrame())

        if not pub_df.empty and "Is_Flagged" not in pub_df.columns:
            detector = get_toxicity_detector()
            pub_df = detector.scan_dataframe(pub_df, text_col="comment")
            res["analysis_df"] = pub_df


        # Public Channel Overview Card
        st.markdown(f"""
        <div class="public-info-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h3 style="margin: 0; font-weight: 700;">{c_info.get('title', 'Public Entity')}</h3>
                    <p style="margin: 4px 0 0 0; opacity: 0.85;">{c_info.get('description', '')}</p>
                    <div style="margin-top: 8px; font-size: 0.9rem;">
                        <b>Platform:</b> <code>{c_info.get('platform', '')}</code> &nbsp;|&nbsp;
                        <b>Category:</b> <code>{c_info.get('category', '')}</code> &nbsp;|&nbsp;
                        <b>Audience Reach:</b> <code>{c_info.get('subscribers', '')}</code>
                    </div>
                </div>
                <div>
                    <span class="badge badge-clean" style="font-size: 0.9rem; padding: 6px 12px;">PUBLIC AUDIT</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recent Posts Showcase
        if res.get("recent_posts"):
            st.markdown("##### 📌 Monitored Post / Video")
            for p in res["recent_posts"]:
                st.markdown(f"- **{p.get('title', 'Post')}** &nbsp;•&nbsp; 💬 `{p.get('comments_count', len(pub_df))} comments` &nbsp;•&nbsp; 👍 `{p.get('likes', 'N/A')} likes`")
            st.markdown("<br>", unsafe_allow_html=True)

        if pub_df.empty:
            st.warning("No public comments found for this stream.")
            return

        # Metrics for Public Explorer
        p_total = len(pub_df)
        p_pos = (pub_df["Sentiment"] == "Positive").sum()
        p_neg = (pub_df["Sentiment"] == "Negative").sum()
        p_neu = (pub_df["Sentiment"] == "Neutral").sum()
        p_flagged = (pub_df["Is_Flagged"] == True).sum()
        p_nss = ((p_pos - p_neg) / p_total * 100) if p_total > 0 else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Analyzed Comments</div>
                <div class="metric-value total-color">{p_total}</div>
                <div class="metric-subtext">Public sample</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Positive Sentiment</div>
                <div class="metric-value pos-color">{p_pos}</div>
                <div class="metric-subtext pos-color">{(p_pos/p_total*100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Negative Sentiment</div>
                <div class="metric-value neg-color">{p_neg}</div>
                <div class="metric-subtext neg-color">{(p_neg/p_total*100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Net Score (NSS)</div>
                <div class="metric-value nss-color">{'+' if p_nss > 0 else ''}{p_nss:.1f}</div>
                <div class="metric-subtext">Brand polarity</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚨 Inappropriate Comments</div>
                <div class="metric-value flagged-color">{p_flagged}</div>
                <div class="metric-subtext flagged-color">{(p_flagged/p_total*100):.1f}% toxic</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("##### 🍩 Public Reaction Sentiment")
            p_counts = pub_df["Sentiment"].value_counts().reset_index()
            p_counts.columns = ["Sentiment", "Count"]
            fig_p = px.pie(
                p_counts,
                names="Sentiment",
                values="Count",
                hole=0.55,
                color="Sentiment",
                color_discrete_map=COLOR_MAP_SENTIMENT
            )
            fig_p.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

        with c_right:
            st.markdown("##### 🔤 Discussion Keywords & Topics")
            render_keyword_frequency_chart(pub_df)

        # Public Comments Table
        st.markdown("---")
        st.markdown("##### 📋 Public Comments Feed")
        st.dataframe(
            pub_df[["author", "comment", "Sentiment", "Confidence", "Is_Flagged", "Violation_Category", "like_count"]],
            column_config={
                "author": st.column_config.TextColumn("Author", width="medium"),
                "comment": st.column_config.TextColumn("Khmer Comment", width="large"),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.5, max_value=1.0, format="%.2f", width="small"),
                "Is_Flagged": st.column_config.CheckboxColumn("🚨 Flagged", width="small"),
                "Violation_Category": st.column_config.TextColumn("Violation", width="medium"),
                "like_count": st.column_config.NumberColumn("Likes", width="small"),
            },
            use_container_width=True,
            hide_index=True,
            height=320
        )

        # Export Public Audit
        p_csv = pub_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 Export Public Audit Report (CSV utf-8-sig)",
            data=p_csv,
            file_name=f"public_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ---------------------------------------------------------
# Authenticated Main App Router
# ---------------------------------------------------------
def render_dashboard_view():
    """Renders the main authenticated dashboard with Dual Monitoring Modes."""
    current_user = st.session_state.get("current_user", {})

    with st.sidebar:
        # User Active Session Card
        st.markdown(f"""
        <div class="user-profile-box">
            <div style="font-size: 0.8rem; text-transform: uppercase; color: #818CF8; font-weight: 600;">Active Session</div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-top: 2px;">👤 {current_user.get('full_name', 'User')}</div>
            <div style="font-size: 0.85rem; opacity: 0.8;">📧 {current_user.get('username', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.analysis_df = None
            st.session_state.public_result = None
            st.rerun()

        st.markdown("---")
        st.markdown("### 🎛️ Monitoring Mode")

        mode_choice = st.radio(
            "Select Dashboard Mode:",
            [
                "Mode A: 🛡️ My Channel / Page Monitor",
                "Mode B: 🌐 Public Page & Content Explorer"
            ],
            index=0 if "Mode A" in st.session_state.monitoring_mode else 1
        )
        st.session_state.monitoring_mode = mode_choice

        # Sidebar parameters for Mode A
        if "Mode A" in mode_choice:
            st.markdown("---")
            st.markdown("### 📌 Social Media Connector")

            platform = st.selectbox(
                "Select Social Platform",
                ["YouTube", "Facebook"],
                index=0 if st.session_state.current_platform == "YouTube" else 1
            )
            st.session_state.current_platform = platform

            data_mode = st.radio(
                "Data Retrieval Mode",
                ["⚡ Live API Stream", "🧪 Demo / Sample Data"],
                index=1
            )

            st.markdown("---")
            st.markdown("#### ⚙️ Connection Settings")

            if platform == "YouTube":
                if data_mode == "⚡ Live API Stream":
                    yt_api_key = st.text_input("YouTube Data API v3 Key", type="password", placeholder="AIzaSy...")
                    yt_input = st.text_input("YouTube Video ID or URL", placeholder="e.g. dQw4w9WgXcQ or URL")
                    yt_order = st.selectbox("Order", ["relevance", "time"])
                else:
                    st.info("💡 Demo: Ingesting 15 realistic comments including positive, negative, and flagged toxic remarks.")
                    yt_input = "demo_youtube_khmer_2026"
                    yt_api_key = "demo_key"
                    yt_order = "relevance"

                max_comments = st.slider("Max Comments", 10, 300, 50, 10)

            else:  # Facebook
                if data_mode == "⚡ Live API Stream":
                    fb_token = st.text_input("Meta Page Access Token", type="password", placeholder="EAA...")
                    fb_post_id = st.text_input("Facebook Post ID or URL", placeholder="Post ID or URL")
                else:
                    st.info("💡 Demo: Ingesting 13 realistic comments with product feedback and flagged scam/vulgar remarks.")
                    fb_post_id = "demo_facebook_post_2026"
                    fb_token = "demo_token"

                max_comments = st.slider("Max Comments", 10, 300, 50, 10)

            st.markdown("---")
            st.markdown("#### 🧠 Classification Engine")
            classification_mode = st.selectbox("Strategy", ["hybrid", "lexicon", "ml"], index=0)

            fetch_button = st.button("🚀 Ingest & Moderate Comments", use_container_width=True, type="primary")

            if fetch_button:
                raw_comments = []
                fetch_success = False

                with st.spinner("Connecting to social media feed and retrieving comments..."):
                    try:
                        if data_mode == "🧪 Demo / Sample Data":
                            if platform == "YouTube":
                                raw_comments = MOCK_YOUTUBE_COMMENTS[:max_comments]
                            else:
                                raw_comments = MOCK_FACEBOOK_COMMENTS[:max_comments]
                            fetch_success = True
                        else:
                            if platform == "YouTube":
                                if not yt_api_key.strip():
                                    st.sidebar.error("❌ YouTube API Key cannot be empty.")
                                elif not yt_input.strip():
                                    st.sidebar.error("❌ YouTube Video ID or URL is required.")
                                else:
                                    yt_connector = YouTubeConnector(api_key=yt_api_key)
                                    raw_comments = yt_connector.fetch_comments(yt_input, max_comments, yt_order)
                                    fetch_success = True
                            else:
                                if not fb_token.strip():
                                    st.sidebar.error("❌ Facebook Page Access Token cannot be empty.")
                                elif not fb_post_id.strip():
                                    st.sidebar.error("❌ Facebook Post ID or URL is required.")
                                else:
                                    fb_connector = FacebookConnector(page_access_token=fb_token)
                                    raw_comments = fb_connector.fetch_comments(fb_post_id, max_comments)
                                    fetch_success = True
                    except Exception as e:
                        st.error(f"⚠️ Error while fetching comments: {str(e)}")
                        fetch_success = False

                if fetch_success:
                    if not raw_comments:
                        st.warning("No comments found for this source.")
                    else:
                        with st.spinner("Running Khmer NLP tokenization, sentiment engine, and toxicity scanning..."):
                            raw_df = pd.DataFrame(raw_comments)
                            try:
                                analyzed_df = sentiment_engine.analyze_dataframe(
                                    raw_df,
                                    text_column="comment",
                                    mode=classification_mode,
                                    include_moderation=True
                                )
                            except TypeError:
                                analyzed_df = sentiment_engine.analyze_dataframe(
                                    raw_df,
                                    text_column="comment",
                                    mode=classification_mode
                                )
                            if "Is_Flagged" not in analyzed_df.columns:
                                detector = get_toxicity_detector()
                                analyzed_df = detector.scan_dataframe(analyzed_df, text_col="comment")

                            st.session_state.analysis_df = analyzed_df
                            st.success(f"🎉 Successfully ingested and moderated {len(analyzed_df)} comments!")


        st.markdown("---")
        st.caption("Khmer NLP & Content Moderation v2.5")

    # Header Banner
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700;">
            🛡️ Social Media Monitor & Content Moderation Dashboard
        </h1>
        <p style="margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.85;">
            Unified AI Sentiment Analytics, Toxicity Detection & Community Moderation for YouTube & Facebook.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Route based on selected Monitoring Mode
    if "Mode A" in st.session_state.monitoring_mode:
        render_channel_monitor_mode()
    else:
        render_public_explorer_mode()


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if not st.session_state.get("authenticated", False):
    render_authentication_view()
else:
    render_dashboard_view()
