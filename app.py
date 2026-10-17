import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path

from chatbot import ChatbotModel
from response_engine import ResponseEngine
from context_manager import ContextManager
from analytics import AnalyticsTracker
from speech_to_text import SpeechToTextEngine
from text_to_speech import TextToSpeechEngine
from emotion_detector import EmotionDetector
from translator_engine import TranslationEngine
from history_manager import new_session_id, save_session, load_all_sessions, delete_session
import translator

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VocaMind – Voice AI Chatbot",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── INLINE CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #111827 !important;
}

/* ─── Background ─────────────────────────────────────── */
.stApp {
    background: #FFFFFF !important;
}

/* ─── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

/* ─── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #111827 !important;
    font-weight: 600 !important;
}

/* ─── Headers ────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px;
}

/* ─── Buttons ────────────────────────────────────────── */
.stButton > button {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #EFF6FF !important;
    border-color: #3B82F6 !important;
    color: #1D4ED8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.15) !important;
}

/* ─── Form submit ─────────────────────────────────────── */
div[data-testid="stForm"] button[type="submit"] {
    background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
}
div[data-testid="stForm"] button[type="submit"]:hover {
    background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(14,165,233,0.5) !important;
}
div[data-testid="stForm"] button[type="submit"] * {
    color: #FFFFFF !important;
}

/* ─── Inputs & Selects ───────────────────────────────── */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}
div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
div[data-baseweb="select"] { background: #FFFFFF !important; border-radius: 10px !important; border: 1px solid #D1D5DB !important; }
div[data-baseweb="select"] * { color: #111827 !important; background: #FFFFFF !important; }

/* ─── Tabs ───────────────────────────────────────────── */
button[data-baseweb="tab"] {
    color: #6B7280 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}
button[aria-selected="true"] {
    color: #1D4ED8 !important;
    border-bottom: 2px solid #3B82F6 !important;
}

/* ─── Alert/Info ─────────────────────────────────────── */
div[data-testid="stAlert"] {
    background: #F0F9FF !important;
    border: 1px solid #BAE6FD !important;
    border-radius: 12px !important;
}
div[data-testid="stAlert"] * { color: #0C4A6E !important; }

/* ─── Metrics ─────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
div[data-testid="stMetricValue"] { color: #1D4ED8 !important; font-weight: 800 !important; }
div[data-testid="stMetricLabel"] { color: #6B7280 !important; font-weight: 600 !important; }

/* ─── Download buttons ───────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* ─── Radio ──────────────────────────────────────────── */
div[data-testid="stRadio"] label { color: #111827 !important; font-weight: 600 !important; }
div[data-testid="stRadio"] label span { color: #111827 !important; }

/* ─── Custom Classes ──────────────────────────────────── */
.vm-header {
    background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.vm-header::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.05), transparent);
    animation: shimmer 3s infinite;
}
@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.vm-title {
    font-size: 36px; font-weight: 900; letter-spacing: -1px; margin: 0;
    background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 50%, #7C3AED 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.vm-subtitle {
    font-size: 15px; color: #6B7280; font-weight: 600; margin-top: 6px;
}

.topic-card {
    border-radius: 14px;
    padding: 16px 18px;
    cursor: pointer;
    border: 1.5px solid transparent;
    transition: all 0.2s ease;
    margin-bottom: 4px;
    position: relative;
    overflow: hidden;
}
.topic-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}

.chat-user-msg {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 14px 14px 4px 14px;
    padding: 14px 18px;
    margin: 8px 0 8px 40px;
    position: relative;
}
.chat-bot-msg {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 4px 14px 14px 14px;
    padding: 14px 18px;
    margin: 8px 40px 8px 0;
    position: relative;
}
.msg-meta {
    font-size: 11px; color: #6B7280; font-weight: 600; margin-bottom: 6px;
    display: flex; align-items: center; gap: 6px;
}
.msg-text {
    font-size: 15px; color: #111827; font-weight: 500; line-height: 1.6;
}
.msg-badge {
    display: inline-block; font-size: 11px; font-weight: 700;
    padding: 2px 8px; border-radius: 20px;
    background: #DBEAFE; color: #1D4ED8;
    border: 1px solid #BFDBFE;
    margin-left: 6px;
}

.analysis-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px;
}
.analysis-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid #F1F5F9;
}
.analysis-row:last-child { border-bottom: none; }
.analysis-label { font-size: 12px; color: #6B7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.analysis-value { font-size: 14px; color: #111827; font-weight: 700; }

.session-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.session-card:hover {
    border-color: #93C5FD;
    background: #EFF6FF;
}
.session-card-title {
    font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 4px;
}
.session-card-meta {
    font-size: 11px; color: #6B7280;
}

.hint-chip {
    display: inline-block;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    font-size: 12px; font-weight: 600;
    padding: 4px 10px; border-radius: 20px;
    margin: 3px;
}

.stat-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 20px; padding: 6px 12px;
    font-size: 12px; color: #6B7280; font-weight: 600;
}

.divider-glow {
    height: 1px;
    background: linear-gradient(90deg, transparent, #BFDBFE, transparent);
    margin: 20px 0;
}

.empty-chat {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 60px 20px; text-align: center;
}
.empty-chat-icon { font-size: 52px; margin-bottom: 16px; }
.empty-chat-title { font-size: 20px; font-weight: 800; color: #374151; margin-bottom: 8px; }
.empty-chat-sub { font-size: 14px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
LANGUAGE_CONFIG = {
    "English (en-US)": {"code": "en-US", "name": "English", "tts_lang": "en"},
    "Telugu (te-IN)":  {"code": "te-IN", "name": "Telugu",  "tts_lang": "te"},
    "Hindi (hi-IN)":   {"code": "hi-IN", "name": "Hindi",   "tts_lang": "hi"},
    "Tamil (ta-IN)":   {"code": "ta-IN", "name": "Tamil",   "tts_lang": "ta"},
    "Spanish (es-ES)": {"code": "es-ES", "name": "Spanish", "tts_lang": "es"},
    "French (fr-FR)":  {"code": "fr-FR", "name": "French",  "tts_lang": "fr"},
}

TOPICS = {
    "🌐 All Topics": {
        "color": "#38BDF8", "bg": "rgba(14,165,233,0.07)",
        "intents": None,
        "desc": "Ask anything — greetings, orders, support & more.",
        "hints": ["Hello!", "What can you do?", "Tell me about VocaMind"],
    },
    "💬 General Chat": {
        "color": "#A78BFA", "bg": "rgba(167,139,250,0.07)",
        "intents": ["greeting", "goodbye", "thanks", "help", "general_question"],
        "desc": "Greet the bot, say thanks, or learn what it can do.",
        "hints": ["Hello!", "Thank you", "Goodbye", "What can you help with?"],
    },
    "📦 Orders & Delivery": {
        "color": "#FBBF24", "bg": "rgba(251,191,36,0.07)",
        "intents": ["order_status", "cancel_order", "refund"],
        "desc": "Check, cancel, or refund your orders.",
        "hints": ["Where is my order?", "Cancel my order", "I want a refund"],
    },
    "💳 Payments & Pricing": {
        "color": "#34D399", "bg": "rgba(52,211,153,0.07)",
        "intents": ["pricing", "payment", "services"],
        "desc": "Explore plans, payment options, and service details.",
        "hints": ["What are your pricing plans?", "Which payments do you accept?"],
    },
    "🔧 Technical Support": {
        "color": "#F87171", "bg": "rgba(248,113,113,0.07)",
        "intents": ["technical_support", "account_help", "password_reset", "complaint"],
        "desc": "Fix issues, reset passwords, or file a complaint.",
        "hints": ["I have a technical problem", "Reset my password", "I want to complain"],
    },
    "🏢 Company Info": {
        "color": "#60A5FA", "bg": "rgba(96,165,250,0.07)",
        "intents": ["location", "contact", "working_hours"],
        "desc": "Office address, contact details, and business hours.",
        "hints": ["What are your working hours?", "Where are you located?"],
    },
}

CONFIDENCE_THRESHOLD = 0.50

# ── SESSION STATE ─────────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "session_id": new_session_id(),
        "current_topic": "🌐 All Topics",
        "context": ContextManager(),
        "chatbot_model": ChatbotModel(),
        "response_engine": ResponseEngine(),
        "emotion_detector": EmotionDetector(),
        "stt_engine": SpeechToTextEngine(),
        "last_audio_id": None,
        "auto_speak_html": None,
        "trans_source_lang": "Telugu",
        "trans_target_lang": "English",
        "translator_history": [],
        "last_trans_audio_id": None,
        "loaded_session_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;">
        <div style="font-size:22px;font-weight:900;color:#38BDF8;letter-spacing:-0.5px;">🎙️ VOCAMIND</div>
        <div style="font-size:12px;color:#6B7280;font-weight:600;margin-top:2px;">AI Voice Chatbot v3.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    nav_option = st.radio(
        "Navigate",
        ["🏠 AI Chatbot", "🌐 Voice Translator", "📊 Analytics", "🔍 Intent Explorer", "⚙️ Settings", "ℹ️ About"],
        index=0, label_visibility="collapsed"
    )

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:12px;color:#6B7280;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Quick Controls</div>', unsafe_allow_html=True)

    selected_language_name = st.selectbox("Language", list(LANGUAGE_CONFIG.keys()), index=0)
    lang_info  = LANGUAGE_CONFIG[selected_language_name]
    bcp_code   = lang_info["code"]
    lang_display = lang_info["name"]
    tts_lang_code = lang_info["tts_lang"]

    tts_enabled = st.checkbox("Auto-speak replies", value=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 New Chat", use_container_width=True):
            save_session(
                st.session_state.session_id,
                st.session_state.current_topic,
                st.session_state.context.history
            )
            st.session_state.session_id = new_session_id()
            st.session_state.context.clear_history()
            st.session_state.last_audio_id = None
            st.session_state.auto_speak_html = None
            st.rerun()
    with col_b:
        if st.button("💾 Save", use_container_width=True):
            save_session(
                st.session_state.session_id,
                st.session_state.current_topic,
                st.session_state.context.history
            )
            st.success("Saved!")

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    # ─── Past sessions browser ────────────────────────────────
    st.markdown('<div style="font-size:12px;color:#6B7280;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">📂 Chat History</div>', unsafe_allow_html=True)

    all_sessions = load_all_sessions()
    if all_sessions:
        active_sid = st.session_state.session_id
        for sess in all_sessions[:8]:
            sid   = sess.get("session_id", "")
            topic = sess.get("topic", "Unknown")
            saved = sess.get("saved_at", "")[:16].replace("T", " ")
            turns = sess.get("turn_count", 0)
            preview = sess.get("preview", "")[:40]
            is_active = sid == active_sid
            border = "1px solid #3B82F6" if is_active else "1px solid #E2E8F0"
            bg = "rgba(59,130,246,0.06)" if is_active else "#F8FAFC"

            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:10px;padding:10px 12px;margin-bottom:6px;">
                <div style="font-size:12px;font-weight:700;color:#111827;">{topic}</div>
                <div style="font-size:11px;color:#6B7280;margin-top:2px;">{saved} · {turns} turns</div>
                <div style="font-size:11px;color:#6B7280;margin-top:2px;font-style:italic;">"{preview}..."</div>
            </div>
            """, unsafe_allow_html=True)

            btn_cols = st.columns([1, 1])
            with btn_cols[0]:
                if st.button("▶ Load", key=f"load_{sid}", use_container_width=True):
                    save_session(
                        st.session_state.session_id,
                        st.session_state.current_topic,
                        st.session_state.context.history
                    )
                    st.session_state.session_id = sid
                    st.session_state.current_topic = topic
                    ctx = ContextManager()
                    ctx.history = sess.get("history", [])
                    st.session_state.context = ctx
                    st.session_state.auto_speak_html = None
                    st.rerun()
            with btn_cols[1]:
                if st.button("🗑", key=f"del_{sid}", use_container_width=True):
                    delete_session(sid)
                    if sid == st.session_state.session_id:
                        st.session_state.session_id = new_session_id()
                        st.session_state.context.clear_history()
                    st.rerun()
    else:
        st.markdown('<div style="font-size:12px;color:#374151;text-align:center;padding:12px 0;">No saved chats yet.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 1 ── AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
if nav_option == "🏠 AI Chatbot":

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="vm-header">
        <div class="vm-title">🎙️ VocaMind Chatbot</div>
        <div class="vm-subtitle">Deep Learning BiLSTM · Multi-language · Voice + Text · Topic-focused Q&amp;A</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Topic Selector ────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:13px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;">Choose a Topic</div>', unsafe_allow_html=True)

    topic_names = list(TOPICS.keys())
    num_topics  = len(topic_names)
    topic_cols  = st.columns(num_topics)

    for idx, (tname, tdata) in enumerate(TOPICS.items()):
        with topic_cols[idx]:
            is_sel = st.session_state.current_topic == tname
            border_style = f"2px solid {tdata['color']}" if is_sel else "1px solid rgba(51,65,85,0.4)"
            bg_style = tdata["bg"] if is_sel else "rgba(15,23,42,0.4)"
            st.markdown(f"""
            <div style="background:{bg_style};border:{border_style};border-radius:12px;padding:10px 8px;text-align:center;margin-bottom:4px;">
                <div style="font-size:20px;">{tname.split()[0]}</div>
                <div style="font-size:11px;font-weight:700;color:{tdata['color'] if is_sel else '#64748B'};margin-top:4px;line-height:1.3;">{' '.join(tname.split()[1:])}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select" if not is_sel else "✓ Active", key=f"topic_{idx}", use_container_width=True):
                st.session_state.current_topic = tname
                st.rerun()

    cur_topic  = st.session_state.current_topic
    cur_tdata  = TOPICS[cur_topic]

    # Show topic hints
    hints_html = "".join(f'<span class="hint-chip">"{h}"</span>' for h in cur_tdata["hints"])
    st.markdown(f"""
    <div style="background:{cur_tdata['bg']};border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:10px 14px;margin:8px 0 16px;">
        <span style="font-size:12px;color:{cur_tdata['color']};font-weight:700;">{cur_topic}</span>
        <span style="font-size:12px;color:#6B7280;margin-left:8px;">{cur_tdata['desc']}</span>
        <div style="margin-top:8px;">{hints_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

    # ── Main Layout ────────────────────────────────────────────────────────────
    chat_col, info_col = st.columns([1.7, 1], gap="large")

    # ── LEFT: Chat Window ─────────────────────────────────────────────────────
    with chat_col:
        st.markdown(f'<div style="font-size:16px;font-weight:800;color:#111827;margin-bottom:12px;">💬 Conversation — <span style="color:{cur_tdata["color"]}">{cur_topic}</span></div>', unsafe_allow_html=True)

        history = st.session_state.context.history
        chat_area = st.container()

        with chat_area:
            if not history:
                st.markdown(f"""
                <div class="empty-chat">
                    <div class="empty-chat-icon">🤖</div>
                    <div class="empty-chat-title">Ready to chat!</div>
                    <div class="empty-chat-sub">Ask something about <strong>{cur_topic}</strong> in <strong>{lang_display}</strong><br>using voice or text below.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for idx, turn in enumerate(history):
                    icon = "🎤" if turn["input_type"] == "voice" else "⌨️"
                    conf_pct = f"{turn['confidence']*100:.0f}%"

                    # User bubble
                    st.markdown(f"""
                    <div class="chat-user-msg">
                        <div class="msg-meta">
                            <span>{icon} You</span>
                            <span style="color:#6B7280;">{turn['timestamp'][11:16]}</span>
                            <span class="msg-badge">{turn['input_type']}</span>
                        </div>
                        <div class="msg-text">{turn['user_text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Bot bubble
                    st.markdown(f"""
                    <div class="chat-bot-msg">
                        <div class="msg-meta">
                            <span style="color:#34D399;">🤖 VocaMind</span>
                            <span style="color:#6B7280;">{turn['timestamp'][11:16]}</span>
                            <span class="msg-badge" style="color:#34D399;border-color:rgba(52,211,153,0.2);background:rgba(52,211,153,0.08);">{conf_pct} conf</span>
                        </div>
                        <div class="msg-text">{turn['bot_response']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # TTS play button (manual, for all turns)
                    if tts_enabled:
                        tts_html = TextToSpeechEngine.get_tts_audio_html(
                            turn["bot_response"],
                            button_text="🔊 Play",
                            lang=tts_lang_code
                        )
                        components.html(tts_html, height=44)

                    # Feedback
                    fb1, fb2, _ = st.columns([1, 1, 4])
                    with fb1:
                        if st.button("👍", key=f"pos_{idx}", help="Mark as helpful"):
                            st.session_state.context.update_feedback(idx, "Positive")
                    with fb2:
                        if st.button("👎", key=f"neg_{idx}", help="Mark as unhelpful"):
                            st.session_state.context.update_feedback(idx, "Negative")

                    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)

        # Auto-speak the latest response if new (gTTS audio embed)
        if tts_enabled and st.session_state.auto_speak_html:
            components.html(st.session_state.auto_speak_html, height=0)
            st.session_state.auto_speak_html = None

        # ── Input Area ─────────────────────────────────────────────────────────
        st.markdown(f'<div class="divider-glow"></div><div style="font-size:14px;font-weight:700;color:#6B7280;margin:4px 0 10px;">🎤 Input — {lang_display}</div>', unsafe_allow_html=True)

        tab_mic, tab_web, tab_txt = st.tabs([
            f"🎙️ Microphone ({lang_display})",
            "🌐 Live Dictation",
            f"⌨️ Type ({lang_display})",
        ])

        user_input_text  = None
        input_source_type = None

        with tab_mic:
            st.write(f"Speak in **{lang_display}** ({bcp_code}):")
            audio_rec = st.audio_input("Record voice query", key="native_mic")
            if audio_rec is not None:
                audio_bytes = audio_rec.getvalue()
                audio_id    = hash(audio_bytes)
                if audio_id != st.session_state.last_audio_id:
                    st.session_state.last_audio_id = audio_id
                    with st.spinner(f"Transcribing {lang_display}..."):
                        stt_result = st.session_state.stt_engine.transcribe_audio_buffer(
                            audio_rec, language=bcp_code
                        )
                    if stt_result["success"]:
                        user_input_text   = stt_result["text"]
                        input_source_type = "voice"
                        st.success(f'Recognized: "{user_input_text}"')
                    else:
                        st.error(f"Speech error: {stt_result['error']}")

        with tab_web:
            st.write(f"Click **SPEAK NOW** for live dictation in **{lang_display}**:")
            web_html = SpeechToTextEngine.get_web_speech_component_html(
                language_code=bcp_code, lang_display_name=lang_display
            )
            components.html(web_html, height=200)
            manual = st.text_input("Paste recognized text here:", key="manual_speech")
            if st.button("Submit Voice Text 🚀", key="btn_web_speech"):
                if manual.strip():
                    user_input_text   = manual.strip()
                    input_source_type = "voice"

        with tab_txt:
            with st.form("text_form", clear_on_submit=True):
                typed = st.text_input(
                    f"Ask about {cur_topic} in {lang_display}…",
                    placeholder=f"e.g. {cur_tdata['hints'][0]}"
                )
                submitted = st.form_submit_button(f"Send Message 🚀")
                if submitted and typed.strip():
                    user_input_text   = typed.strip()
                    input_source_type = "text"

        # ── Quick hint chips → pre-fill input ──────────────────────────────────
        st.markdown('<div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">Quick questions:</div>', unsafe_allow_html=True)
        hint_cols = st.columns(len(cur_tdata["hints"]))
        for hi, hint in enumerate(cur_tdata["hints"]):
            with hint_cols[hi]:
                if st.button(hint, key=f"hint_{hi}_{cur_topic}", use_container_width=True):
                    user_input_text   = hint
                    input_source_type = "text"

        # ── Process through deep learning pipeline ────────────────────────────
        if user_input_text:
            with st.spinner("🧠 Thinking..."):
                english_text = TranslationEngine.translate_to_english(user_input_text, bcp_code)
                prediction   = st.session_state.chatbot_model.predict_intent(
                    english_text, threshold=CONFIDENCE_THRESHOLD
                )
                intent      = prediction["intent"]
                confidence  = prediction["confidence"]
                is_confident = prediction["is_confident"]

                # Topic filter: if topic has specific intents, check alignment
                topic_intents = cur_tdata.get("intents")
                if topic_intents is not None and prediction.get("raw_intent") not in topic_intents and is_confident:
                    is_confident = False
                    intent = prediction.get("raw_intent", intent)

                english_response = st.session_state.response_engine.get_response(
                    intent=prediction.get("raw_intent", intent),
                    confidence=confidence,
                    is_confident=prediction["is_confident"],
                )
                final_response = TranslationEngine.translate_from_english(english_response, bcp_code)
                emotion_info   = st.session_state.emotion_detector.predict_emotion(english_text)

            st.session_state.context.add_turn(
                input_type=input_source_type,
                user_text=user_input_text,
                bot_response=final_response,
                intent=prediction.get("raw_intent", intent),
                confidence=confidence,
                emotion=f"{emotion_info['emoji']} {emotion_info['emotion']}",
            )

            # Build auto-play audio
            if tts_enabled:
                b64 = TextToSpeechEngine.generate_audio_base64(final_response, lang=tts_lang_code)
                if b64:
                    st.session_state.auto_speak_html = (
                        f'<audio autoplay src="data:audio/mp3;base64,{b64}" '
                        f'style="display:none;"></audio>'
                    )

            # Auto-save after each turn
            save_session(
                st.session_state.session_id,
                st.session_state.current_topic,
                st.session_state.context.history,
            )
            st.rerun()

    # ── RIGHT: Analysis + Export ───────────────────────────────────────────────
    with info_col:
        st.markdown('<div style="font-size:16px;font-weight:800;color:#111827;margin-bottom:12px;">🧠 AI Analysis</div>', unsafe_allow_html=True)

        if history:
            last = history[-1]
            conf_color = "#34D399" if last["confidence"] >= CONFIDENCE_THRESHOLD else "#F87171"
            conf_pct   = f"{last['confidence']*100:.1f}%"

            st.markdown(f"""
            <div class="analysis-panel">
                <div style="font-size:13px;color:#6B7280;font-weight:700;margin-bottom:12px;">LAST RESPONSE BREAKDOWN</div>

                <div class="analysis-row">
                    <div class="analysis-label">Input</div>
                    <div class="analysis-value" style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">"{last['user_text'][:30]}…"</div>
                </div>
                <div class="analysis-row">
                    <div class="analysis-label">Intent</div>
                    <div class="analysis-value" style="color:#38BDF8;">{last['intent'].replace('_',' ').title()}</div>
                </div>
                <div class="analysis-row">
                    <div class="analysis-label">Confidence</div>
                    <div class="analysis-value" style="color:{conf_color};">{conf_pct}</div>
                </div>
                <div class="analysis-row">
                    <div class="analysis-label">Emotion</div>
                    <div class="analysis-value">{last['emotion']}</div>
                </div>
                <div class="analysis-row">
                    <div class="analysis-label">Input Type</div>
                    <div class="analysis-value">{last['input_type'].title()}</div>
                </div>
                <div class="analysis-row">
                    <div class="analysis-label">Topic</div>
                    <div class="analysis-value" style="color:{cur_tdata['color']};">{cur_topic}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown(f"""
            <div style="margin-top:14px;">
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#6B7280;font-weight:700;margin-bottom:4px;">
                    <span>Model Confidence</span><span>{conf_pct}</span>
                </div>
                <div style="background:rgba(51,65,85,0.5);border-radius:20px;height:8px;overflow:hidden;">
                    <div style="width:{conf_pct};height:100%;background:linear-gradient(90deg,{conf_color},{conf_color}88);border-radius:20px;transition:width 0.5s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if last["confidence"] < CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ Low confidence ({conf_pct}) — fallback triggered.")
            else:
                st.success("✅ High-confidence match.")
        else:
            st.markdown("""
            <div class="analysis-panel" style="text-align:center;padding:30px 20px;">
                <div style="font-size:32px;margin-bottom:10px;">📊</div>
                <div style="font-size:14px;color:#374151;font-weight:600;">Analysis appears here<br>after your first message.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Session Stats ─────────────────────────────────────────────────────
        st.markdown('<div class="divider-glow" style="margin:16px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:700;color:#6B7280;margin-bottom:10px;">SESSION STATS</div>', unsafe_allow_html=True)

        total = len(history)
        voice_cnt = sum(1 for t in history if t["input_type"] == "voice")
        avg_conf  = (sum(t["confidence"] for t in history) / total * 100) if total else 0

        m1, m2 = st.columns(2)
        m1.metric("Messages", total)
        m2.metric("Avg Confidence", f"{avg_conf:.0f}%")
        m3, m4 = st.columns(2)
        m3.metric("Voice Inputs", voice_cnt)
        m4.metric("Text Inputs", total - voice_cnt)

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown('<div class="divider-glow" style="margin:16px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;font-weight:700;color:#6B7280;margin-bottom:10px;">EXPORT TRANSCRIPT</div>', unsafe_allow_html=True)

        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "📄 TXT", data=st.session_state.context.export_as_txt(),
                file_name="vocamind_chat.txt", mime="text/plain",
                use_container_width=True,
            )
        with e2:
            st.download_button(
                "📊 JSON", data=st.session_state.context.export_as_json(),
                file_name="vocamind_chat.json", mime="application/json",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 2 ── VOICE TRANSLATOR
# ══════════════════════════════════════════════════════════════════════════════
elif nav_option == "🌐 Voice Translator":
    st.markdown("""
    <div class="vm-header">
        <div class="vm-title">🌐 Voice Translator</div>
        <div class="vm-subtitle">Speech Recognition → Translation → Voice Synthesis across 4 languages</div>
    </div>
    """, unsafe_allow_html=True)

    supported_langs = list(translator.SUPPORTED_LANGUAGES.keys())

    col_src, col_swap, col_tgt = st.columns([2, 0.6, 2], gap="small")
    with col_src:
        src_lang = st.selectbox(
            "Source Language", supported_langs,
            index=supported_langs.index(st.session_state.trans_source_lang)
            if st.session_state.trans_source_lang in supported_langs else 3,
            key="sel_src",
        )
        st.session_state.trans_source_lang = src_lang

    with col_swap:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("⇄ Swap"):
            st.session_state.trans_source_lang, st.session_state.trans_target_lang = (
                st.session_state.trans_target_lang, st.session_state.trans_source_lang
            )
            st.rerun()

    with col_tgt:
        tgt_lang = st.selectbox(
            "Target Language", supported_langs,
            index=supported_langs.index(st.session_state.trans_target_lang)
            if st.session_state.trans_target_lang in supported_langs else 0,
            key="sel_tgt",
        )
        st.session_state.trans_target_lang = tgt_lang

    if src_lang == tgt_lang:
        st.warning("⚠️ Source and target languages must differ.")
    else:
        src_info = translator.SUPPORTED_LANGUAGES[src_lang]
        tgt_info = translator.SUPPORTED_LANGUAGES[tgt_lang]
        st.markdown(f"""
        <div style="text-align:center;margin:14px 0;font-size:20px;font-weight:800;color:#38BDF8;">
            {src_info['flag']} {src_lang} &nbsp;→&nbsp; {tgt_info['flag']} {tgt_lang}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)
        st.subheader(f"🎤 Record in {src_info['flag']} {src_lang}")
        audio_rec = st.audio_input(f"Speak in {src_lang}", key="trans_mic")

        if audio_rec is not None:
            ab   = audio_rec.getvalue()
            aid  = hash(ab)
            if aid != st.session_state.last_trans_audio_id:
                st.session_state.last_trans_audio_id = aid
                with st.spinner(f"Transcribing {src_lang}..."):
                    stt_res = st.session_state.stt_engine.transcribe_audio_buffer(
                        audio_rec, language=src_info["stt"]
                    )
                if not stt_res["success"]:
                    st.error(f"Speech error: {stt_res['error']}")
                else:
                    recog = stt_res["text"]
                    with st.spinner(f"Translating {src_lang} → {tgt_lang}..."):
                        trans_res = translator.translate(recog, src_lang, tgt_lang)
                    if not trans_res["success"]:
                        st.error(f"Translation error: {trans_res['error']}")
                    else:
                        entry = {
                            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                            "src_lang": src_lang, "tgt_lang": tgt_lang,
                            "src_flag": src_info["flag"], "tgt_flag": tgt_info["flag"],
                            "audio_bytes": ab,
                            "recognized_text": recog,
                            "translated_text": trans_res["translated_text"],
                            "tgt_tts_code": tgt_info["tts"],
                        }
                        st.session_state.translator_history.insert(0, entry)
                        st.rerun()

    if st.session_state.translator_history:
        st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)
        st.subheader("📋 Translation Results")
        for entry in st.session_state.translator_history:
            st.markdown(f"""
            <div style="background:rgba(248,250,252,0.95);border:1px solid #BFDBFE;border-radius:14px;padding:14px 18px;margin-bottom:14px;">
                <div style="font-size:13px;font-weight:700;color:#38BDF8;">[{entry['timestamp']}] {entry['src_flag']} {entry['src_lang']} → {entry['tgt_flag']} {entry['tgt_lang']}</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🎙️ Original**")
                st.audio(entry["audio_bytes"], format="audio/wav")
                st.info(f'"{entry["recognized_text"]}"')
            with c2:
                st.markdown("**🌐 Translation**")
                st.success(f'"{entry["translated_text"]}"')
                tts_html = TextToSpeechEngine.get_tts_audio_html(
                    entry["translated_text"],
                    button_text=f"▶ Play ({entry['tgt_lang']})",
                    lang=entry["tgt_tts_code"],
                )
                components.html(tts_html, height=50)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 3 ── ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif nav_option == "📊 Analytics":
    st.markdown("""
    <div class="vm-header">
        <div class="vm-title">📊 Analytics Dashboard</div>
        <div class="vm-subtitle">Session statistics, intent distribution, emotion tracking</div>
    </div>
    """, unsafe_allow_html=True)

    stats = AnalyticsTracker.compute_analytics(st.session_state.context.history)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Messages",   stats["total_messages"])
    k2.metric("Voice Ratio",      f"{stats['voice_ratio']}%",   f"{stats['voice_messages']} voice")
    k3.metric("Avg Confidence",   f"{stats['avg_confidence']}%")
    k4.metric("Fallback Queries", stats["low_confidence_queries"])

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🎯 Intent Frequency")
        if stats["intent_counts"]:
            df = pd.DataFrame(list(stats["intent_counts"].items()), columns=["Intent", "Count"]).sort_values("Count", ascending=False)
            st.bar_chart(df.set_index("Intent"))
        else:
            st.info("No data yet.")
    with col_b:
        st.subheader("😊 Emotion Distribution")
        if stats["emotion_counts"]:
            df_e = pd.DataFrame(list(stats["emotion_counts"].items()), columns=["Emotion", "Count"])
            st.dataframe(df_e, use_container_width=True)
        else:
            st.info("No data yet.")

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)
    st.subheader("👍 Feedback")
    fb1, fb2 = st.columns(2)
    fb1.metric("Positive", stats["positive_feedback"])
    fb2.metric("Negative", stats["negative_feedback"])


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 4 ── INTENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif nav_option == "🔍 Intent Explorer":
    st.markdown("""
    <div class="vm-header">
        <div class="vm-title">🔍 Intent Explorer</div>
        <div class="vm-subtitle">Browse all 18 trained intent categories, examples & responses</div>
    </div>
    """, unsafe_allow_html=True)

    dataset_path  = os.path.join(BASE_DIR, "dataset", "intents.json")
    responses_path = os.path.join(BASE_DIR, "data", "responses.json")

    if os.path.exists(dataset_path) and os.path.exists(responses_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            intents_data = json.load(f).get("intents", [])
        with open(responses_path, "r", encoding="utf-8") as f:
            responses_data = json.load(f).get("responses", {})

        intent_names = [item["intent"] for item in intents_data]
        sel = st.selectbox("Select Intent Category:", intent_names)
        intent_obj = next((i for i in intents_data if i["intent"] == sel), None)
        intent_resps = responses_data.get(sel, [])

        if intent_obj:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("💬 Training Examples")
                for ex in intent_obj["examples"]:
                    st.markdown(f'<div style="background:rgba(14,165,233,0.06);border:1px solid rgba(14,165,233,0.15);border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:14px;color:#374151;">"{ex}"</div>', unsafe_allow_html=True)
            with c2:
                st.subheader("🤖 Bot Responses")
                for resp in intent_resps:
                    st.markdown(f'<div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);border-radius:8px;padding:8px 12px;margin-bottom:6px;font-size:14px;color:#374151;">"{resp}"</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 5 ── SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif nav_option == "⚙️ Settings":
    st.markdown("""
    <div class="vm-header">
        <div class="vm-title">⚙️ System Settings</div>
        <div class="vm-subtitle">Model architecture, hyperparameters, and capability overview</div>
    </div>
    """, unsafe_allow_html=True)

    metadata = st.session_state.chatbot_model.metadata or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Architecture", "BiLSTM")
    m2.metric("Vocab Size",   metadata.get("vocab_size", 234))
    m3.metric("Max Seq Len",  metadata.get("max_len", 8))
    m4.metric("Total Intents", metadata.get("num_classes", 18))

    st.markdown('<div class="divider-glow"></div>', unsafe_allow_html=True)
    st.subheader("🏷️ Trained Intent Classes")
    classes_list = metadata.get("classes", [
        "account_help","cancel_order","complaint","contact","general_question",
        "goodbye","greeting","help","location","order_status","password_reset",
        "payment","pricing","refund","services","technical_support","thanks","working_hours"
    ])
    badge_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;'>"
    for cls in classes_list:
        badge_html += f"<span style='background:rgba(56,189,248,0.08);border:1px solid #93C5FD;color:#38BDF8;font-size:12px;font-weight:700;padding:4px 10px;border-radius:20px;'>🏷️ {cls.replace('_',' ')}</span>"
    badge_html += "</div>"
    st.markdown(badge_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 6 ── ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif nav_option == "ℹ️ About":
    st.markdown("""
    <div class="vm-header">
        <div class="vm-title">ℹ️ About VocaMind</div>
        <div class="vm-subtitle">Academic AI Project — Deep Learning · NLP · Voice Synthesis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 📌 Overview
    **VocaMind** is an end-to-end Voice-Enabled AI Chatbot with:
    - **Topic-based Q&A** — choose a subject area before asking
    - **BiLSTM Deep Learning** — intent classification with 18 categories
    - **Text + Voice replies** — every answer is spoken aloud automatically
    - **Persistent chat history** — sessions saved to disk, resumable anytime
    - **4-Language Translation** — English, Hindi, Tamil, Telugu voice-to-voice

    ### 🏗️ Architecture
    ```
    USER INPUT (Voice / Text)
         ↓
    STT (Speech Recognition)  →  Translation to English
         ↓
    BiLSTM Intent Classification
         ↓
    Response Selection  →  Translation to Target Language
         ↓
    Text Display  +  gTTS Audio Auto-play
         ↓
    Session saved to history/
    ```

    ### 🚀 Deploy to Streamlit Cloud
    1. Push code to GitHub
    2. Sign in at [share.streamlit.io](https://share.streamlit.io)
    3. Set main file to `app.py` and click Deploy
    """)
