import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from transformers import pipeline
import pandas as pd
from datetime import datetime
import os
import re
import requests

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="🎓",
    layout="centered"
)

# =========================================================
# SHARED HELPERS (used by both modes)
# =========================================================
EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😔", "anger": "😠", "fear": "😨",
    "disgust": "🤢", "surprise": "😲", "neutral": "😐",
    "bored": "🥱", "confident": "💪", "confused": "😵", "curious": "🤔", "frustrated": "😤"
}

def normalize(emotion):
    mapping = {
        "happy": "joy", "sad": "sadness", "angry": "anger",
        "fear": "fear", "disgust": "disgust", "surprise": "surprise",
        "neutral": "neutral", "joy": "joy", "sadness": "sadness", "anger": "anger"
    }
    return mapping.get(emotion.lower(), "neutral")

def log_session(mode, face_emotion, text_emotion, final_emotion, user_text, redact=False):
    stored_text = "[redacted - flagged content]" if redact else user_text
    df = pd.DataFrame([[datetime.now(), mode, face_emotion, text_emotion, final_emotion, stored_text]],
                       columns=["timestamp", "mode", "face", "text_emotion", "final", "input_text"])
    file_exists = os.path.isfile("emotion_log.csv")
    df.to_csv("emotion_log.csv", mode='a', header=not file_exists, index=False)

# =========================================================
# MODE 1: WEBCAM + TEXT (existing pipeline)
# =========================================================
RESPONSES = {
    "sadness": "It's okay to feel stuck sometimes. Want to try an easier example first?",
    "anger": "Frustration means you're pushing your limits. Take a 2-min breather and come back sharper.",
    "fear": "This looks harder than it is. Let's break it into smaller steps.",
    "joy": "You're on a roll! Keep this momentum going.",
    "surprise": "New concept, huh? Totally normal to pause and process.",
    "disgust": "Not vibing with this topic? Let's try a different explanation style.",
    "neutral": "Steady progress. Let me know if you want a quick recap."
}

EMOTION_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EMOTION_DISPLAY_LABEL = {
    "anger": "Anger", "disgust": "Disgust", "fear": "Fear", "joy": "Happiness",
    "sadness": "Sadness", "surprise": "Surprise", "neutral": "Neutral"
}

def fuse_emotions(face_emotion, text_emotion):
    face_n = normalize(face_emotion)
    text_n = normalize(text_emotion)
    return face_n if face_n == text_n else text_n

def get_support_message(emotion):
    return RESPONSES.get(emotion, RESPONSES["neutral"])

def draw_viewfinder(img_path, region):
    img = cv2.imread(img_path)
    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
    pad = int(0.12 * max(w, h))
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = w + 2 * pad, h + 2 * pad
    color = (140, 255, 80)
    thickness = 4
    corner_len = int(min(w, h) * 0.22)
    corners = [
        [(x, y + corner_len), (x, y), (x + corner_len, y)],
        [(x + w - corner_len, y), (x + w, y), (x + w, y + corner_len)],
        [(x, y + h - corner_len), (x, y + h), (x + corner_len, y + h)],
        [(x + w - corner_len, y + h), (x + w, y + h), (x + w, y + h - corner_len)],
    ]
    for pts in corners:
        cv2.polylines(img, [np.array(pts)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def render_emotion_bars(emotion_scores):
    # No leading whitespace on these lines on purpose — Streamlit markdown
    # treats 4+ leading spaces as a code block, which breaks HTML rendering.
    rows = ""
    for key in EMOTION_ORDER:
        deepface_key = {"joy": "happy", "sadness": "sad", "anger": "angry"}.get(key, key)
        pct = emotion_scores.get(deepface_key, 0)
        rows += (
            '<div class="ebar-row">'
            f'<div class="ebar-label">{EMOTION_DISPLAY_LABEL[key]}</div>'
            f'<div class="ebar-track"><div class="ebar-fill" style="width:{pct}%;"></div></div>'
            '</div>'
        )
    return f'<div class="ebar-panel">{rows}</div>'

# =========================================================
# MODE 2: STUDY CHALLENGE (new — rule-based + transformer + Gemini)
# =========================================================
STUDY_EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

KEYWORD_MAP = {
    "Bored":       ["bored", "boring", "tedious", "dull", "uninteresting", "monotonous", "sleepy"],
    "Confident":   ["confident", "easy", "got this", "understand", "clear", "makes sense", "comfortable", "know this"],
    "Confused":    ["confused", "don't understand", "dont understand", "lost", "unclear", "not sure", "stuck",
                     "no idea", "makes no sense", "what does this mean"],
    "Curious":     ["curious", "interesting", "wonder", "want to know", "intrigued", "fascinating", "how does"],
    "Frustrated":  ["frustrated", "annoying", "give up", "hate this", "ugh", "can't get", "cant get",
                     "so hard", "keeps failing", "ridiculous", "ffs"]
}

def rule_based_scores(text):
    """Simple keyword-count model — the 'lightweight' comparison model.
    Not a trained BiLSTM: this is an honest, fast substitute (keyword scoring),
    used here because a labeled dataset for these 5 custom categories doesn't
    exist off-the-shelf within a short build window."""
    text_lower = text.lower()
    raw_scores = {}
    for emotion, keywords in KEYWORD_MAP.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        raw_scores[emotion] = count
    total = sum(raw_scores.values())
    if total == 0:
        # no keyword hits — spread evenly rather than falsely picking one
        return {e: 100 / len(STUDY_EMOTIONS) for e in STUDY_EMOTIONS}
    return {e: round((c / total) * 100, 1) for e, c in raw_scores.items()}

@st.cache_resource
def load_zero_shot_classifier():
    # Stands in for the "BERT-family deep learning model" in the comparison.
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

def transformer_scores(text, classifier):
    result = classifier(text, candidate_labels=STUDY_EMOTIONS)
    return {label: round(score * 100, 1) for label, score in zip(result['labels'], result['scores'])}

def enhanced_fusion(rule_scores, trans_scores, boost=12):
    """Rule-based keyword hits enhance (boost) the transformer's scores,
    matching the spec's 'rule-based keyword enhancement' step."""
    text_lower_scores = dict(trans_scores)
    for emotion in STUDY_EMOTIONS:
        if rule_scores.get(emotion, 0) > (100 / len(STUDY_EMOTIONS)):  # had real keyword signal
            text_lower_scores[emotion] = text_lower_scores.get(emotion, 0) + boost
    total = sum(text_lower_scores.values())
    return {e: round((v / total) * 100, 1) for e, v in text_lower_scores.items()}

def detect_mixed_emotion(final_scores, gap_threshold=10):
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = ranked[0], ranked[1]
    if (top1[1] - top2[1]) <= gap_threshold:
        return f"{top1[0]} + {top2[0]}", [top1[0], top2[0]], top1[1]
    return top1[0], [top1[0]], top1[1]

# ---- Safety filter: keeps this a study-support tool, not a crisis tool ----
SAFETY_KEYWORDS = [
    "kill myself", "suicide", "end my life", "want to die", "not worth living",
    "hurt myself", "self harm", "self-harm", "give up on everything", "no reason to live"
]

def check_safety_concern(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in SAFETY_KEYWORDS)

SAFETY_RESPONSE = (
    "It sounds like you might be dealing with something heavier than a study challenge right now. "
    "This tool isn't equipped to help with that, but please reach out to someone you trust — a friend, "
    "family member, or counselor — or a local crisis helpline if things feel urgent. You deserve real support, not just an app."
)

# ---- Canned fallback messages when the Gemini toggle is off ----
STUDY_RESPONSES = {
    "Bored": "Try switching up the format — a video, a quick quiz, or teaching the concept to someone else can re-engage you.",
    "Confident": "You've got a solid handle on this — a good time to try a harder problem or help someone else with it.",
    "Confused": "Confusion is normal at this stage. Try breaking the concept into the smallest possible piece and starting there.",
    "Curious": "That curiosity is worth following — look up one related example to go a bit deeper.",
    "Frustrated": "This is a tough spot. Step away for a few minutes, then come back to just one small part of the problem."
}

def get_study_fallback(label):
    primary = label.split(" + ")[0]
    return STUDY_RESPONSES.get(primary, "Keep going — steady effort adds up even when it doesn't feel like it.")

def call_gemini(user_text, emotion_label):
    # Check Streamlit's secrets.toml first (used for local dev), then fall
    # back to a plain environment variable (how Hugging Face Spaces exposes
    # Repository Secrets). st.secrets raises an exception entirely if no
    # secrets.toml file exists at all, so it must be wrapped in try/except
    # rather than just using .get().
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ No Gemini API key found — add one as a Streamlit secret (local) or a Repository Secret (Hugging Face Spaces) to enable live AI guidance."

    prompt = (
        f"A student describes their study challenge as: \"{user_text}\"\n"
        f"Their detected emotional state is: {emotion_label}.\n"
        "In 2-3 short, warm, encouraging sentences, give them a practical next step "
        "and a supportive note. Keep it concise and specific to what they said."
    )
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=20
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"⚠️ Couldn't reach Gemini right now ({e}). Here's a fallback tip: break the topic into one small piece and tackle just that."

def render_study_bars(scores_dict, color_start, color_end):
    rows = ""
    for emotion in STUDY_EMOTIONS:
        pct = scores_dict.get(emotion, 0)
        rows += (
            '<div class="ebar-row">'
            f'<div class="ebar-label">{emotion}</div>'
            f'<div class="ebar-track"><div class="ebar-fill" style="width:{pct}%; '
            f'background:linear-gradient(90deg,{color_start},{color_end});"></div></div>'
            f'<div class="ebar-pct">{pct}%</div>'
            '</div>'
        )
    return f'<div class="ebar-panel">{rows}</div>'

# =========================================================
# MODELS (loaded once, shared)
# =========================================================
@st.cache_resource
def load_text_classifier():
    return pipeline("text-classification",
                     model="j-hartmann/emotion-english-distilroberta-base",
                     top_k=1)

text_classifier = load_text_classifier()
zero_shot_classifier = load_zero_shot_classifier()

# =========================================================
# GLOBAL STYLE: animated space background
# =========================================================
st.markdown("""
<style>
    .stApp { background: #05060f; }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -2;
        background-color: #05060f;
        background-image:
            radial-gradient(1.6px 1.6px at 8% 15%,  #ffffff 100%, transparent 100%),
            radial-gradient(1.2px 1.2px at 22% 38%, #ffffff 100%, transparent 100%),
            radial-gradient(1.8px 1.8px at 35% 8%,  #ffffff 100%, transparent 100%),
            radial-gradient(1.2px 1.2px at 48% 55%, #ffffff 100%, transparent 100%),
            radial-gradient(1.5px 1.5px at 60% 20%, #ffffff 100%, transparent 100%),
            radial-gradient(1.2px 1.2px at 72% 65%, #ffffff 100%, transparent 100%),
            radial-gradient(1.8px 1.8px at 85% 30%, #ffffff 100%, transparent 100%),
            radial-gradient(1.3px 1.3px at 92% 78%, #ffffff 100%, transparent 100%),
            radial-gradient(1.4px 1.4px at 15% 82%, #ffffff 100%, transparent 100%),
            radial-gradient(1.6px 1.6px at 65% 90%, #ffffff 100%, transparent 100%),
            radial-gradient(1.2px 1.2px at 5% 60%,  #ffffff 100%, transparent 100%),
            radial-gradient(1.5px 1.5px at 78% 5%,  #ffffff 100%, transparent 100%);
        background-repeat: repeat;
        background-size: 340px 340px;
        animation: twinkle 4s ease-in-out infinite alternate;
    }
    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -1;
        background:
            radial-gradient(circle at 20% 25%, rgba(120, 90, 255, 0.28), transparent 40%),
            radial-gradient(circle at 80% 20%, rgba(0, 200, 255, 0.16), transparent 45%),
            radial-gradient(circle at 60% 85%, rgba(255, 80, 160, 0.16), transparent 42%),
            radial-gradient(circle at 15% 80%, rgba(60, 255, 190, 0.12), transparent 40%);
        background-size: 180% 180%;
        filter: blur(50px);
        animation: nebulaDrift 35s ease infinite;
    }
    @keyframes twinkle { 0% { opacity: 0.55; } 100% { opacity: 1; } }
    @keyframes nebulaDrift { 0% { background-position: 0% 0%; } 50% { background-position: 100% 100%; } 100% { background-position: 0% 0%; } }
    @media (prefers-reduced-motion: reduce) { .stApp::before, .stApp::after { animation: none; } }

    .block-container { padding-top: 2rem; max-width: 780px; }
    h1, h2, h3, p, span, div, label { color: #EAEAF5; }

    .app-header { text-align: center; margin-bottom: 0.3rem; }
    .app-header h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.2rem; color: #ffffff; }
    .app-subtitle { text-align: center; font-size: 1rem; margin-bottom: 1.5rem; color: #A9A9C4; }
    .step-label { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem; color: #C7C7E3; }

    div[data-testid="stCameraInput"], div[data-testid="stTextArea"] {
        background-color: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 10px;
        border: 1px solid rgba(140,255,180,0.18);
    }

    .scan-card, .study-card {
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(140,255,180,0.18);
        border-radius: 18px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .scan-title {
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;
        color: #8CF0B6; margin-bottom: 0.6rem; text-align: center;
    }

    .ebar-panel { padding: 0.4rem 0.2rem; }
    .ebar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .ebar-label { width: 90px; font-size: 0.85rem; color: #D5D5EA; flex-shrink: 0; }
    .ebar-pct { width: 42px; font-size: 0.8rem; color: #A9A9C4; text-align: right; flex-shrink: 0; }
    .ebar-track { flex: 1; height: 10px; background-color: rgba(255,255,255,0.08); border-radius: 6px; overflow: hidden; }
    .ebar-fill { height: 100%; background: linear-gradient(90deg, #3fd67a, #8CF0B6); border-radius: 6px; transition: width 0.6s ease; }

    .result-card { border-radius: 18px; padding: 2rem 1.5rem; margin: 1.5rem 0 1rem 0; text-align: center; border: 1px solid; backdrop-filter: blur(6px); }
    .result-emoji { font-size: 3.2rem; line-height: 1; margin-bottom: 0.4rem; }
    .result-label { font-size: 1.6rem; font-weight: 700; text-transform: capitalize; margin-bottom: 1.1rem; }

    .signal-row { display: flex; justify-content: center; gap: 14px; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .signal-chip { background-color: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 12px; padding: 0.6rem 1.1rem; min-width: 130px; }
    .signal-chip-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.6px; color: #A9A9C4; margin-bottom: 0.2rem; }
    .signal-chip-value { font-size: 1.05rem; font-weight: 600; text-transform: capitalize; color: #ffffff; }

    .support-message { background-color: rgba(255,255,255,0.07); border-radius: 12px; padding: 1rem 1.3rem; font-size: 1.05rem; line-height: 1.5; color: #ffffff; }
    .empty-hint { text-align: center; color: #A9A9C4; padding: 1rem; font-size: 0.95rem; }

    .card-joy      { background-color: #1f4d33; border-color: #2f7a52; color: #8CF0B6; }
    .card-sadness  { background-color: #1c3a5c; border-color: #2c5c8f; color: #8FC2F5; }
    .card-anger    { background-color: #5c231c; border-color: #8f382c; color: #F5988F; }
    .card-fear     { background-color: #3a1f5c; border-color: #5c318f; color: #C9A6F5; }
    .card-disgust  { background-color: #33401f; border-color: #516b2f; color: #C3E58C; }
    .card-surprise { background-color: #5c4a1c; border-color: #8f722c; color: #F5D98F; }
    .card-neutral  { background-color: #333333; border-color: #555555; color: #D9D9D9; }
    .card-study    { background-color: #1f2d4d; border-color: #2f4a7a; color: #8CB6F0; }

    .model-badge {
        display: inline-block; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.6px;
        padding: 3px 10px; border-radius: 20px; margin-bottom: 0.6rem;
    }
    .badge-rule { background-color: rgba(255,200,80,0.15); color: #FFD98F; border: 1px solid rgba(255,200,80,0.3); }
    .badge-bert { background-color: rgba(140,182,255,0.15); color: #8CB6FF; border: 1px solid rgba(140,182,255,0.3); }

    .gemini-panel {
        background: linear-gradient(135deg, rgba(120,90,255,0.12), rgba(0,200,255,0.08));
        border: 1px solid rgba(150,130,255,0.3);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-top: 1rem;
        font-size: 1rem;
        line-height: 1.55;
    }
    .gemini-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; color: #C9BFFF; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="app-header"><h1>🎓 Emotion Detection & Learning Support Engine</h1></div>
<div class="app-subtitle">Reads how you're feeling — from your face, your words, or both — and responds with the right support.</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Webcam + Text", "📝 Study Challenge (AI-guided)"])

# =========================================================
# TAB 1 — original webcam + text mode
# =========================================================
with tab1:
    with st.form(key="mood_form"):
        st.markdown('<div class="step-label">📸 Step 1 — Show your face</div>', unsafe_allow_html=True)
        img_file = st.camera_input(" ", label_visibility="collapsed", key="cam1")

        st.markdown('<div class="step-label">💬 Step 2 — Tell me how you feel</div>', unsafe_allow_html=True)
        user_text = st.text_area(" ", placeholder="e.g. I don't get this at all, it's frustrating...",
                                  label_visibility="collapsed", height=120, key="text1")

        analyze_clicked = st.form_submit_button("🔍 Analyze my mood", type="primary")

    if analyze_clicked:
        if not img_file or not user_text.strip():
            st.markdown('<div class="empty-hint">I need both a photo and a bit of text to give you a read 🙂</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Scanning..."):
                with open("temp.jpg", "wb") as f:
                    f.write(img_file.getbuffer())

                emotion_scores, annotated_img = None, None
                try:
                    face_result = DeepFace.analyze(img_path="temp.jpg", actions=['emotion'], enforce_detection=False)
                    face_emotion = face_result[0]['dominant_emotion']
                    emotion_scores = face_result[0]['emotion']
                    region = face_result[0].get('region', {})
                    if region.get('w', 0) > 0:
                        annotated_img = draw_viewfinder("temp.jpg", region)
                except Exception:
                    face_emotion = "neutral"
                    st.warning("Couldn't clearly detect a face — using neutral as fallback.")

                text_result = text_classifier(user_text)[0][0]['label']
                final_emotion = fuse_emotions(face_emotion, text_result)

            if annotated_img is not None and emotion_scores is not None:
                st.markdown('<div class="scan-card">', unsafe_allow_html=True)
                st.markdown('<div class="scan-title">Facial analysis</div>', unsafe_allow_html=True)
                col_img, col_bars = st.columns([1, 1])
                with col_img:
                    st.image(annotated_img, use_container_width=True)
                with col_bars:
                    st.markdown(render_emotion_bars(emotion_scores), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            emoji = EMOTION_EMOJI.get(final_emotion, "😐")
            face_emoji = EMOTION_EMOJI.get(normalize(face_emotion), "😐")
            text_emoji = EMOTION_EMOJI.get(normalize(text_result), "😐")

            st.markdown(f"""
            <div class="result-card card-{final_emotion}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{final_emotion}</div>
                <div class="signal-row">
                    <div class="signal-chip"><div class="signal-chip-title">Face says</div><div class="signal-chip-value">{face_emoji} {face_emotion}</div></div>
                    <div class="signal-chip"><div class="signal-chip-title">Text says</div><div class="signal-chip-value">{text_emoji} {text_result}</div></div>
                </div>
                <div class="support-message">{get_support_message(final_emotion)}</div>
            </div>
            """, unsafe_allow_html=True)

            log_session("webcam_text", face_emotion, text_result, final_emotion, user_text)

# =========================================================
# TAB 2 — Study Challenge mode
# =========================================================
with tab2:
    with st.form(key="challenge_form"):
        st.markdown('<div class="step-label">💬 Describe what you\'re studying and what\'s hard about it</div>', unsafe_allow_html=True)
        challenge_text = st.text_area(" ", placeholder="e.g. I'm lost on recursion, I don't understand how the function calls itself...",
                                       label_visibility="collapsed", height=130, key="text2")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            show_comparison = st.toggle("🔍 Show model comparison", value=True)
        with col_t2:
            use_gemini = st.toggle("✨ Use Gemini for guidance", value=True,
                                    help="If off, a quick canned tip is shown instead of calling the Gemini API")

        analyze_challenge_clicked = st.form_submit_button("Analyze my challenge", type="primary")

    if analyze_challenge_clicked:
        if not challenge_text.strip():
            st.markdown('<div class="empty-hint">Type something first so I have a challenge to analyze 🙂</div>', unsafe_allow_html=True)
        elif check_safety_concern(challenge_text):
            st.markdown(f"""
            <div class="gemini-panel">
                <div class="gemini-title">💛 A note before anything else</div>
                {SAFETY_RESPONSE}
            </div>
            """, unsafe_allow_html=True)
            log_session("study_challenge", "-", "flagged", "flagged", challenge_text, redact=True)
        else:
            with st.spinner("Analyzing your challenge..."):
                r_scores = rule_based_scores(challenge_text)
                t_scores = transformer_scores(challenge_text, zero_shot_classifier)
                final_scores = enhanced_fusion(r_scores, t_scores)
                final_label, emotion_parts, top_confidence = detect_mixed_emotion(final_scores)

            if show_comparison:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown('<div class="study-card">', unsafe_allow_html=True)
                    st.markdown('<span class="model-badge badge-rule">Rule-based (keyword)</span>', unsafe_allow_html=True)
                    st.markdown(render_study_bars(r_scores, "#FFD98F", "#FFA84D"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_b:
                    st.markdown('<div class="study-card">', unsafe_allow_html=True)
                    st.markdown('<span class="model-badge badge-bert">Transformer (BERT-family)</span>', unsafe_allow_html=True)
                    st.markdown(render_study_bars(t_scores, "#8CB6FF", "#5B8CFF"), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            emoji_str = " ".join(EMOTION_EMOJI.get(e.lower(), "🙂") for e in emotion_parts)
            st.markdown(f"""
            <div class="result-card card-study">
                <div class="result-emoji">{emoji_str}</div>
                <div class="result-label">{final_label}</div>
            </div>
            """, unsafe_allow_html=True)

            if top_confidence < 35:
                st.caption(f"⚠️ Low confidence read ({top_confidence}%) — treating this as a best guess, not a certain diagnosis.")

            if use_gemini:
                with st.spinner("Getting personalized guidance..."):
                    guidance_text = call_gemini(challenge_text, final_label)
                panel_title = "✨ AI-generated guidance"
            else:
                guidance_text = get_study_fallback(final_label)
                panel_title = "💡 Quick tip"

            st.markdown(f"""
            <div class="gemini-panel">
                <div class="gemini-title">{panel_title}</div>
                {guidance_text}
            </div>
            """, unsafe_allow_html=True)

            log_session("study_challenge", "-", final_label, final_label, challenge_text)

# =========================================================
# HISTORY / ANALYTICS DASHBOARD
# =========================================================
st.divider()
with st.expander("📊 View session history & analytics"):
    if os.path.isfile("emotion_log.csv"):
        try:
            log = pd.read_csv("emotion_log.csv")
        except Exception:
            st.error("The session log file appears corrupted or from an older app version "
                      "(different column structure). Click below to reset it.")
            if st.button("🗑️ Reset corrupted history"):
                os.remove("emotion_log.csv")
                st.success("History reset. Refresh the page to start fresh.")
            log = None

        if log is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Sessions logged", len(log))
            m2.metric("Most common mood", log['final'].mode()[0] if not log.empty else "-")
            m3.metric("Last mood", log['final'].iloc[-1] if not log.empty else "-")

            mode_filter = st.multiselect("Filter by mode", options=log['mode'].unique().tolist(),
                                          default=log['mode'].unique().tolist())
            filtered = log[log['mode'].isin(mode_filter)]

            st.dataframe(filtered, use_container_width=True, hide_index=True)

            st.markdown("**Emotion counts**")
            st.bar_chart(filtered['final'].value_counts())

            if len(filtered) >= 2:
                st.markdown("**Emotion trend over time (cumulative)**")
                trend_df = filtered.copy()
                trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'])
                trend = (
                    trend_df.groupby([trend_df['timestamp'], 'final'])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                    .cumsum()
                )
                st.line_chart(trend)
            else:
                st.caption("Log a few more sessions to see the trend chart.")

            if st.button("🗑️ Clear history"):
                os.remove("emotion_log.csv")
                st.success("History cleared. Refresh the page to see changes.")
    else:
        st.write("No sessions logged yet — try either mode above! 👆")