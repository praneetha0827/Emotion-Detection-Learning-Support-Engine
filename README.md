# Emotion Detection & Learning Support Engine

An AI-powered assistant that detects a learner's emotional state — from their
face, their words, or both — and responds with personalized, supportive
guidance. Includes two modes: a webcam + text check-in, and an AI-guided
Study Challenge mode with dual-model emotion comparison and live Gemini
feedback.

## Problem statement

Learners often disengage silently — frustration, confusion, or boredom go
unnoticed until performance drops. This project explores whether combining
multiple emotion signals (facial expression, free text, keyword patterns)
can catch this earlier and respond with timely, relevant support instead of
a generic message.

## Features

### Mode 1 — Webcam + Text
- Captures a webcam photo and a short text description of how the learner feels
- Detects facial emotion (DeepFace) with a viewfinder overlay showing the
  detected face region
- Detects text emotion (DistilRoBERTa transformer)
- Fuses both signals into one final emotion and shows a matching supportive message

### Mode 2 — Study Challenge (AI-guided)
- Learner describes a specific study challenge in free text
- Two models analyze it side by side:
  - **Rule-based (keyword) model** — scores text against 5 study-specific
    emotions (Bored, Confident, Confused, Curious, Frustrated) by keyword matching
  - **Transformer model** — a pretrained BERT-family zero-shot classifier
    (`typeform/distilbert-base-uncased-mnli`) scores the same 5 categories
    without needing training data for them
- Rule-based keyword hits enhance (boost) the transformer's scores
- Detects **mixed emotions** (e.g. "Confused + Curious") when top scores are close
- Flags **low-confidence reads** instead of presenting an uncertain guess as fact
- Sends the challenge + detected emotion to the **Gemini API** for a live,
  personalized 2–3 sentence tip — with a toggle to use a canned fallback
  tip instead (no API call)
- **Safety keyword filter**: concerning language (self-harm indicators) is
  redirected to a supportive message instead of being processed as a study
  emotion, and is redacted before being logged

### Shared
- Animated space-themed UI (twinkling stars, drifting nebula background)
- Session logging to `emotion_log.csv` (mode, detected emotions, timestamp, input text)
- Analytics dashboard: session counts, most common mood, emotion count chart,
  and a cumulative emotion trend-over-time chart, filterable by mode

## Tech stack

| Component | Tool |
|---|---|
| Face emotion | DeepFace (pretrained CNN) |
| Face detection / overlay | OpenCV |
| Text emotion (Mode 1) | HuggingFace `j-hartmann/emotion-english-distilroberta-base` |
| Text emotion (Mode 2, transformer) | HuggingFace `typeform/distilbert-base-uncased-mnli` (zero-shot) |
| Text emotion (Mode 2, rule-based) | Custom keyword-matching classifier |
| AI-generated guidance | Google Gemini API (`gemini-2.5-flash`) |
| UI | Streamlit |
| Logging & analytics | Pandas + CSV |

## How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your Gemini API key**

Create `.streamlit/secrets.toml` in the project root:
```toml
GEMINI_API_KEY = "your-key-here"
```
Get a key from https://aistudio.google.com/apikey. This file is gitignored
and should never be committed.

**3. Run the app**
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `localhost:8501`).

## Project structure

```
emotion-learning-engine/
├── app.py                   # Main Streamlit app (both modes)
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── secrets.toml         # Gemini API key (gitignored, not in repo)
├── .gitignore
├── README.md
└── RESPONSIBLE_AI.md        # Limitations, bias, safety & privacy notes
```

## Known limitations

See `RESPONSIBLE_AI.md` for the full write-up. In short:
- The "rule-based vs transformer" comparison substitutes a keyword model for
  a trained BiLSTM, since no labeled dataset exists for the 5 custom study
  emotions — this is disclosed, not hidden.
- Facial and zero-shot text emotion detection both have real accuracy limits
  (lighting/angle for face; no fine-tuning for zero-shot text).
- The safety filter is a basic keyword net, not a clinical crisis-detection system.
- Session data is stored locally in plaintext CSV; Gemini calls send text to
  Google's servers for processing.
