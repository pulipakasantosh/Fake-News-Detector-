import sys
sys.path.append("src")

import os
import re
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import subprocess

def ensure_models():
    models_needed = [
        "models/svm_model.pkl",
        "models/lr_model.pkl",
        "models/tfidf_vectorizer.pkl"
    ]
    if not all(os.path.exists(m)
               for m in models_needed):
        with st.spinner(
            "🔄 Setting up models for "
            "first time — 2-3 minutes..."
        ):
            result = subprocess.run(
                ["python", "src/train.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Models ready!")
                st.rerun()
            else:
                st.error(
                    f"Training failed: "
                    f"{result.stderr[:200]}")

ensure_models()
# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Text cleaning ─────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens
              if w not in stop_words]
    return " ".join(tokens)

# ── Check models ──────────────────────────────
SVM_EXISTS = (
    os.path.exists("models/svm_model.pkl") and
    os.path.exists("models/tfidf_vectorizer.pkl"))
LR_EXISTS  = (
    os.path.exists("models/lr_model.pkl") and
    os.path.exists("models/tfidf_vectorizer.pkl"))

# ── Load models ───────────────────────────────
@st.cache_resource
def load_models():
    if not SVM_EXISTS:
        return None, None, None
    svm = joblib.load("models/svm_model.pkl")
    lr  = joblib.load("models/lr_model.pkl")
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    return svm, lr, vec

# ── Predictions ───────────────────────────────
def predict_svm(text, svm, vec):
    cleaned = clean_text(text)
    vector  = vec.transform([cleaned])
    pred    = svm.predict(vector)[0]
    return pred

def predict_lr(text, lr, vec):
    cleaned = clean_text(text)
    vector  = vec.transform([cleaned])
    pred    = lr.predict(vector)[0]
    conf    = lr.predict_proba(vector).max()
    return pred, conf

def analyze_text(text):
    words      = text.split()
    caps_ratio = sum(
        1 for c in text if c.isupper()
    ) / max(len(text), 1)
    red_flags  = []
    if caps_ratio > 0.3:
        red_flags.append(
            "⚠️ Excessive capitalization")
    if text.count('!') > 1:
        red_flags.append(
            "⚠️ Multiple exclamation marks")
    if len(words) < 5:
        red_flags.append(
            "⚠️ Very short headline")
    if any(w in text.lower() for w in [
            "secret", "shocking", "exposed",
            "hidden", "conspiracy", "hoax"]):
        red_flags.append(
            "⚠️ Sensational language detected")
    if any(w in text.lower() for w in [
            "100%", "guaranteed", "miracle",
            "cure", "proven"]):
        red_flags.append(
            "⚠️ Absolute claims detected")
    return {
        "word_count" : len(words),
        "char_count" : len(text),
        "caps_ratio" : round(caps_ratio * 100, 1),
        "exclamation": text.count('!'),
        "red_flags"  : red_flags
    }

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown(
    "AI-powered fake news detection using "
    "**SVM** and **Logistic Regression** "
    "trained on the LIAR dataset (12,836 statements)."
)
st.markdown("---")

# ── Layout ────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📝 Enter a news headline")
    headline = st.text_area(
        label="headline",
        label_visibility="collapsed",
        placeholder="Paste or type a news headline...",
        height=120
    )

    st.markdown("**Try an example:**")
    ex1, ex2, ex3, ex4 = st.columns(4)
    examples = {
        "🔴 Fake 1": "Government secretly adding chemicals to drinking water nationwide",
        "🔴 Fake 2": "SHOCKING cure hidden by pharma companies exposed",
        "🟢 Real 1": "Senate passes bipartisan infrastructure spending bill",
        "🟢 Real 2": "Federal Reserve raises interest rates by 0.25 percent"
    }
    for col, (label, text) in zip(
            [ex1, ex2, ex3, ex4],
            examples.items()):
        with col:
            if st.button(label,
                         use_container_width=True):
                headline = text

    analyse = st.button(
        "🔍 Analyse Headline",
        type="primary",
        use_container_width=True
    )

with col_right:
    st.markdown("### 📊 Model Performance")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    models_list = ["LR", "SVM"]
    accs        = [78, 81]
    colors      = ["#9b9b9b", "#534AB7"]
    bars = ax.barh(models_list, accs, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Baseline model comparison")
    for bar, val in zip(bars, accs):
        ax.text(
            val + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{val}%", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    c1, c2 = st.columns(2)
    c1.metric("Train samples", "10,240")
    c2.metric("Test samples",  "1,267")
    c1.metric("Fake news",     "51%")
    c2.metric("Real news",     "49%")

# ── Results ───────────────────────────────────
if analyse and headline.strip():
    st.markdown("---")
    st.markdown("## 📋 Analysis Results")

    svm, lr, vec = load_models()

    if svm is None:
        st.error(
            "Models not found! "
            "Please run `python src/train.py` "
            "to train the models first.")
        st.stop()

    # Predictions
    svm_pred         = predict_svm(headline, svm, vec)
    lr_pred, lr_conf = predict_lr(headline, lr, vec)
    text_stats       = analyze_text(headline)

    # Ensemble vote
    votes  = [svm_pred, lr_pred]
    final  = "fake" if votes.count("fake") >= 1 \
             else "real"

    # ── Verdict ───────────────────────────────
    st.markdown("### 🏆 Final Verdict")
    if final == "fake":
        st.error(
            f"🔴 **FAKE NEWS** detected — "
            f"{votes.count('fake')}/2 models agree")
    else:
        st.success(
            f"🟢 **REAL NEWS** — "
            f"{votes.count('real')}/2 models agree")

    st.markdown("---")

    # ── Model results ─────────────────────────
    st.markdown("### 🤖 Individual Model Results")
    m1, m2 = st.columns(2)

    with m1:
        st.markdown("**SVM (TF-IDF)**")
        if svm_pred == "fake":
            st.error("🔴 FAKE")
        else:
            st.success("🟢 REAL")
        st.caption("Support Vector Machine")
        st.caption("Validation accuracy: ~81%")

    with m2:
        st.markdown("**Logistic Regression**")
        if lr_pred == "fake":
            st.error("🔴 FAKE")
        else:
            st.success("🟢 REAL")
        st.progress(float(lr_conf))
        st.caption(f"Confidence: {lr_conf:.1%}")
        st.caption("Validation accuracy: ~78%")

    st.markdown("---")

    # ── Confidence chart ──────────────────────
    st.markdown("### 📊 Confidence Breakdown")
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    conf_vals = [lr_conf, 1 - lr_conf]
    conf_lbls = ["Fake", "Real"]
    conf_cols = ["#E24B4A", "#1D9E75"]
    bars2     = ax2.barh(
        conf_lbls, conf_vals, color=conf_cols)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_title(
        "Logistic Regression confidence")
    for bar, val in zip(bars2, conf_vals):
        ax2.text(
            val + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:.1%}",
            va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # ── Text analysis ─────────────────────────
    st.markdown("### 🔎 Headline Analysis")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Words",        text_stats["word_count"])
    t2.metric("Characters",   text_stats["char_count"])
    t3.metric("CAPS ratio",
              f"{text_stats['caps_ratio']}%")
    t4.metric("Exclamations",
              text_stats["exclamation"])

    st.markdown("**Cleaned text (model input):**")
    st.info(clean_text(headline))

    if text_stats["red_flags"]:
        st.markdown("**🚩 Red flags detected:**")
        for flag in text_stats["red_flags"]:
            st.warning(flag)
    else:
        st.success(
            "✅ No obvious red flags detected")

    st.markdown("---")

    # ── Voting table ──────────────────────────
    st.markdown("### 🗳️ Model Voting Summary")
    vote_df = pd.DataFrame({
        "Model"     : ["SVM", "Logistic Regression"],
        "Prediction": [
            svm_pred.upper(),
            lr_pred.upper()
        ],
        "Vote"      : [
            "🔴 Fake" if svm_pred == "fake"
            else "🟢 Real",
            "🔴 Fake" if lr_pred  == "fake"
            else "🟢 Real"
        ]
    })
    st.table(vote_df)

elif analyse and not headline.strip():
    st.warning("⚠️ Please enter a headline first!")

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 About")
    st.markdown(
        "Detects fake news using 2 ML models:")
    st.markdown("- 📊 **SVM** — TF-IDF based")
    st.markdown(
        "- 📈 **Logistic Regression** — baseline")
    st.markdown("---")
    st.markdown("## 📚 How it works")
    st.markdown(
        "1. Text cleaned and vectorized\n"
        "2. Both models predict\n"
        "3. Majority vote = final verdict\n"
        "4. Confidence scores shown"
    )
    st.markdown("---")
    st.markdown("## 🗃️ Dataset")
    st.markdown(
        "**LIAR Dataset**\n\n"
        "12,836 labeled statements\n\n"
        "Source: UCSB NLP Group"
    )
    st.markdown("---")
    st.markdown("## ⚠️ Disclaimer")
    st.info(
        "Educational purposes only. "
        "Always verify from trusted sources."
    )
    st.markdown("---")
    st.markdown(
        "Built by **Santosh Pulipaka**\n\n"
        "3rd Year B.Tech | AI/ML\n\n"
        "[GitHub]"
        "(https://github.com/pulipakasantosh)"
    )