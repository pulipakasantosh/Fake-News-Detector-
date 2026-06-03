import sys
sys.path.append("src")

import streamlit as st
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from transformers import (DistilBertTokenizer,
    DistilBertForSequenceClassification)
from data_utils import clean_text

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
.big-label {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 8px;
}
.metric-card {
    background: #f0f2f6;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.fake-badge {
    background: #ffcccc;
    color: #cc0000;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: bold;
}
.real-badge {
    background: #ccffcc;
    color: #006600;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: bold;
}
.info-box {
    background: #e8f4fd;
    border-left: 4px solid #1a73e8;
    padding: 12px;
    border-radius: 4px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────
@st.cache_resource
def load_bert_model():
    device    = torch.device("cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased")
    model     = DistilBertForSequenceClassification\
        .from_pretrained(
            "distilbert-base-uncased",
            num_labels=2)
    model.load_state_dict(
        torch.load("models/best_distilbert.pt",
                   map_location=device))
    model.eval()
    return model, tokenizer, device

@st.cache_resource
def load_baseline_models():
    lr  = joblib.load("models/lr_model.pkl")
    svm = joblib.load("models/svm_model.pkl")
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    return lr, svm, vec

# ── Prediction functions ──────────────────────
def predict_bert(text, model, tokenizer, device):
    cleaned  = clean_text(text)
    encoding = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        probs     = torch.softmax(outputs.logits, dim=1)
        pred      = torch.argmax(probs, dim=1).item()
        fake_conf = probs[0][0].item()
        real_conf = probs[0][1].item()
    label = "fake" if pred == 0 else "real"
    return label, fake_conf, real_conf

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
    words       = text.split()
    sentences   = text.split('.')
    has_caps    = sum(1 for c in text if c.isupper())
    caps_ratio  = has_caps / max(len(text), 1)
    exclamation = text.count('!')
    question    = text.count('?')
    word_count  = len(words)

    # Red flags
    red_flags = []
    if caps_ratio > 0.3:
        red_flags.append("⚠️ Excessive capitalization")
    if exclamation > 1:
        red_flags.append("⚠️ Multiple exclamation marks")
    if word_count < 5:
        red_flags.append("⚠️ Very short headline")
    if any(w in text.lower() for w in
           ["secret", "shocking", "exposed",
            "hidden", "conspiracy", "hoax"]):
        red_flags.append("⚠️ Sensational language detected")
    if any(w in text.lower() for w in
           ["100%", "guaranteed", "proven",
            "miracle", "cure"]):
        red_flags.append("⚠️ Absolute claims detected")

    return {
        "word_count"  : word_count,
        "char_count"  : len(text),
        "caps_ratio"  : round(caps_ratio * 100, 1),
        "exclamation" : exclamation,
        "question"    : question,
        "red_flags"   : red_flags
    }

# ── Header ────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown(
    "AI-powered fake news detection using "
    "**DistilBERT** and **TF-IDF** models "
    "trained on the LIAR dataset."
)
st.markdown("---")

# ── Main layout ───────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📝 Enter a news headline")
    headline = st.text_area(
        label="headline",
        label_visibility="collapsed",
        placeholder="Paste or type a news headline here...",
        height=120
    )

    # Example headlines
    st.markdown("**Try an example:**")
    ex1, ex2, ex3, ex4 = st.columns(4)
    examples = {
        "🔴 Fake 1": "Government secretly adding chemicals to drinking water nationwide",
        "🔴 Fake 2": "SHOCKING: Scientists discover miracle cure hidden by pharma",
        "🟢 Real 1": "Senate passes bipartisan infrastructure spending bill",
        "🟢 Real 2": "Federal Reserve raises interest rates by 0.25 percent"
    }
    for col, (label, text) in zip(
            [ex1, ex2, ex3, ex4], examples.items()):
        with col:
            if st.button(label, use_container_width=True):
                headline = text

    analyse = st.button(
        "🔍 Analyse Headline",
        type="primary",
        use_container_width=True
    )

with col_right:
    st.markdown("### 📊 Model Performance")
    perf_data = {
        "Model"   : ["LR", "SVM", "DistilBERT"],
        "Accuracy": [78, 81, 84]
    }
    fig, ax = plt.subplots(figsize=(4, 2.5))
    bars = ax.barh(
        perf_data["Model"],
        perf_data["Accuracy"],
        color=["#9b9b9b", "#6b6bcc", "#534AB7"]
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Model comparison")
    for bar, val in zip(bars, perf_data["Accuracy"]):
        ax.text(val + 0.5, bar.get_y() +
                bar.get_height()/2,
                f"{val}%", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Dataset stats:**")
    c1, c2 = st.columns(2)
    c1.metric("Train samples", "10,240")
    c2.metric("Test samples",  "1,267")
    c1.metric("Fake news",     "51%")
    c2.metric("Real news",     "49%")

# ── Results ───────────────────────────────────
if analyse and headline.strip():
    st.markdown("---")
    st.markdown("## 📋 Analysis Results")

    # Load models
    bert_model, tokenizer, device = load_bert_model()
    lr, svm, vec = load_baseline_models()

    # Get predictions
    bert_label, fake_conf, real_conf = predict_bert(
        headline, bert_model, tokenizer, device)
    svm_label         = predict_svm(headline, svm, vec)
    lr_label, lr_conf = predict_lr(headline, lr, vec)
    text_stats        = analyze_text(headline)

    # Verdict
    votes = [bert_label, svm_label, lr_label]
    fake_votes = votes.count("fake")
    real_votes = votes.count("real")
    final = "fake" if fake_votes >= 2 else "real"

    # ── Verdict banner ────────────────────────
    st.markdown("### 🏆 Final Verdict")
    if final == "fake":
        st.error(
            f"🔴 **FAKE NEWS** — "
            f"{fake_votes}/3 models flagged this as fake")
    else:
        st.success(
            f"🟢 **REAL NEWS** — "
            f"{real_votes}/3 models classified this as real")

    st.markdown("---")

    # ── Three model results ───────────────────
    st.markdown("### 🤖 Individual Model Results")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("**DistilBERT**")
        if bert_label == "fake":
            st.error(f"🔴 FAKE")
        else:
            st.success(f"🟢 REAL")
        st.progress(fake_conf if bert_label == "fake"
                    else real_conf)
        st.caption(
            f"Fake: {fake_conf:.1%} | "
            f"Real: {real_conf:.1%}")
        st.caption("Fine-tuned transformer model")

    with m2:
        st.markdown("**SVM (TF-IDF)**")
        if svm_label == "fake":
            st.error(f"🔴 FAKE")
        else:
            st.success(f"🟢 REAL")
        st.caption("Support Vector Machine")
        st.caption("Accuracy: ~81%")

    with m3:
        st.markdown("**Logistic Regression**")
        if lr_label == "fake":
            st.error(f"🔴 FAKE")
        else:
            st.success(f"🟢 REAL")
        st.progress(lr_conf)
        st.caption(f"Confidence: {lr_conf:.1%}")
        st.caption("TF-IDF baseline model")

    st.markdown("---")

    # ── Confidence chart ──────────────────────
    st.markdown("### 📊 DistilBERT Confidence Breakdown")
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    colors = ["#E24B4A", "#1D9E75"]
    bars2  = ax2.barh(
        ["Fake", "Real"],
        [fake_conf * 100, real_conf * 100],
        color=colors
    )
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title("Probability distribution")
    for bar, val in zip(
            bars2, [fake_conf, real_conf]):
        ax2.text(
            val * 100 + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # ── Text analysis ─────────────────────────
    st.markdown("### 🔎 Headline Analysis")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Word count",    text_stats["word_count"])
    t2.metric("Characters",    text_stats["char_count"])
    t3.metric("CAPS ratio",    f"{text_stats['caps_ratio']}%")
    t4.metric("Exclamations",  text_stats["exclamation"])

    # Cleaned text
    st.markdown("**Cleaned text (model input):**")
    st.info(clean_text(headline))

    # Red flags
    if text_stats["red_flags"]:
        st.markdown("**🚩 Red flags detected:**")
        for flag in text_stats["red_flags"]:
            st.warning(flag)
    else:
        st.success(
            "✅ No obvious red flags in text style")

    st.markdown("---")

    # ── Voting summary ────────────────────────
    st.markdown("### 🗳️ Model Voting Summary")
    vote_df = pd.DataFrame({
        "Model"     : ["DistilBERT", "SVM", "LR"],
        "Prediction": [
            bert_label.upper(),
            svm_label.upper(),
            lr_label.upper()
        ],
        "Vote": [
            "🔴 Fake" if bert_label == "fake"
            else "🟢 Real",
            "🔴 Fake" if svm_label  == "fake"
            else "🟢 Real",
            "🔴 Fake" if lr_label   == "fake"
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
        "This app uses 3 ML models to detect "
        "fake news headlines:"
    )
    st.markdown("- 🤖 **DistilBERT** — transformer")
    st.markdown("- 📊 **SVM** — TF-IDF based")
    st.markdown("- 📈 **Logistic Regression**")
    st.markdown("---")
    st.markdown("## 📚 How it works")
    st.markdown(
        "1. Text is cleaned and tokenized\n"
        "2. Each model makes a prediction\n"
        "3. Majority vote = final verdict\n"
        "4. Confidence scores are shown"
    )
    st.markdown("---")
    st.markdown("## 🗃️ Dataset")
    st.markdown(
        "Trained on **LIAR Dataset**\n\n"
        "12,836 labeled political statements\n\n"
        "Source: UCSB NLP Group"
    )
    st.markdown("---")
    st.markdown("## ⚠️ Disclaimer")
    st.info(
        "This tool is for educational purposes. "
        "Always verify news from multiple "
        "trusted sources."
    )
    st.markdown("---")
    st.markdown(
        "Built by **Santosh Pulipaka**\n\n"
        "3rd Year B.Tech | AI/ML\n\n"
        "[GitHub](https://github.com/pulipakasantosh)"
    )