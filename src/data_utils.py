import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# ── Label conversion ──────────────────────────────────────────
fake_labels = ["false", "barely-true", "pants-fire"]
real_labels = ["true", "mostly-true", "half-true"]

def convert_label(label):
    if label in fake_labels:
        return "fake"
    elif label in real_labels:
        return "real"
    return None

# ── Text cleaning ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ── Load raw dataset ──────────────────────────────────────────
def load_data(train_path, valid_path, test_path):
    columns = [
        "id", "label", "statement", "subject",
        "speaker", "job_title", "state_info",
        "party_affiliation", "barely_true_count",
        "false_count", "half_true_count",
        "mostly_true_count", "pants_on_fire_count",
        "context"
    ]
    train_df = pd.read_csv(train_path, sep="\t", header=None, names=columns)
    valid_df = pd.read_csv(valid_path, sep="\t", header=None, names=columns)
    test_df  = pd.read_csv(test_path,  sep="\t", header=None, names=columns)
    return train_df, valid_df, test_df

# ── Full preprocessing pipeline ───────────────────────────────
def preprocess(df):
    df = df.copy()
    df["binary_label"]    = df["label"].apply(convert_label)
    df = df.dropna(subset=["binary_label"])
    df["clean_statement"] = df["statement"].apply(clean_text)
    df["word_count"]      = df["clean_statement"].apply(lambda x: len(x.split()))
    return df

# ── Save processed data ───────────────────────────────────────
def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")