import sys
sys.path.append("src")

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix,
                             classification_report)
from data_utils import load_data, preprocess

def evaluate_model(name, model, X, y_true):
    preds = model.predict(X)

    print(f"\n{'='*45}")
    print(f"Model: {name}")
    print(f"{'='*45}")
    print(f"Accuracy  : {accuracy_score(y_true, preds):.4f}")
    print(f"F1 Score  : {f1_score(y_true, preds, pos_label='fake'):.4f}")
    print(f"Precision : {precision_score(y_true, preds, pos_label='fake'):.4f}")
    print(f"Recall    : {recall_score(y_true, preds, pos_label='fake'):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, preds))

    return preds

def plot_confusion_matrix(name, y_true, preds):
    cm = confusion_matrix(
             y_true, preds, labels=["fake", "real"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["fake", "real"],
                yticklabels=["fake", "real"],
                cmap="Blues")
    plt.title(f"Confusion matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 50)
    print("FAKE NEWS DETECTOR — EVALUATION")
    print("=" * 50)

    print("\nLoading data...")
    _, _, test_df = load_data(
        "data/liar_dataset/train.tsv",
        "data/liar_dataset/valid.tsv",
        "data/liar_dataset/test.tsv"
    )
    test_df = preprocess(test_df)

    print("Loading models...")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    lr    = joblib.load("models/lr_model.pkl")
    nb    = joblib.load("models/nb_model.pkl")
    svm   = joblib.load("models/svm_model.pkl")

    X_test = tfidf.transform(test_df["clean_statement"])
    y_test = test_df["binary_label"]

    models = {
        "Logistic Regression": lr,
        "Naive Bayes"        : nb,
        "SVM"                : svm
    }

    for name, model in models.items():
        preds = evaluate_model(name, model, X_test, y_test)
        plot_confusion_matrix(name, y_test, preds)

if __name__ == "__main__":
    main()