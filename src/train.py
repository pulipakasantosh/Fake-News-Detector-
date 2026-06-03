import sys
import os
sys.path.append("src")

import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

from data_utils import load_data, preprocess

def build_vectorizer():
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

def train_all_models(X_train, y_train,
                     X_valid, y_valid):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42),
        "Naive Bayes"        : MultinomialNB(alpha=0.1),
        "SVM"                : LinearSVC(
            max_iter=2000, C=1.0, random_state=42)
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

        acc = accuracy_score(y_valid, preds)
        f1  = f1_score(y_valid, preds, pos_label='fake')

        results.append({
            "Model"   : name,
            "Accuracy": round(acc, 4),
            "F1 Score": round(f1, 4)
        })
        trained[name] = model
        print(f"{name:<25} Accuracy: {acc:.4f}  F1: {f1:.4f}")

    return trained, pd.DataFrame(results)

def save_models(trained_models, vectorizer):
    os.makedirs("models", exist_ok=True)

    joblib.dump(vectorizer,
                "models/tfidf_vectorizer.pkl")
    joblib.dump(trained_models["Logistic Regression"],
                "models/lr_model.pkl")
    joblib.dump(trained_models["Naive Bayes"],
                "models/nb_model.pkl")
    joblib.dump(trained_models["SVM"],
                "models/svm_model.pkl")

    print("\nAll models saved to models/ folder")

def main():
    print("=" * 50)
    print("FAKE NEWS DETECTOR — BASELINE TRAINING")
    print("=" * 50)

    print("\nStep 1: Loading data...")
    train_df, valid_df, test_df = load_data(
        "data/liar_dataset/train.tsv",
        "data/liar_dataset/valid.tsv",
        "data/liar_dataset/test.tsv"
    )

    print("Step 2: Preprocessing...")
    train_df = preprocess(train_df)
    valid_df = preprocess(valid_df)
    test_df  = preprocess(test_df)

    print("Step 3: Vectorizing text...")
    tfidf         = build_vectorizer()
    X_train_tfidf = tfidf.fit_transform(
                        train_df["clean_statement"])
    X_valid_tfidf = tfidf.transform(
                        valid_df["clean_statement"])

    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

    print("\nStep 4: Training models...")
    trained, results_df = train_all_models(
        X_train_tfidf, train_df["binary_label"],
        X_valid_tfidf, valid_df["binary_label"]
    )

    print("\nStep 5: Results summary")
    print(results_df.to_string(index=False))

    print("\nStep 6: Saving models...")
    save_models(trained, tfidf)

    print("\nTraining complete!")
    return trained, tfidf, results_df

if __name__ == "__main__":
    main()
