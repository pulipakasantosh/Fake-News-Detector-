import sys
sys.path.append("src")

import joblib
import warnings
warnings.filterwarnings('ignore')

from data_utils import clean_text

def predict(text,
            model_path="models/svm_model.pkl",
            vectorizer_path="models/tfidf_vectorizer.pkl"):

    model      = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    cleaned    = clean_text(text)
    vector     = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    # confidence only available for LR not SVM
    try:
        confidence = model.predict_proba(vector).max()
    except AttributeError:
        confidence = None

    return prediction, confidence

def main():
    test_headlines = [
        "Government secretly adding chemicals to drinking water",
        "Senate approves new infrastructure spending bill",
        "Scientists find that coffee cures all diseases",
        "Federal Reserve raises interest rates by 0.25 percent",
        "President signs executive order on climate change"
    ]

    print("=" * 60)
    print("FAKE NEWS DETECTOR — PREDICTIONS")
    print("=" * 60)

    for headline in test_headlines:
        pred, conf = predict(headline)
        conf_str   = f"{conf:.2%}" if conf else "N/A"
        print(f"\nHeadline  : {headline}")
        print(f"Prediction: {pred.upper()}")
        print(f"Confidence: {conf_str}")
        print("-" * 60)

if __name__ == "__main__":
    main()