import joblib
import sys
sys.path.append("src")
from data_utils import clean_text

def predict(text, model_path="models/baseline_model.pkl",
            vectorizer_path="models/tfidf_vectorizer.pkl"):
    model     = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    cleaned = clean_text(text)
    vector  = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()

    return prediction, confidence

if __name__ == "__main__":
    sample = "The government is hiding the truth about vaccines"
    pred, conf = predict(sample)
    print(f"Prediction : {pred}")
    print(f"Confidence : {conf:.2%}")