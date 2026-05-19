# Fake News Detector 🔍

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end NLP project that classifies news headlines as 
**real or fake** using machine learning and transformer models,
trained on the LIAR benchmark dataset.

---

## Project Progress

- [x] Week 1: Data loading, EDA, text cleaning, preprocessing pipeline
- [ ] Week 2: Baseline ML models (TF-IDF + Logistic Regression)
- [ ] Week 3: DistilBERT fine-tuning
- [ ] Week 4: Streamlit deployment + live demo

---

## Dataset

[LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
— 12,836 labeled political statements across 6 classes,
converted to binary (real / fake) for this project.

| Split | Samples |
|---|---|
| Train | 10,240 |
| Validation | 1,284 |
| Test | 1,267 |

---

## Project Structure

---

## Week 1 Results — EDA Findings

- Dataset is nearly balanced: 51% fake, 49% real
- Average statement length: ~18 words
- Top speakers: Barack Obama, Donald Trump, Hillary Clinton
- Most common subjects: healthcare, economy, elections

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas + NumPy | Data manipulation |
| NLTK | Text preprocessing |
| Scikit-learn | Baseline ML models (Week 2) |
| HuggingFace Transformers | DistilBERT fine-tuning (Week 3) |
| SHAP | Model explainability (Week 4) |
| Streamlit | Web app deployment (Week 4) |

---

## Results (updating as project progresses)

| Model | Accuracy | F1 Score |
|---|---|---|
| TF-IDF + Logistic Regression | Coming Week 2 | - |
| TF-IDF + SVM | Coming Week 2 | - |
| DistilBERT fine-tuned | Coming Week 3 | - |

---

## Setup Instructions

```bash
# Clone the repo
git clone https://github.com/pulipakasantosh/Fake-News-Detector-.git
cd Fake-News-Detector-

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py
```

---

## Author

**Santosh Pulipaka**
3rd Year B.Tech Student | AI/ML Enthusiast

[GitHub](https://github.com/pulipakasantosh)

---

## License

MIT License — free to use and modify.