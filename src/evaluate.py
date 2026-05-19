from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_name, y_true, y_pred):
    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred, pos_label='fake'):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, pos_label='fake'):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, pos_label='fake'):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["fake", "real"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["fake", "real"],
                yticklabels=["fake", "real"],
                cmap="Blues")
    plt.title(f"Confusion matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()