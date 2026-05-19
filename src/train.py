import sys
sys.path.append("src")

import pandas as pd
from data_utils import load_data, preprocess

def train_baseline():
    print("Loading data...")
    train_df, valid_df, test_df = load_data(
        "data/liar_dataset/train.tsv",
        "data/liar_dataset/valid.tsv",
        "data/liar_dataset/test.tsv"
    )

    print("Preprocessing...")
    train_df = preprocess(train_df)
    valid_df = preprocess(valid_df)
    test_df  = preprocess(test_df)

    X_train = train_df["clean_statement"]
    y_train = train_df["binary_label"]
    X_valid = valid_df["clean_statement"]
    y_valid = valid_df["binary_label"]

    print("Data ready!")
    print(f"Train: {len(X_train)} | Valid: {len(X_valid)}")

    # Week 2 — models will be added here

if __name__ == "__main__":
    train_baseline()