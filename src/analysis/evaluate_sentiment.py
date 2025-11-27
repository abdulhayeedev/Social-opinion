import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "kaggle_posts_with_sentiment.csv"

# Kaggle dataset's sentiment column (ground truth)
# Adjust this if Kaggle uses different names like "SentimentLabel", etc.
TRUE_COL = "Sentiment"

# Your model's output
PRED_COL = "sent_label"

def clean_label(x):
    """Normalize labels so they match (lowercase or consistent wording)."""
    if isinstance(x, str):
        x = x.strip().lower()
    return x

def main():
    df = pd.read_csv(INPUT_FILE)

    # Clean labels
    df[TRUE_COL] = df[TRUE_COL].apply(clean_label)
    df[PRED_COL] = df[PRED_COL].apply(clean_label)

    # OPTIONAL: If Kaggle uses words like "Positive", "Negative", "Neutral"
    mapping = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
    }

    df[TRUE_COL] = df[TRUE_COL].map(mapping)
    df[PRED_COL] = df[PRED_COL].map(mapping)

    # Drop missing values after mapping
    df = df.dropna(subset=[TRUE_COL, PRED_COL])

    y_true = df[TRUE_COL]
    y_pred = df[PRED_COL]

    print("\n===== SENTIMENT CLASSIFICATION EVALUATION =====\n")

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}")

    # Full classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
