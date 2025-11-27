import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "kaggle_posts.csv"
OUTPUT_FILE = DATA_DIR / "kaggle_posts_with_sentiment.csv"

# 👉 CHANGE THIS to match the actual text column name in your Kaggle CSV
TEXT_COLUMN = "Text"  # e.g. "tweet", "content", "Review", etc.

def label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "POSITIVE"
    elif c <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def main():
    print(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"TEXT_COLUMN='{TEXT_COLUMN}' not found in CSV columns: {list(df.columns)}"
        )

    analyzer = SentimentIntensityAnalyzer()

    compounds = []
    labels = []
    negs = []
    neus = []
    poss = []

    for text in df[TEXT_COLUMN]:
        scores = analyzer.polarity_scores(str(text))
        c = scores["compound"]

        compounds.append(c)
        labels.append(label_from_compound(c))
        negs.append(scores["neg"])
        neus.append(scores["neu"])
        poss.append(scores["pos"])

    df["sent_compound"] = compounds
    df["sent_label"] = labels
    df["sent_neg"] = negs
    df["sent_neu"] = neus
    df["sent_pos"] = poss

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved with sentiment to: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
