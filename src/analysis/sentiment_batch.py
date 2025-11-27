import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Paths
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "sample_posts.csv"
OUTPUT_FILE = DATA_DIR / "sample_posts_with_sentiment.csv"

def label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "POSITIVE"
    elif c <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def main():
    # Load the posts
    df = pd.read_csv(INPUT_FILE)

    analyzer = SentimentIntensityAnalyzer()

    # Run sentiment analysis row by row
    compounds = []
    labels = []
    negs = []
    neu = []
    pos = []

    for text in df["text"]:
        scores = analyzer.polarity_scores(str(text))
        compound = scores["compound"]

        compounds.append(compound)
        labels.append(label_from_compound(compound))
        negs.append(scores["neg"])
        neu.append(scores["neu"])
        pos.append(scores["pos"])

    # Add results to dataframe
    df["sent_compound"] = compounds
    df["sent_label"] = labels
    df["sent_neg"] = negs
    df["sent_neu"] = neu
    df["sent_pos"] = pos

    # Save to new CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved sentiment results to: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
