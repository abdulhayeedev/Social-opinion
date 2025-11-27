import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "sample_posts_with_sentiment.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    total = len(df)
    pos = (df["sent_label"] == "POSITIVE").sum()
    neg = (df["sent_label"] == "NEGATIVE").sum()
    neu = (df["sent_label"] == "NEUTRAL").sum()

    def pct(x): 
        return round(100 * x / total, 1) if total > 0 else 0.0

    print(f"\nTotal posts: {total}\n")
    print(f"POSITIVE: {pos} ({pct(pos)}%)")
    print(f"NEGATIVE: {neg} ({pct(neg)}%)")
    print(f"NEUTRAL : {neu} ({pct(neu)}%)\n")

    # Simple auto-generated summary text (you can reuse in your report)
    if pos > neg:
        overall = "mostly positive"
    elif neg > pos:
        overall = "mostly negative"
    else:
        overall = "mixed / balanced"

    print("Auto-summary:")
    print(
        f"Overall public opinion in this sample is {overall}. "
        f"{pct(pos)}% of posts are positive, {pct(neg)}% negative, and {pct(neu)}% neutral."
    )

if __name__ == "__main__":
    main()
