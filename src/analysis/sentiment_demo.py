from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

examples = [
    "I love this new movie! It's amazing.",
    "This is the worst experience ever.",
    "I don't know how to feel about this."
]

print("Sentiment Results:\n")

for text in examples:
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    print(f"Text: {text}")
    print(f"Label: {label} | Scores: {scores}\n")
