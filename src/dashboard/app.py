import pandas as pd
import streamlit as st
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

# ========== BASIC PAGE CONFIG & STYLE ==========
st.set_page_config(
    page_title="Social Media Public Opinion Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .big-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">📊 Social Media Public Opinion Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a CSV with social media posts, choose the text column, '
    'and explore sentiment, trends, and key topics.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ========== FILE UPLOAD ==========
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin (max 200MB).")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded CSV is empty.")
    st.stop()

st.subheader("Step 1: Choose columns")
st.write("Preview of the uploaded data:")
st.dataframe(df.head())

# Helper: hide obvious index columns
def filter_real_columns(columns):
    return [c for c in columns if not c.lower().startswith("unnamed")]

real_columns = filter_real_columns(list(df.columns))

# --- Select text column ---
if not real_columns:
    st.error("No valid columns found (only 'Unnamed' columns detected).")
    st.stop()

text_col = st.selectbox(
    "Select the column that contains the text/posts:",
    options=real_columns,
)

# --- Optional: select label column for evaluation / ML model ---
label_options = ["None (no ground truth labels)"] + real_columns
label_col = st.selectbox(
    "Optional: select the column with TRUE sentiment labels (for evaluation / ML model):",
    options=label_options,
)

# --- Optional: timestamp column for time trends ---
time_options = ["None (no timestamp)"] + real_columns
time_col = st.selectbox(
    "Optional: select timestamp/date column for time trends:",
    options=time_options,
)

# --- Model choice ---
st.subheader("Step 2: Choose sentiment engine")
model_choice = st.radio(
    "Sentiment model:",
    options=[
        "VADER (lexicon-based, fast)",
        "Logistic Regression (ML model trained on uploaded labels)",
    ],
    help=(
        "VADER uses a predefined sentiment lexicon.\n"
        "The ML model learns from your labelled column (if provided)."
    ),
)

run_button = st.button("🚀 Run Sentiment Analysis")

if not run_button:
    st.stop()

# ========== SENTIMENT FUNCTIONS ==========
def vader_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    compounds, labels, negs, neus, poss = [], [], [], [], []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        c = scores["compound"]
        if c >= 0.05:
            lab = "POSITIVE"
        elif c <= -0.05:
            lab = "NEGATIVE"
        else:
            lab = "NEUTRAL"
        compounds.append(c)
        labels.append(lab)
        negs.append(scores["neg"])
        neus.append(scores["neu"])
        poss.append(scores["pos"])
    return compounds, labels, negs, neus, poss


def ml_sentiment(texts, labels_true):
    """
    Train a simple TF-IDF + Logistic Regression model on the provided labels,
    then predict sentiment for all texts.
    """
    # Clean labels
    def clean_label(x):
        if isinstance(x, str):
            return x.strip().lower()
        return x

    y = labels_true.apply(clean_label)

    mapping = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
    }
    y = y.map(mapping)
    # Drop rows with unknown labels
    mask = y.notna()
    texts_train = texts[mask]
    y_train = y[mask]

    if len(y_train.unique()) < 2:
        raise ValueError("Need at least 2 different classes in label column to train ML model.")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(texts_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict on all texts
    X_all = vectorizer.transform(texts)
    y_pred = clf.predict(X_all)

    # Map back to POS/NEU/NEG labels used elsewhere
    label_map = {
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
    }
    pred_labels = [label_map.get(lbl, "NEUTRAL") for lbl in y_pred]

    # For ML model we won't produce fine-grained scores, but we can fake compound from probabilities if needed
    # Here, just use 0.0 as placeholder compound and pos/neu/neg probabilities are not shown.
    compounds = [0.0] * len(pred_labels)
    negs = neus = poss = [0.0] * len(pred_labels)

    return compounds, pred_labels, negs, neus, poss


# ========== RUN ANALYSIS ==========
texts = df[text_col].astype(str).fillna("")

st.subheader("Step 3: Running analysis...")
with st.spinner("Analysing sentiment, please wait..."):

    use_ml = model_choice.startswith("Logistic Regression")
    if use_ml:
        if label_col == "None (no ground truth labels)":
            st.warning("You selected the ML model but no label column. Falling back to VADER.")
            use_ml = False
        else:
            try:
                compounds, labels_pred, negs, neus, poss = ml_sentiment(texts, df[label_col])
            except Exception as e:
                st.warning(f"Could not train ML model ({e}). Falling back to VADER.")
                use_ml = False

    if not use_ml:
        compounds, labels_pred, negs, neus, poss = vader_sentiment(texts)

    result_df = df.copy()
    result_df["sent_compound"] = compounds
    result_df["sent_label"] = labels_pred
    result_df["sent_neg"] = negs
    result_df["sent_neu"] = neus
    result_df["sent_pos"] = poss

st.success("Analysis complete!")

# ========== OVERALL SUMMARY ==========
st.markdown("### 🎯 Overall Sentiment Summary")

total_posts = len(result_df)
pos = (result_df["sent_label"] == "POSITIVE").sum()
neu = (result_df["sent_label"] == "NEUTRAL").sum()
neg = (result_df["sent_label"] == "NEGATIVE").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Posts", total_posts)
c2.metric("Positive 😊", pos)
c3.metric("Neutral 😐", neu)
c4.metric("Negative 😡", neg)

st.markdown("### 📊 Sentiment Distribution")
sent_counts = result_df["sent_label"].value_counts().reindex(
    ["POSITIVE", "NEUTRAL", "NEGATIVE"]
).fillna(0)
st.bar_chart(sent_counts)

# ========== TIME TREND (OPTIONAL) ==========
if time_col != "None (no timestamp)" and time_col in result_df.columns:
    st.markdown("### ⏱ Sentiment Over Time")
    time_data = result_df.copy()
    time_data[time_col] = pd.to_datetime(time_data[time_col], errors="coerce")
    time_data = time_data.dropna(subset=[time_col])
    if not time_data.empty:
        time_data["date_only"] = time_data[time_col].dt.date
        daily = (
            time_data.groupby("date_only")["sent_label"]
            .value_counts()
            .unstack(fill_value=0)
        )
        st.line_chart(daily)
    else:
        st.info("Could not parse any valid dates from the selected timestamp column.")

# ========== TOPIC / KEYWORD EXTRACTION ==========
st.markdown("### 🔑 Top Keywords by Sentiment")

def top_keywords_for_label(df_in, label_value, n=10):
    subset = df_in[df_in["sent_label"] == label_value]
    if subset.empty:
        return pd.DataFrame({"keyword": [], "count": []})
    vec = CountVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(subset[text_col].astype(str))
    sums = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    top_indices = sums.argsort()[::-1][:n]
    return pd.DataFrame(
        {"keyword": vocab[top_indices], "count": sums[top_indices]}
    )

col_pos, col_neu, col_neg = st.columns(3)

with col_pos:
    st.write("**Positive 😊**")
    top_pos = top_keywords_for_label(result_df, "POSITIVE")
    st.dataframe(top_pos)

with col_neu:
    st.write("**Neutral 😐**")
    top_neu = top_keywords_for_label(result_df, "NEUTRAL")
    st.dataframe(top_neu)

with col_neg:
    st.write("**Negative 😡**")
    top_neg = top_keywords_for_label(result_df, "NEGATIVE")
    st.dataframe(top_neg)

# ========== EXAMPLE POSTS ==========
st.markdown("### 💬 Example Posts")

sentiment_filter = st.selectbox(
    "Filter by predicted sentiment:",
    options=["ALL", "POSITIVE", "NEUTRAL", "NEGATIVE"],
)

filtered_df = result_df.copy()
if sentiment_filter != "ALL":
    filtered_df = filtered_df[filtered_df["sent_label"] == sentiment_filter]

st.write("Showing up to 20 posts:")
cols_to_show = [c for c in [text_col, "sent_label"] if c in filtered_df.columns]
if label_col != "None (no ground truth labels)" and label_col in filtered_df.columns:
    cols_to_show.append(label_col)

st.dataframe(filtered_df[cols_to_show].head(20))

# ========== EVALUATION (IF LABELS PROVIDED) ==========
if label_col != "None (no ground truth labels)" and label_col in result_df.columns:
    st.markdown("### 🧪 Model Evaluation (vs Ground Truth Labels)")

    def clean_label(x):
        if isinstance(x, str):
            return x.strip().lower()
        return x

    y_true = result_df[label_col].apply(clean_label)
    y_pred = result_df["sent_label"].apply(lambda x: x.lower())

    mapping = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
    }
    y_true = y_true.map(mapping)
    y_pred = y_pred.map(mapping)

    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        st.write(f"**Accuracy:** `{acc:.3f}`")

        st.write("**Classification Report:**")
        st.text(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
        cm_df = pd.DataFrame(
            cm,
            index=["true_negative", "true_neutral", "true_positive"],
            columns=["pred_negative", "pred_neutral", "pred_positive"],
        )
        st.write("**Confusion Matrix:**")
        st.dataframe(cm_df)
    else:
        st.info("Not enough valid labels to compute evaluation metrics.")

# ========== DOWNLOAD ==========
st.markdown("### 📥 Download Results")

csv_data = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV with Sentiment Columns",
    data=csv_data,
    file_name="sentiment_results.csv",
    mime="text/csv",
)
