# 📊 Social Media Public Opinion Dashboard

An interactive Streamlit-based web application for analyzing public sentiment from social media text data.  
Users can upload CSV files, run sentiment analysis using multiple models, explore trends over time, extract key topics, and export enriched results.

---

## 🚀 Features

### ✓ **CSV Upload & Column Selection**
- Upload any CSV file containing text data.
- Select which column contains the social media posts.
- Optional column selections:
  - Ground-truth sentiment labels (for evaluation)
  - Timestamp column (for time-series analysis)

### ✓ **Two Sentiment Engines**
1. **VADER (Lexicon-Based Model)**
   - Fast and easy to use
   - Works well for short informal texts such as tweets or comments

2. **Logistic Regression (Machine Learning Model)**
   - Trains on user-provided sentiment labels
   - Uses TF-IDF features for text classification
   - Provides a more adaptive alternative to rule-based models

### ✓ **Visual Insights**
- Sentiment distribution bar chart  
- Time trend line chart (daily sentiment patterns)  
- Example posts browser with category filter  
- Evaluation results (accuracy, precision, recall, F1-score)  
- Confusion matrix heatmap (when ground-truth labels are available)

### ✓ **Topic Extraction**
For each sentiment category (Positive, Neutral, Negative):

- Extracts the most common keywords & phrases  
- Helps understand key themes behind each sentiment group  

### ✓ **Data Export**
- Download a CSV enriched with additional sentiment columns:
  - `sent_label`
  - `sent_compound`
  - `sent_neg`
  - `sent_neu`
  - `sent_pos`

---

## 🧩 Project Structure

SocialOpinionFYP/
│
├── src/
│ ├── analysis/
│ │ ├── sentiment_batch.py
│ │ ├── sentiment_summary.py
│ │ └── evaluate_sentiment.py
│ │
│ └── dashboard/
│ └── app.py
│
├── data/
│ └── (User-uploaded CSV files – ignored in Git)
│
├── .gitignore
└── README.md

yaml
Copy code

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/social-opinion-fyp.git
cd social-opinion-fyp
2. Create and activate a virtual environment
Windows
bash
Copy code
python -m venv .venv
.\.venv\Scripts\activate
macOS / Linux
bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
3. Install required dependencies
bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt, generate one:

bash
Copy code
pip freeze > requirements.txt
▶️ Running the Dashboard
Start the Streamlit app:

bash
Copy code
streamlit run src/dashboard/app.py
Then open the URL (usually auto-generated):

arduino
Copy code
http://localhost:8501
📂 How to Use
Upload a CSV file

Select:

Text column

Optional: True sentiment label column

Optional: Timestamp column

Choose a model:

VADER

Logistic Regression (if labels available)

Click Run Sentiment Analysis

View:

Sentiment stats

Charts

Keyword insights

Evaluation metrics

Example posts

Download processed results

🤝 Contributing
Contributions, improvements, and feature requests are welcome!
Please open an issue or submit a pull request.

📜 License
This project is issued under the MIT License.
You are free to use, modify, and distribute the software.

⭐ Acknowledgements
VADER Sentiment Analyzer

scikit-learn

Streamlit

Pandas & NumPy