# 🧠 Tweet Sentiment Analyzer

A web app to analyze the sentiment of tweets using Logistic Regression and Support Vector Machines (SVM). Built using Python, Scikit-learn, NLTK, and Streamlit.

---

## 🚀 Features

- Preprocess tweets (lowercase, punctuation removal, stopwords, lemmatization)
- Predict sentiment using Logistic Regression and SVM
- Handles conflicting predictions with a "Neutral 🤔" label
- Displays confidence scores using progress bars
- Streamlit GUI with sidebar project info

---

## 📁 Project Structure

```
Sentiment-Analysis/
├── data/
│   └── (your training data - NOT pushed to GitHub)
├── models/
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   └── vectorizer.pkl
├── src/
│   └── preprocess.py
├── sentiment_gui.py
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Hem-Dolia/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Run the App

```bash
streamlit run sentiment_gui.py
```

Then open the provided local URL in your browser (e.g., http://localhost:8501)

---

## 📊 Models Used

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **TF-IDF Vectorizer**

All models are pre-trained and stored in the `/models/` directory.

---

## ✨ Example Predictions

| Tweet                       | Logistic | SVM    | Final Prediction |
|----------------------------|----------|--------|------------------|
| `I love this app`          | Positive | Positive | 😊 Positive     |
| `Not sure how I feel`      | Negative | Negative | 😠 Negative     |
| `App is nice but buggy`    | Positive | Negative | 🤔 Neutral      |

---

## 📦 Requirements

Make sure you have Python 3.8+ installed.

Contents of `requirements.txt`:

```txt
streamlit
scikit-learn
nltk
```

---

## 📝 Acknowledgments

- [NLTK](https://www.nltk.org/) for preprocessing
- [Scikit-learn](https://scikit-learn.org/) for modeling
- [Streamlit](https://streamlit.io/) for frontend


> 🚫 **Note**: Large datasets are not pushed due to GitHub's file size limit. You can download datasets separately and retrain if needed.
