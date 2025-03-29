import streamlit as st
import pickle
import os
from src.preprocess import preprocess_text

# Define Sentiment Mapping
sentiment_map = {
    0: ("Negative 😠", "red"),
    1: ("Positive 😊", "green"),
    -1: ("Neutral 🤔", "gray")  # New category for conflicting results
}

# Load Models & Vectorizer
model_path = "models"

with open(os.path.join(model_path, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(model_path, "logistic_regression.pkl"), "rb") as f:
    logistic_regression = pickle.load(f)

with open(os.path.join(model_path, "svm_model.pkl"), "rb") as f:
    svm_model = pickle.load(f)

# Streamlit UI Configuration
st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="centered")

# Sidebar with Info
with st.sidebar:
    st.title("ℹ️ About the App")
    st.write("""
    This app predicts the sentiment of a tweet using:
    - Logistic Regression  
    - Support Vector Machine (SVM)
    """)
    st.markdown("🔗 [View Source Code](https://github.com/your_username/sentiment-analysis)")
    st.markdown("📊 **Confidence** scores represent model certainty.")

# Main Title and Input Area
st.title("🧠 Tweet Sentiment Analyzer")
st.subheader("Analyze the emotional tone of your tweet!")
tweet_input = st.text_area("✏️ Enter your tweet below:", height=120, placeholder="e.g., I love this application!")

# Analyze Sentiment Button
if st.button("🔍 Analyze Sentiment"):
    if tweet_input.strip() == "":
        st.warning("⚠️ Please enter a valid tweet.")
    else:
        # Preprocess and Vectorize Input
        cleaned_tweet = preprocess_text(tweet_input)
        vector = vectorizer.transform([cleaned_tweet])

        # Logistic Regression Prediction
        lr_pred = int(logistic_regression.predict(vector)[0])
        lr_conf = logistic_regression.predict_proba(vector).max()

        # SVM Prediction
        svm_pred = int(svm_model.predict(vector.toarray())[0])
        try:
            svm_conf = abs(svm_model.decision_function(vector.toarray())[0])
        except:
            svm_conf = 0.0  # Default to 0 if unavailable

        # Resolve Conflicting Predictions
        if lr_pred != svm_pred:
            avg_conf = (lr_conf + svm_conf) / 2
            final_pred, color = sentiment_map[-1]  # Neutral
        else:
            avg_conf = (lr_conf + svm_conf) / 2
            final_pred, color = sentiment_map[lr_pred]

        # Clamp the confidence value between 0.0 and 1.0
        clamped_conf = min(avg_conf, 1.0)

        # Display Results
        st.write("### 📊 Prediction Results")
        st.markdown(f"<h3 style='color:{color};'>🔹 Final Sentiment: {final_pred}</h3>", unsafe_allow_html=True)
        st.progress(clamped_conf)  # Show confidence as progress bar
        st.caption(f"🧪 **Overall Confidence Score:** {clamped_conf * 100:.2f}%")

        # Model-Specific Results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Logistic Regression")
            st.markdown(f"**Prediction:** {sentiment_map[lr_pred][0]}")
            st.progress(lr_conf)
            st.caption(f"Confidence: {lr_conf * 100:.2f}%")

        with col2:
            st.markdown("#### SVM")
            st.markdown(f"**Prediction:** {sentiment_map[svm_pred][0]}")
            st.progress(min(svm_conf, 1.0))  # Normalize confidence
            st.caption(f"Confidence: {svm_conf * 100:.2f}%")

        st.success("✅ Sentiment Analysis Complete!")
