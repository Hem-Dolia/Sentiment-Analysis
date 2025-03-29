import pickle
from preprocess import preprocess_text

# Load saved models and vectorizer
with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("../models/logistic_regression.pkl", "rb") as f:
    logistic_regression = pickle.load(f)

with open("../models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

print("Enter a tweet (or type 'exit'):")

while True:
    tweet = input()
    if tweet.lower() == "exit":
        break

    cleaned_tweet = preprocess_text(tweet)
    tweet_vector = vectorizer.transform([cleaned_tweet])

    # Predictions
    lr_pred = logistic_regression.predict(tweet_vector)[0]
    lr_conf = logistic_regression.predict_proba(tweet_vector).max()

    svm_pred = svm_model.predict(tweet_vector)[0]
    try:
        svm_conf = abs(svm_model.decision_function(tweet_vector)[0])
    except:
        svm_conf = "N/A"

    sentiment_map = {0: "Negative", 4: "Positive"}

    result = {
        "Input Tweet": tweet,
        "Logistic Regression": f"{sentiment_map[lr_pred]} ({lr_conf:.2f})",
        "SVM": f"{sentiment_map[svm_pred]} ({svm_conf:.2f})" if svm_conf != "N/A" else sentiment_map[svm_pred]
    }

    print(result)
