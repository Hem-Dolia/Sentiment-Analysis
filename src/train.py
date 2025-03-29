import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from preprocess import preprocess_text

# Load dataset
df = pd.read_csv("../data/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['sentiment', 'tweet']
#df = df.sample(n=5000, random_state=42)  # Use only 5000 tweets for testing


# Convert sentiment (0 = negative, 4 = positive)
df = df[df["sentiment"] != 2]  # Remove neutral if any
df["sentiment"] = df["sentiment"].apply(lambda x: 0 if x == 0 else 1)

# Sample data to speed up training (optional)
df = df.sample(n=5000, random_state=42)  # Reduce for quick testing
print("data loaded")

# Preprocess tweets
df["cleaned_tweet"] = df["tweet"].apply(preprocess_text)
print("data cleaned")

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_tweet"]).toarray()
y = df["sentiment"]

print("data vectorised")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

print("trained logistic")
# Train SVM
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# Save models
with open("../models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("../models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Models trained and saved!")
