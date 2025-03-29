import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Download stopwords and wordnet
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags, and non-alphabetic characters
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stopwords.words("english")
    ]

    return " ".join(cleaned_tokens)
