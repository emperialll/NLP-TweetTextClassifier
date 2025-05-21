import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


# Download NLTK resources (run once)

# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

# Sample dataset (in real cases, load from a CSV or database)
data = {
    'tweet': [
        "Your service is terrible!",
        "Love your quick response!",
        "Can you add more payment options?",
        "This is the worst experience ever.",
        "Great job, keep it up!",
        "Please fix the app crashing issue.",
        "Awful customer support.",
        "Amazing service, thank you!",
        "Could you help with my account?",
        "Not happy with the delay."
    ],
    'label': [
        'complaint',
        'complement',
        'request',
        'complaint',
        'complement',
        'request',
        'complaint',
        'complement',
        'request',
        'complaint'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing function


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a string
    return ' '.join(tokens)


# Apply preprocessing to the tweet column
df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()
y = df['label']

# Print sample of cleaned data and TF-IDF features
print("Sample cleaned tweets:\n", df[['tweet', 'cleaned_tweet']].head())
print("\nTF-IDF feature shape:", X.shape)
