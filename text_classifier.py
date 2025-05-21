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
        # Complaints (10 samples)
        "Your service is terrible!",
        "This is the worst experience ever.",
        "Awful customer support.",
        "Not happy with the delay.",
        "Your app keeps crashing, fix it!",
        "Terrible product quality.",
        "Customer service is a nightmare.",
        "Delivery was late again.",
        "Completely disappointed with my order.",
        "Why is your service so bad?",
        # Complements (10 samples)
        "Love your quick response!",
        "Great job, keep it up!",
        "Amazing service, thank you!",
        "Fantastic support team!",
        "Really impressed with your product!",
        "Best customer service ever!",
        "Super happy with my purchase!",
        "Your app is awesome!",
        "Thanks for the great experience!",
        "You guys rock!",
        # Requests (10 samples)
        "Can you add more payment options?",
        "Please fix the app crashing issue.",
        "Could you help with my account?",
        "Can you provide a refund option?",
        "Please add more features to the app.",
        "Could you expedite my order?",
        "Can you clarify your return policy?",
        "Please send me a replacement part.",
        "Can you update the delivery status?",
        "Could you offer a discount code?"
    ],
    'label': [
        'complaint', 'complaint', 'complaint', 'complaint', 'complaint',
        'complaint', 'complaint', 'complaint', 'complaint', 'complaint',
        'complement', 'complement', 'complement', 'complement', 'complement',
        'complement', 'complement', 'complement', 'complement', 'complement',
        'request', 'request', 'request', 'request', 'request',
        'request', 'request', 'request', 'request', 'request'
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

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Print the sizes of the splits
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Note: 10-fold cross-validation will be implemented during model training
