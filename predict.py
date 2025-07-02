import joblib
import os
from preprocess import preprocess_text
import numpy as np

# Load the saved model, vectorizer, and scaler
model = joblib.load('best_text_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl') if 'scaler.pkl' in os.listdir() else None


def classify_tweet(tweet, vectorizer, model, scaler=None):
    try:
        cleaned_tweet = preprocess_text(tweet)
        tweet_vector = vectorizer.transform([cleaned_tweet]).toarray()
        tweet_vector = np.clip(tweet_vector, -1, 1)
        if scaler:
            tweet_vector = scaler.transform(tweet_vector)
        prediction = model.predict(tweet_vector)[0]
        return prediction
    except Exception as e:
        return f"Error classifying tweet: {str(e)}"


# Example tweets
new_tweets = [
    "I love the new update, it's so smooth and intuitive!",
    "Why does it crash every time I open it? So frustrating.",
    "Is there a way to change my email address on file?",
    "Great customer support—quick and helpful responses!",
    "This feature is useless now, bring back the old version."
]

print("Classifying New Tweets:")
for tweet in new_tweets:
    prediction = classify_tweet(tweet, vectorizer, model, scaler)
    print(f"Tweet: {tweet}")
    print(f"Predicted Category: {prediction}\n")
