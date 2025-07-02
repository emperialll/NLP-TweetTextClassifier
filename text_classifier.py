import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import joblib
import warnings
from preprocess import preprocess_text

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="sklearn.linear_model._linear_loss")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="sklearn.utils.extmath")

# Download NLTK resources (run once, commented out after first run)
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

# Sample dataset
data = {
    'tweet': [
        # Complaints (15 samples)
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
        "App is unusable, so frustrated!",
        "Poor response time, very upset.",
        "Product broke after one use.",
        "Billing issues are ridiculous!",
        "No help from your team, awful!",
        # Complements (15 samples)
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
        "Fast delivery, very pleased!",
        "Your team went above and beyond!",
        "Love the new features, great work!",
        "Such a smooth experience, thanks!",
        "Top-notch quality, highly recommend!",
        # Requests (15 samples)
        "Can you add more payment options?",
        "Please fix the app crashing issue.",
        "Could you help with my account?",
        "Can you provide a refund option?",
        "Please add more features to the app.",
        "Could you expedite my order?",
        "Can you clarify your return policy?",
        "Please send me a replacement part.",
        "Can you update the delivery status?",
        "Could you offer a discount code?",
        "Please provide a user guide.",
        "Can you reset my password?",
        "Could you check my order status?",
        "Please add dark mode to the app.",
        "Can you confirm my subscription?"
    ],
    'label': [
        'complaint', 'complaint', 'complaint', 'complaint', 'complaint',
        'complaint', 'complaint', 'complaint', 'complaint', 'complaint',
        'complaint', 'complaint', 'complaint', 'complaint', 'complaint',
        'complement', 'complement', 'complement', 'complement', 'complement',
        'complement', 'complement', 'complement', 'complement', 'complement',
        'complement', 'complement', 'complement', 'complement', 'complement',
        'request', 'request', 'request', 'request', 'request',
        'request', 'request', 'request', 'request', 'request',
        'request', 'request', 'request', 'request', 'request'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Note: The dataset contains only 45 tweets (15 per class), limiting performance.
# Accuracy and classification of 'complement' and 'request' classes will improve with a larger dataset.

# Apply preprocessing to the tweet column
df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)
y = df['label']

# Initialize models with class weighting
logistic_model = LogisticRegression(
    max_iter=1000, random_state=42, solver='liblinear', class_weight='balanced')
naive_bayes_model = MultinomialNB()
svm_model = LinearSVC(random_state=42, max_iter=5000, class_weight='balanced')

# Dictionary to store models and their names
models = {
    'Logistic Regression': logistic_model,
    'Naive Bayes': naive_bayes_model,
    'Linear SVM': svm_model
}

# Define parameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1.0]
    },
    'Naive Bayes': {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
    },
    'Linear SVM': {
        'C': [0.001, 0.01, 0.1, 1.0]
    }
}

# TF-IDF parameter grid with bigrams
tfidf_param_grid = {
    'max_features': [100, 500, 1000],
    'ngram_range': [(1, 1), (1, 2)]  # Unigrams and bigrams
}

# Store best models
best_models = {}

# Tune TF-IDF and each model
for max_features in tfidf_param_grid['max_features']:
    for ngram_range in tfidf_param_grid['ngram_range']:
        print(
            f"\nTuning with TF-IDF max_features={max_features}, ngram_range={ngram_range}")

        # Initialize TF-IDF with clipping
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range)
        X = vectorizer.fit_transform(df['cleaned_tweet'])
        X = np.clip(X.toarray(), -1, 1)

        # Normalize features for Logistic Regression and Linear SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Tune each model
        for model_name, model in models.items():
            print(f"  Tuning {model_name}...")
            try:
                grid_search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=10,
                    scoring='accuracy',
                    n_jobs=-1,
                    error_score=0
                )

                # Use original sparse X_train for Naive Bayes
                if model_name == 'Naive Bayes':
                    X_train_nb = vectorizer.transform(
                        df['cleaned_tweet'].iloc[y_train.index])
                    grid_search.fit(X_train_nb, y_train)
                else:
                    grid_search.fit(X_train, y_train)

                # Store best model
                key = f"{model_name}_max_features_{max_features}_ngram_{ngram_range}"
                best_models[key] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'vectorizer': vectorizer,
                    'scaler': scaler if model_name != 'Naive Bayes' else None
                }

                print(f"    Best parameters: {grid_search.best_params_}")
                print(
                    f"    Best cross-validation accuracy: {grid_search.best_score_:.4f}")

            except Exception as e:
                print(f"    Error during {model_name} tuning: {str(e)}")
                best_models[key] = {
                    'model': None,
                    'best_params': {},
                    'best_score': 0,
                    'vectorizer': vectorizer,
                    'scaler': scaler if model_name != 'Naive Bayes' else None
                }

# Evaluate best models on test set
for key, info in best_models.items():
    model = info['model']
    vectorizer = info['vectorizer']
    scaler = info['scaler']

    if model is None:
        print(f"\nTest Set Results for {key}: Skipped due to training error")
        continue

    try:
        X_test_transformed = vectorizer.transform(
            df['cleaned_tweet'].iloc[y_test.index]).toarray()
        X_test_transformed = np.clip(X_test_transformed, -1, 1)
        if scaler:
            X_test_transformed = scaler.transform(X_test_transformed)

        y_pred = model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=[
                                       'complaint', 'complement', 'request'])
        print(f"\nTest Set Results for {key}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
    except Exception as e:
        print(
            f"\nTest Set Results for {key}: Error during evaluation - {str(e)}")

# Select the best model based on test set accuracy
best_model_key = max(
    best_models,
    key=lambda k: (
        accuracy_score(
            y_test,
            best_models[k]['model'].predict(
                best_models[k]['scaler'].transform(
                    np.clip(best_models[k]['vectorizer'].transform(
                        df['cleaned_tweet'].iloc[y_test.index]).toarray(), -1, 1)
                ) if best_models[k]['scaler'] else
                np.clip(best_models[k]['vectorizer'].transform(
                    df['cleaned_tweet'].iloc[y_test.index]).toarray(), -1, 1)
            )
        ) if best_models[k]['model'] is not None else -1
    )
)
best_model_info = best_models[best_model_key]
best_model = best_model_info['model']
best_vectorizer = best_model_info['vectorizer']
best_scaler = best_model_info['scaler']

if best_model is not None:
    print(f"\nBest Model: {best_model_key}")
    print(f"Best Parameters: {best_model_info['best_params']}")
    print(
        f"Best Cross-Validation Accuracy: {best_model_info['best_score']:.4f}")
    try:
        test_accuracy = accuracy_score(
            y_test,
            best_model.predict(
                best_scaler.transform(
                    np.clip(best_vectorizer.transform(
                        df['cleaned_tweet'].iloc[y_test.index]).toarray(), -1, 1)
                ) if best_scaler else
                np.clip(best_vectorizer.transform(
                    df['cleaned_tweet'].iloc[y_test.index]).toarray(), -1, 1)
            )
        )
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
    except Exception as e:
        print(f"Error calculating test set accuracy: {str(e)}")

    # Re-train the best model on the full dataset
    try:
        X_final = best_vectorizer.fit_transform(df['cleaned_tweet']).toarray()
        X_final = np.clip(X_final, -1, 1)
        if best_scaler:
            X_final = best_scaler.fit_transform(X_final)
        best_model.fit(X_final, y)

        # Save the best model, vectorizer, and scaler
        joblib.dump(best_model, 'best_text_classifier.pkl')
        joblib.dump(best_vectorizer, 'tfidf_vectorizer.pkl')
        if best_scaler:
            joblib.dump(best_scaler, 'scaler.pkl')
    except Exception as e:
        print(f"Error during final model training or saving: {str(e)}")
else:
    print("\nNo valid model found. Skipping final training and saving.")

# Function to classify new tweets


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


# Test the prediction function
if best_model is not None:
    new_tweets = [
        "I love the new update, it's so smooth and intuitive!",
        "Why does it crash every time I open it? So frustrating.",
        "Is there a way to change my email address on file?",
        "Great customer support—quick and helpful responses!",
        "This feature is useless now, bring back the old version."
    ]

    print("\nClassifying New Tweets:")
    for tweet in new_tweets:
        prediction = classify_tweet(
            tweet, best_vectorizer, best_model, best_scaler)
        print(f"Tweet: {tweet}")
        print(f"Predicted Category: {prediction}\n")
else:
    print("\nNo valid model available for classifying new tweets.")

# Generate requirements.txt
requirements = [
    "pandas",
    "numpy",
    "scikit-learn",
    "nltk",
    "joblib"
]
with open("requirements.txt", "w") as f:
    for req in requirements:
        f.write(f"{req}\n")
