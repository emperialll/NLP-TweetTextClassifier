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
from sklearn.model_selection import GridSearchCV


# Download NLTK resources (run once)

# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

# Sample dataset (in real cases, load from a CSV or database)
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

# Initialize models
logistic_model = LogisticRegression(
    multi_class='multinomial', max_iter=1000, random_state=42)
naive_bayes_model = MultinomialNB()
svm_model = LinearSVC(multi_class='ovr', random_state=42)

# Dictionary to store models and their names
models = {
    'Logistic Regression': logistic_model,
    'Naive Bayes': naive_bayes_model,
    'Linear SVM': svm_model
}

# Perform 10-fold cross-validation for each model
for model_name, model in models.items():
    # Train the model on the training set
    model.fit(X_train, y_train)

    # Evaluate with 10-fold cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=10, scoring='accuracy')


# Evaluate models on the test set
for model_name, model in models.items():
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report (includes precision, recall, F1-score)
    report = classification_report(y_test, y_pred, target_names=[
                                   'complaint', 'complement', 'request'])


# Define parameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1.0, 10.0],  # Inverse of regularization strength
        'max_iter': [1000]
    },
    'Naive Bayes': {
        'alpha': [0.1, 0.5, 1.0]  # Smoothing parameter
    },
    'Linear SVM': {
        'C': [0.1, 1.0, 10.0]  # Regularization parameter
    }
}

# Also tune TF-IDF max_features
tfidf_param_grid = {
    'max_features': [100, 500, 1000]
}

# Store best models
best_models = {}

# Tune TF-IDF and each model
for max_features in tfidf_param_grid['max_features']:
    print(f"\nTuning with TF-IDF max_features={max_features}")

    # Re-run TF-IDF with different max_features
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tune each model
    for model_name, model in models.items():
        print(f"  Tuning {model_name}...")
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=10,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Store best model
        best_models[f"{model_name}_max_features_{max_features}"] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        # Print best parameters and cross-validation score
        print(f"    Best parameters: {grid_search.best_params_}")
        print(
            f"    Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate best models on test set
for key, info in best_models.items():
    model = info['model']
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
                                   'complaint', 'complement', 'request'])
    print(f"\nTest Set Results for {key}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
