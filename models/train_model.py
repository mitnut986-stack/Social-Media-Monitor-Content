"""
Khmer Sentiment ML Model Training Script
Trains a TF-IDF + Logistic Regression / LinearSVC classifier on Khmer sentiment corpus
and serializes the pipeline to `models/khmer_sentiment_model.joblib`.
"""

import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nlp.preprocessor import KhmerPreprocessor
from nlp.sample_data import TRAINING_DATA


def train_and_evaluate():
    print("=" * 60)
    print("🇰🇭 Training Khmer Sentiment Analysis Machine Learning Model")
    print("=" * 60)

    preprocessor = KhmerPreprocessor()
    
    # Tokenize corpus
    print(f"Loaded {len(TRAINING_DATA)} labeled Khmer sentiment samples.")
    texts = [preprocessor.get_tokenized_string(item[0]) for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples.")

    # Build Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=42))
    ])

    print("Fitting TF-IDF Vectorizer and Logistic Regression classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, y_pred, digits=3))

    # Fit on 100% of data for production deployment
    print("Retraining pipeline on full dataset for production artifact...")
    pipeline.fit(texts, labels)

    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "khmer_sentiment_model.joblib")
    
    joblib.dump(pipeline, model_path)
    print(f"✅ Model successfully saved to: {model_path}")
    print("Ready for Streamlit inference!")


if __name__ == "__main__":
    train_and_evaluate()
