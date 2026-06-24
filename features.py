import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def extract_tfidf_features(df, text_column, max_features=5000, save_path=None):
    """
    Extract TF-IDF features from text.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    features = vectorizer.fit_transform(df[text_column])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(vectorizer, save_path)
        print(f"Vectorizer saved to {save_path}")
        
    return features, vectorizer

def load_vectorizer(path):
    return joblib.load(path)
