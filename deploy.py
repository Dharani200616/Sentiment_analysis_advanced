import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import load_model
from src.features import load_vectorizer
from src.preprocess import clean_text

class Predictor:
    def __init__(self, model_path, vectorizer_path):
        self.model = load_model(model_path)
        self.vectorizer = load_vectorizer(vectorizer_path)
    
    def predict(self, text):
        cleaned = clean_text(text)
        features = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0].tolist()
        return {
            "sentiment": "Positive" if prediction == 1 else "Negative",
            "confidence": max(probability)
        }
