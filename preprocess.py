import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    """
    Advanced text cleaning:
    - Lowercase
    - Remove special characters
    - Handle negations (optional logic can be added here)
    - Lemmatization
    - Stopword removal
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return " ".join(cleaned_tokens)

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    if 'review' in df.columns:
        df['cleaned_review'] = df['review'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
