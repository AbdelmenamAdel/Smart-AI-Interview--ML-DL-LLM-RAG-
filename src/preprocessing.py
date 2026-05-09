import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        """Clean and normalize job descriptions."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove symbols and special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove newlines
        text = text.replace('\n', ' ')
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def prepare_data(self, df, text_column='description'):
        """Clean the specified column in the dataframe."""
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        return df

class VectorizerManager:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_vectorizer(self):
        return self.vectorizer
