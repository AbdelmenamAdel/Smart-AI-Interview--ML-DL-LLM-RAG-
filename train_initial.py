import pandas as pd
from src.preprocessing import TextPreprocessor, VectorizerManager
from src.model_trainer import JobClassifier
import os

def train_initial_model():
    print("Loading data...")
    df = pd.read_csv('data/job_title_des.csv')
    
    # Drop NaNs
    df = df.dropna(subset=['Job Description', 'Job Title'])
    
    print("Preprocessing...")
    preprocessor = TextPreprocessor()
    df = preprocessor.prepare_data(df, text_column='Job Description')
    
    print("Vectorizing...")
    v_manager = VectorizerManager(max_features=5000)
    X = v_manager.fit_transform(df['cleaned_text'])
    y = df['Job Title']
    
    print("Training Model...")
    classifier = JobClassifier(model_type='logistic_regression')
    classifier.train(X, y)
    
    print("Saving artifacts...")
    classifier.save_model(
        'models/job_classifier.pkl', 
        v_manager.get_vectorizer(), 
        'models/tfidf_vectorizer.pkl'
    )
    print("Done!")

if __name__ == "__main__":
    train_initial_model()
