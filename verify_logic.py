import pandas as pd
from src.preprocessing import TextPreprocessor
from src.model_trainer import JobClassifier
from src.interview_logic import InterviewLogic

def verify_system():
    print("Initializing components...")
    preprocessor = TextPreprocessor()
    model_path = 'models/job_classifier.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    if not os.path.exists(model_path):
        print("Error: Model file not found. Run train_initial.py first.")
        return

    model, vectorizer = JobClassifier.load_model(model_path, vectorizer_path)
    interview_logic = InterviewLogic('data/mcq_questions.csv', 'data/essay_questions.csv')

    test_des = "We are looking for a Software Engineer with experience in Java and SQL."
    print(f"\nTesting with Description: {test_des}")
    
    cleaned = preprocessor.clean_text(test_des)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    
    print(f"Predicted Role: {prediction}")
    
    questions = interview_logic.get_questions(prediction, num_mcq=2, num_essay=1)
    print("\nSample Questions:")
    print(interview_logic.format_questions(questions))
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    import os
    verify_system()
