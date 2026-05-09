import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class JobClassifier:
    def __init__(self, model_type='logistic_regression'):
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Unsupported model type. Choose 'logistic_regression' or 'random_forest'.")
        
        self.model_type = model_type

    def train(self, X, y):
        """Train the model on the provided features and labels."""
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Predict labels for the given features."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def save_model(self, model_path, vectorizer, vectorizer_path):
        """Save both the model and the vectorizer."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    @staticmethod
    def load_model(model_path, vectorizer_path):
        """Load a saved model and vectorizer."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
