# Smart AI Interview Project 🚀

A professional AI-powered system designed to classify job descriptions and automatically retrieve relevant technical interview questions (MCQs and Essays).

## 📌 Project Overview
This project leverages Machine Learning (Logistic Regression) to analyze job descriptions and map them to specific tech roles. Once a role is predicted, the system fetches a curated set of interview questions from a structured knowledge base, helping recruiters and candidates prepare for technical interviews.

### ✨ Key Features
- **Job Classification**: Predicts roles like Full Stack Developer, Data Scientist, DevOps Engineer, etc.
- **Automated Question Retrieval**: Fetches relevant MCQs and Essay questions based on the predicted role.
- **Modular Codebase**: Organized into clear Python scripts for preprocessing, training, and logic.
- **Interactive Gradio Demo**: A user-friendly web interface for real-time analysis.

## 📂 Repository Structure
```
Smart_AI_Interview/
├── data/               # CSV datasets (Job Descriptions, MCQs, Essays)
├── models/             # Trained model and vectorizer artifacts (.pkl)
├── src/                # Modular Python scripts
│   ├── preprocessing.py   # Text cleaning and TF-IDF logic
│   ├── model_trainer.py   # Model training and saving
│   └── interview_logic.py # Question mapping and retrieval logic
├── app.py              # Gradio web application
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
└── Smart_AI_Interview.ipynb # Structured project workflow notebook
```

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training the Model
To train the initial model and generate the necessary artifacts in the `models/` folder:
```bash
python3 train_initial.py
```

### 3. Running the Gradio Demo
Launch the interactive web interface:
```bash
python3 app.py
```

### 4. Exploring the Workflow
Open `Smart_AI_Interview.ipynb` to see a step-by-step walkthrough of the data science pipeline using the modular code.

## 🛠️ Built With
- **Python**: Core logic.
- **Scikit-Learn**: TF-IDF and Logistic Regression.
- **Pandas**: Data manipulation.
- **Gradio**: Interactive web application.

## 📝 License
This project is for academic purposes as part of an AI Career project submission.
