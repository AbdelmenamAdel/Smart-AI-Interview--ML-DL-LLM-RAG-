import gradio as gr
import os
import pandas as pd
from src.preprocessing import TextPreprocessor
from src.model_trainer import JobClassifier
from src.interview_logic import InterviewLogic

# Load components
preprocessor = TextPreprocessor()
model, vectorizer = JobClassifier.load_model('models/job_classifier.pkl', 'models/tfidf_vectorizer.pkl')
interview_logic = InterviewLogic('data/mcq_questions.csv', 'data/essay_questions.csv')

def predict_and_retrieve(job_description):
    if not job_description.strip():
        return "Please enter a job description.", ""
    
    # Preprocess
    cleaned_text = preprocessor.clean_text(job_description)
    
    # Vectorize
    X = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(X)[0]
    
    # Get Questions
    questions = interview_logic.get_questions(prediction)
    formatted_q = interview_logic.format_questions(questions)
    
    return prediction, formatted_q

# Custom CSS for premium look
custom_css = """
.container { max-width: 900px; margin: auto; padding-top: 20px; }
.header { text-align: center; margin-bottom: 30px; }
.title { font-size: 2.5em; font-weight: 700; color: #1A202C; }
.subtitle { color: #4A5568; font-size: 1.1em; }
.output-box { 
    background: #FFFFFF; 
    border-radius: 12px; 
    border: 1px solid #CBD5E0; 
    padding: 30px; 
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    color: #2D3748;
    line-height: 1.6;
}
.output-box h3 { color: #2B6CB0; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; margin-top: 20px; }
.output-box strong { color: #1A202C; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            <div class='header'>
                <h1 class='title'>🚀 Smart AI Interviewer</h1>
                <p class='subtitle'>Predict job roles and generate relevant interview questions instantly.</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                job_input = gr.Textbox(
                    label="Job Description", 
                    placeholder="Paste the job description here...", 
                    lines=10
                )
                submit_btn = gr.Button("Analyze Job & Generate Questions", variant="primary")
            
            with gr.Column(scale=1):
                role_output = gr.Label(label="Predicted Job Role")
                
        gr.Markdown("---")
        
        with gr.Row():
            question_output = gr.Markdown(label="Interview Questions", elem_classes="output-box")

        submit_btn.click(
            fn=predict_and_retrieve,
            inputs=job_input,
            outputs=[role_output, question_output]
        )
        
        gr.Examples(
            examples=[
                ["We are looking for a Python developer with experience in Django and SQL to build scalable web applications."],
                ["Looking for a Machine Learning engineer to design and implement deep learning models for image processing."],
                ["Need a DevOps specialist to manage our AWS infrastructure and CI/CD pipelines using Jenkins and Docker."]
            ],
            inputs=job_input
        )

if __name__ == "__main__":
    demo.launch(share=True)
