import gradio as gr
import os
import pandas as pd
from src.preprocessing import TextPreprocessor
from src.model_trainer import JobClassifier
from src.rag_system import RAGSystem
from src.llm_engine import LLMEngine

# Initialize components
preprocessor = TextPreprocessor()
model, vectorizer = JobClassifier.load_model('models/job_classifier.pkl', 'models/tfidf_vectorizer.pkl')

# RAG and LLM (Lazy loading to avoid startup delay)
rag_system = None
llm_engine = None

def get_rag():
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem('data/mcq_questions.csv', 'data/essay_questions.csv')
    return rag_system

def get_llm():
    global llm_engine
    if llm_engine is None:
        llm_engine = LLMEngine()
    return llm_engine

def process_job(job_description):
    if not job_description.strip():
        return "Please enter a job description.", "", "Waiting for analysis..."
    
    # ML Part: Prediction
    cleaned_text = preprocessor.clean_text(job_description)
    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)[0]
    
    # RAG Part: Semantic Retrieval (DL Embeddings)
    rag = get_rag()
    bundle = rag.get_interview_bundle(prediction, job_description)
    
    # Formatting output
    formatted_q = "### 📚 Semantic Search Results (RAG)\n\n"
    formatted_q += "#### Technical MCQs\n"
    for i, q in enumerate(bundle['mcqs']):
        formatted_q += f"{i+1}. **{q['Question']}**\n"
    
    formatted_q += "\n---\n#### Essay Questions\n"
    for i, q in enumerate(bundle['essays']):
        formatted_q += f"{i+1}. **{q['Question']}**\n"

    # LLM Part: Summary
    llm = get_llm()
    summary = llm.summarize_interview(prediction, "Technical Skills")
    
    return prediction, formatted_q, summary

def generate_ai_feedback(job_title, candidate_answer):
    if not candidate_answer.strip():
        return "Please provide an answer to get feedback."
    llm = get_llm()
    feedback = llm.generate_feedback(job_title, candidate_answer)
    return feedback

# UI Layout
custom_css = """
.container { max-width: 1000px; margin: auto; padding-top: 20px; }
.header { text-align: center; margin-bottom: 30px; background: linear-gradient(90deg, #2B6CB0 0%, #4299E1 100%); padding: 40px; border-radius: 15px; color: white; }
.output-box { background: white; border-radius: 12px; border: 1px solid #E2E8F0; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            <div class='header'>
                <h1>🤖 Advanced Smart AI Interviewer</h1>
                <p>ML Prediction + DL Embeddings + RAG Retrieval + LLM Feedback</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                job_input = gr.Textbox(label="Step 1: Paste Job Description", lines=8)
                analyze_btn = gr.Button("Analyze Job (ML + RAG)", variant="primary")
            
            with gr.Column(scale=1):
                role_output = gr.Textbox(label="Predicted Role (ML)")
                llm_summary = gr.Textbox(label="AI Role Summary (LLM)", lines=5)
        
        with gr.Row():
            questions_output = gr.Markdown(label="Interview Questions (RAG System)", elem_classes="output-box")
        
        gr.Markdown("---")
        gr.Markdown("### 📝 Step 2: Practice & Get AI Feedback")
        
        with gr.Row():
            with gr.Column(scale=2):
                answer_input = gr.Textbox(label="Your Answer to an Essay Question", lines=5)
                feedback_btn = gr.Button("Get AI Feedback (LLM)")
            with gr.Column(scale=1):
                feedback_output = gr.Textbox(label="AI Feedback", lines=5)

        analyze_btn.click(
            fn=process_job,
            inputs=job_input,
            outputs=[role_output, questions_output, llm_summary]
        )
        
        feedback_btn.click(
            fn=generate_ai_feedback,
            inputs=[role_output, answer_input],
            outputs=feedback_output
        )

if __name__ == "__main__":
    demo.launch(share=True)
