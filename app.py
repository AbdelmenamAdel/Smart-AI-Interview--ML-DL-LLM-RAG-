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

print("Initializing Systems...")
rag_system = RAGSystem('data/mcq_questions.csv', 'data/essay_questions.csv')
llm_engine = LLMEngine()
print("Systems Ready!")

def start_exam(job_description):
    if not job_description.strip():
        return [gr.update(visible=False)] * 10 + ["Please enter a job description."]
    
    # ML Prediction
    cleaned_text = preprocessor.clean_text(job_description)
    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)[0]
    
    # RAG Retrieval
    bundle = rag_system.get_interview_bundle(prediction, job_description)
    mcqs = bundle['mcqs'][:5] # Take top 5 for the interactive exam
    essays = bundle['essays'][:2] # Take top 2
    
    updates = []
    # Update MCQs
    for i in range(5):
        if i < len(mcqs):
            q = mcqs[i]
            label = f"Q{i+1}: {q['Question']}"
            choices = [q['Option_A'], q['Option_B'], q['Option_C'], q['Option_D']]
            updates.append(gr.update(label=label, choices=choices, visible=True, value=None))
        else:
            updates.append(gr.update(visible=False))
            
    # Update Essays
    for i in range(2):
        if i < len(essays):
            updates.append(gr.update(label=f"Essay Q{i+1}: {essays[i]['Question']}", visible=True, value=""))
        else:
            updates.append(gr.update(visible=False))
            
    return updates + [prediction, mcqs, essays]

def submit_exam(role, mcq1, mcq2, mcq3, mcq4, mcq5, essay1, essay2, mcqs_data, essays_data):
    score = 0
    total_mcqs = len(mcqs_data)
    user_answers = [mcq1, mcq2, mcq3, mcq4, mcq5]
    
    details = "### 📊 Exam Performance Report\n\n"
    
    # Calculate MCQ Score
    for i, q in enumerate(mcqs_data):
        correct_mapping = {'A': q['Option_A'], 'B': q['Option_B'], 'C': q['Option_C'], 'D': q['Option_D']}
        correct_val = correct_mapping.get(q['Correct_Answer'])
        if user_answers[i] == correct_val:
            score += 1
            details += f"✅ **Q{i+1}**: Correct!\n"
        else:
            details += f"❌ **Q{i+1}**: Incorrect. (Correct: {q['Correct_Answer']})\n"
            
    final_score = (score / total_mcqs) * 100
    details += f"\n**Final Score: {final_score:.1f}%**\n\n"
    
    # LLM Feedback on Essays
    details += "### 🤖 AI Feedback on Essays\n"
    if essay1:
        feedback1 = llm_engine.generate_feedback(role, essay1)
        details += f"**Essay 1 Feedback**: {feedback1}\n"
    if essay2:
        feedback2 = llm_engine.generate_feedback(role, essay2)
        details += f"**Essay 2 Feedback**: {feedback2}\n"
        
    return details

# UI Styling
custom_css = """
.container { max-width: 1100px; margin: auto; padding: 20px; }
.header { text-align: center; padding: 40px; background: #2D3748; color: white; border-radius: 15px; margin-bottom: 30px; }
.exam-section { background: #F7FAFC; padding: 30px; border-radius: 15px; border: 1px solid #E2E8F0; }
.submit-btn { background: #48BB78 !important; color: white !important; font-size: 1.2em !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    mcqs_state = gr.State([])
    essays_state = gr.State([])
    
    with gr.Column(elem_classes="container"):
        gr.Markdown("<div class='header'><h1>🎓 Smart AI Examination Portal</h1><p>Interactive Testing powered by ML, RAG & LLM</p></div>")
        
        with gr.Tabs():
            with gr.TabItem("Step 1: Job Setup"):
                job_input = gr.Textbox(label="Paste Job Description", lines=8, placeholder="Enter the job description here...")
                role_output = gr.Textbox(label="Predicted Career Path", interactive=False)
                start_btn = gr.Button("🚀 Generate Interactive Exam", variant="primary")
            
            with gr.TabItem("Step 2: Technical Exam"):
                with gr.Column(elem_classes="exam-section"):
                    gr.Markdown("### 📝 Part 1: Multiple Choice Questions")
                    mcq_q1 = gr.Radio(visible=False)
                    mcq_q2 = gr.Radio(visible=False)
                    mcq_q3 = gr.Radio(visible=False)
                    mcq_q4 = gr.Radio(visible=False)
                    mcq_q5 = gr.Radio(visible=False)
                    
                    gr.Markdown("### ✍️ Part 2: Essay Questions")
                    essay_q1 = gr.Textbox(visible=False, lines=4)
                    essay_q2 = gr.Textbox(visible=False, lines=4)
                    
                    submit_btn = gr.Button("Submit Exam for AI Evaluation", elem_classes="submit-btn")
            
            with gr.TabItem("Step 3: Results & Feedback"):
                final_report = gr.Markdown("Waiting for submission...")

        start_btn.click(
            fn=start_exam,
            inputs=job_input,
            outputs=[mcq_q1, mcq_q2, mcq_q3, mcq_q4, mcq_q5, essay_q1, essay_q2, role_output, mcqs_state, essays_state]
        )
        
        submit_btn.click(
            fn=submit_exam,
            inputs=[role_output, mcq_q1, mcq_q2, mcq_q3, mcq_q4, mcq_q5, essay_q1, essay_q2, mcqs_state, essays_state],
            outputs=final_report
        )

if __name__ == "__main__":
    demo.launch(share=True)
