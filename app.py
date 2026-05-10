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

print("Initializing Advanced Systems...")
rag_system = RAGSystem('data/mcq_questions.csv', 'data/essay_questions.csv')
llm_engine = LLMEngine()
print("Systems Ready!")

def start_exam(job_description):
    if not job_description.strip():
        return [gr.update(visible=False)] * 10 + ["Please enter a job description.", [], [], gr.update(selected="Step 1: Job Setup")]
    
    # ML Prediction
    cleaned_text = preprocessor.clean_text(job_description)
    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)[0]
    
    # RAG Retrieval
    bundle = rag_system.get_interview_bundle(prediction, job_description)
    mcqs = bundle['mcqs'][:5]
    essays = bundle['essays'][:2]
    
    updates = []
    for i in range(5):
        if i < len(mcqs):
            q = mcqs[i]
            label = f"Q{i+1}: {q['Question']}"
            choices = [q['Option_A'], q['Option_B'], q['Option_C'], q['Option_D']]
            updates.append(gr.update(label=label, choices=choices, visible=True, value=None))
        else:
            updates.append(gr.update(visible=False))
            
    for i in range(2):
        if i < len(essays):
            updates.append(gr.update(label=f"Essay Q{i+1}: {essays[i]['Question']}", visible=True, value=""))
        else:
            updates.append(gr.update(visible=False))
            
    # Return everything + switch to Tab 2
    return updates + [prediction, mcqs, essays, gr.update(selected="Step 2: Technical Exam")]

def submit_exam(role, mcq1, mcq2, mcq3, mcq4, mcq5, essay1, essay2, mcqs_data, essays_data):
    score = 0
    total_mcqs = len(mcqs_data)
    user_answers = [mcq1, mcq2, mcq3, mcq4, mcq5]
    
    report_html = f"""
    <div style='text-align: center; padding: 20px; border-radius: 15px; background: #EDF2F7; margin-bottom: 20px;'>
        <h2 style='color: #2D3748;'>🎯 Final Interview Report</h2>
        <p style='font-size: 1.2em; color: #4A5568;'>Predicted Career: <b>{role}</b></p>
    </div>
    """
    
    mcq_details = "<div style='padding: 15px; border-left: 5px solid #4299E1; background: #EBF8FF; margin-bottom: 20px; color: #2C5282;'>"
    mcq_details += "<h4>📝 MCQ Performance</h4>"
    
    for i, q in enumerate(mcqs_data):
        correct_mapping = {'A': q['Option_A'], 'B': q['Option_B'], 'C': q['Option_C'], 'D': q['Option_D']}
        correct_val = correct_mapping.get(q['Correct_Answer'])
        if user_answers[i] == correct_val:
            score += 1
            mcq_details += f"<p>✅ <b>Question {i+1}</b>: Correct</p>"
        else:
            mcq_details += f"<p>❌ <b>Question {i+1}</b>: Incorrect (Expected: {q['Correct_Answer']})</p>"
            
    final_score = (score / total_mcqs) * 100
    mcq_details += f"<hr><h3 style='color: #2B6CB0;'>Total Score: {final_score:.1f}%</h3></div>"
    
    # AI Feedback Section
    ai_details = "<div style='padding: 15px; border-left: 5px solid #48BB78; background: #F0FFF4; color: #22543D;'>"
    ai_details += "<h4>🤖 AI Qualitative Feedback</h4>"
    
    if essay1:
        ai_details += f"<p><b>Essay 1 Analysis:</b> {llm_engine.generate_feedback(role, essay1)}</p>"
    if essay2:
        ai_details += f"<p><b>Essay 2 Analysis:</b> {llm_engine.generate_feedback(role, essay2)}</p>"
    
    ai_details += "</div>"
    
    return report_html + mcq_details + ai_details, gr.update(selected="Step 3: Results & Feedback")

# UI Styling
custom_css = """
.container { max-width: 1000px; margin: auto; padding: 20px; }
.header { text-align: center; padding: 50px; background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%); color: white; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
.header h1 { font-size: 3em; margin: 0; }
.exam-section { background: white; padding: 40px; border-radius: 20px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
.nav-btn { font-size: 1.1em !important; height: 50px !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    mcqs_state = gr.State([])
    essays_state = gr.State([])
    
    with gr.Column(elem_classes="container"):
        gr.HTML("<div class='header'><h1>🤖 Smart AI Interviewer</h1><p>The Complete AI Career Readiness System</p></div>")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Step 1: Job Setup", id="Step 1: Job Setup"):
                job_input = gr.Textbox(label="Job Description", lines=10, placeholder="Paste the job description here to generate a custom exam...")
                role_output = gr.Textbox(label="Predicted Role", interactive=False)
                start_btn = gr.Button("🔍 Analyze & Generate Exam", variant="primary", elem_classes="nav-btn")
            
            with gr.TabItem("Step 2: Technical Exam", id="Step 2: Technical Exam"):
                with gr.Column(elem_classes="exam-section"):
                    gr.Markdown("## 📋 Technical Examination")
                    mcq_q1 = gr.Radio(visible=False)
                    mcq_q2 = gr.Radio(visible=False)
                    mcq_q3 = gr.Radio(visible=False)
                    mcq_q4 = gr.Radio(visible=False)
                    mcq_q5 = gr.Radio(visible=False)
                    
                    gr.Markdown("## ✍️ Qualitative Assessment")
                    essay_q1 = gr.Textbox(visible=False, lines=4)
                    essay_q2 = gr.Textbox(visible=False, lines=4)
                    
                    submit_btn = gr.Button("🏁 Submit & Get Report", variant="primary", elem_classes="nav-btn")
            
            with gr.TabItem("Step 3: Results & Feedback", id="Step 3: Results & Feedback"):
                final_report = gr.HTML("<div style='text-align: center; padding: 50px; color: #718096;'>Finish the exam to see your report...</div>")

        start_btn.click(
            fn=start_exam,
            inputs=job_input,
            outputs=[mcq_q1, mcq_q2, mcq_q3, mcq_q4, mcq_q5, essay_q1, essay_q2, role_output, mcqs_state, essays_state, tabs]
        )
        
        submit_btn.click(
            fn=submit_exam,
            inputs=[role_output, mcq_q1, mcq_q2, mcq_q3, mcq_q4, mcq_q5, essay_q1, essay_q2, mcqs_state, essays_state],
            outputs=[final_report, tabs]
        )

if __name__ == "__main__":
    demo.launch(share=True)
