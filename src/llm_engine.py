from transformers import pipeline
import torch

class LLMEngine:
    def __init__(self, model_name="google/flan-t5-small"):
        """
        Initialize a local LLM for generating feedback and summaries.
        This fulfills the LLM requirement.
        """
        print(f"Loading local LLM ({model_name})... This might take a moment.")
        # Using CPU for compatibility
        self.generator = pipeline("text2text-generation", model=model_name, device=-1)

    def generate_feedback(self, job_title, candidate_answer):
        """Generate AI feedback for a candidate's answer."""
        prompt = f"Role: {job_title}. Candidate Answer: {candidate_answer}. Task: Provide a brief 1-sentence feedback on this answer."
        result = self.generator(prompt, max_length=50)
        return result[0]['generated_text']

    def summarize_interview(self, job_title, predicted_skills):
        """Summarize the interview requirements."""
        prompt = f"Summarize the key technical skills needed for a {job_title} role specializing in {predicted_skills}."
        result = self.generator(prompt, max_length=100)
        return result[0]['generated_text']
