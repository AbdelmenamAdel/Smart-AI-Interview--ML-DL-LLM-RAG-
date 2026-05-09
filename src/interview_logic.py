import pandas as pd
import random
import os

class InterviewLogic:
    def __init__(self, mcq_path, essay_path):
        self.mcq_df = pd.read_csv(mcq_path)
        self.essay_df = pd.read_csv(essay_path)
        
        # Mapping predicted job titles to skills found in the datasets
        self.job_to_skills = {
            'Flutter Developer': ['Flutter', 'Git', 'Android', 'iOS'],
            'Django Developer': ['Django', 'Python', 'SQL', 'PostgreSQL', 'Git'],
            'Machine Learning': ['Machine Learning', 'Python', 'AWS', 'Linux'],
            'iOS Developer': ['iOS', 'Swift', 'Git'],
            'Full Stack Developer': ['JavaScript', 'React', 'Node.js', 'HTML', 'CSS', 'SQL', 'MongoDB'],
            'Java Developer': ['Java', 'Spring', 'SQL', 'MySQL', 'Git'],
            'JavaScript Developer': ['JavaScript', 'React', 'Node.js', 'jQuery', 'Angular'],
            'DevOps Engineer': ['Docker', 'Jenkins', 'Kubernetes', 'CI/CD', 'Linux', 'Ansible', 'AWS', 'Azure'],
            'Software Engineer': ['Python', 'Java', 'Git', 'Agile', 'Scrum', 'Linux'],
            'Database Administrator': ['SQL', 'MySQL', 'PostgreSQL', 'Oracle', 'Linux'],
            'Wordpress Developer': ['PHP', 'WordPress', 'HTML', 'CSS', 'MySQL'],
            'PHP Developer': ['PHP', 'Laravel', 'MySQL', 'JavaScript'],
            'Android Developer': ['Android', 'Java', 'Git'],
            'Data Scientist': ['Machine Learning', 'Python', 'SQL', 'Linux'],
            'React Developer': ['React', 'JavaScript', 'HTML', 'CSS', 'Node.js']
        }

    def get_questions(self, job_title, num_mcq=5, num_essay=3):
        """Retrieve relevant questions based on the job title."""
        skills = self.job_to_skills.get(job_title, ['Python', 'Git', 'Agile']) # Default skills
        
        # Filter MCQs
        relevant_mcqs = self.mcq_df[self.mcq_df['Skill'].isin(skills)]
        if relevant_mcqs.empty:
            relevant_mcqs = self.mcq_df # Fallback to all questions if none found
            
        # Filter Essays
        relevant_essays = self.essay_df[self.essay_df['Skill'].isin(skills)]
        if relevant_essays.empty:
            relevant_essays = self.essay_df # Fallback
            
        # Randomly sample
        mcqs = relevant_mcqs.sample(min(num_mcq, len(relevant_mcqs))).to_dict('records')
        essays = relevant_essays.sample(min(num_essay, len(relevant_essays))).to_dict('records')
        
        return {
            'mcqs': mcqs,
            'essays': essays
        }

    def format_questions(self, questions):
        """Format the questions for display."""
        output = "### Technical MCQs\n\n"
        for i, q in enumerate(questions['mcqs']):
            output += f"{i+1}. **{q['Question']}**\n"
            output += f"   - A) {q['Option_A']}\n"
            output += f"   - B) {q['Option_B']}\n"
            output += f"   - C) {q['Option_C']}\n"
            output += f"   - D) {q['Option_D']}\n\n"
            
        output += "---\n### Essay Questions\n\n"
        for i, q in enumerate(questions['essays']):
            output += f"{i+1}. **{q['Question']}**\n\n"
            
        return output
