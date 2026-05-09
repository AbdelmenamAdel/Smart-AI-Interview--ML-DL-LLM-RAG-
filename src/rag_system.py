import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

class RAGSystem:
    def __init__(self, mcq_path, essay_path, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the RAG system using a pre-trained Deep Learning model (Sentence-Transformers).
        This covers the DL and RAG requirements.
        """
        self.model = SentenceTransformer(model_name)
        self.mcq_df = pd.read_csv(mcq_path)
        self.essay_df = pd.read_csv(essay_path)
        
        # Prepare datasets for indexing
        self.mcq_texts = self.mcq_df.apply(lambda x: f"{x['Skill']} {x['Question']}", axis=1).tolist()
        self.essay_texts = self.essay_df.apply(lambda x: f"{x['Skill']} {x['Question']}", axis=1).tolist()
        
        # Create FAISS Indices
        self.mcq_index = self._build_index(self.mcq_texts)
        self.essay_index = self._build_index(self.essay_texts)

    def _build_index(self, texts):
        """Build a FAISS index for semantic search."""
        embeddings = self.model.encode(texts)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index

    def retrieve_questions(self, query, num_results=5, dataset_type='mcq'):
        """
        Retrieve the most relevant questions using Semantic Search (RAG).
        """
        query_embedding = self.model.encode([query]).astype('float32')
        
        if dataset_type == 'mcq':
            index = self.mcq_index
            df = self.mcq_df
        else:
            index = self.essay_index
            df = self.essay_df
            
        distances, indices = index.search(query_embedding, num_results)
        results = df.iloc[indices[0]].to_dict('records')
        return results

    def get_interview_bundle(self, job_title, job_description):
        """Combine job title and description for a rich search query."""
        query = f"{job_title} {job_description}"
        mcqs = self.retrieve_questions(query, num_results=5, dataset_type='mcq')
        essays = self.retrieve_questions(query, num_results=3, dataset_type='essay')
        
        return {
            'mcqs': mcqs,
            'essays': essays
        }
