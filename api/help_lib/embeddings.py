"""
Word embedding and NLP functions.
Implementation of word embeddings, similarity calculations, and word algebra.
"""

import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingManager:
    """Manager class for word embeddings using spaCy."""
    
    def __init__(self):
        self.nlp = None
        self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model for word embeddings."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_lg")
        except (OSError, ImportError):
            self.nlp = None
            print("Warning: spaCy en_core_web_lg model not found. Word embedding APIs will not work.")
    
    def is_available(self) -> bool:
        """Check if spaCy model is available."""
        return self.nlp is not None
    
    def get_word_embedding(self, word: str) -> List[float]:
        """Get word embedding for a given word."""
        if not self.is_available():
            raise RuntimeError("spaCy model not available")
        
        word_doc = self.nlp(word)
        return word_doc.vector.tolist()
    
    def calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words using their embeddings."""
        if not self.is_available():
            raise RuntimeError("spaCy model not available")
        
        similarity = self.nlp(word1).similarity(self.nlp(word2))
        return float(similarity)
    
    def calculate_sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate similarity between two sentences using their embeddings."""
        if not self.is_available():
            raise RuntimeError("spaCy model not available")
        
        similarity = self.nlp(sentence1).similarity(self.nlp(sentence2))
        return float(similarity)
    
    def calculate_word_algebra(self, word1: str, word2: str, word3: str, word4: str) -> float:
        """Calculate word algebra: word1 + word2 - word3, compared to word4."""
        if not self.is_available():
            raise RuntimeError("spaCy model not available")
        
        word1_embedding = self.nlp(word1).vector
        word2_embedding = self.nlp(word2).vector
        word3_embedding = self.nlp(word3).vector
        word4_embedding = self.nlp(word4).vector
        
        # Calculate: word1 + word2 - word3
        result_embedding = word1_embedding + word2_embedding - word3_embedding
        
        # Calculate similarity with word4
        similarity = cosine_similarity([result_embedding], [word4_embedding])[0][0]
        return float(similarity)


# Global instance
embedding_manager = EmbeddingManager()
