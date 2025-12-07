"""
Text processing and bigram analysis functions.
Implementation of tokenization, bigram analysis, and text generation.
"""

import requests
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
from fastapi import HTTPException


def simple_tokenizer(text: str, frequency_threshold: int = 5) -> List[str]:
    """Simple tokenizer that splits text into words."""
    # Convert to lowercase and extract words using regex
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    # Count word frequencies
    word_counts = Counter(tokens)
    # Define a threshold for less frequent words (e.g., words appearing fewer than 5 times)
    filtered_tokens = [
        token for token in tokens if word_counts[token] >= frequency_threshold
    ]
    return filtered_tokens


def analyze_bigrams(text: str, frequency_threshold: int = None) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Analyze text to compute bigram probabilities."""
    words = simple_tokenizer(text, frequency_threshold)
    bigrams = list(zip(words[:-1], words[1:]))  # Create bigrams

    # Count bigram and unigram frequencies
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(words)

    # Compute bigram probabilities
    bigram_probs = defaultdict(dict)
    for (word1, word2), count in bigram_counts.items():
        bigram_probs[word1][word2] = count / unigram_counts[word1]

    return list(unigram_counts.keys()), dict(bigram_probs)


def generate_text(bigram_probs: Dict[str, Dict[str, float]], start_word: str, num_words: int = 20) -> str:
    """Generate text based on bigram probabilities."""
    import random
    
    current_word = start_word.lower()
    generated_words = [current_word]

    for _ in range(num_words - 1):
        next_words = bigram_probs.get(current_word)
        if not next_words:  # If no bigrams for the current word, stop generating
            break

        # Choose the next word based on probabilities
        next_word = random.choices(
            list(next_words.keys()), weights=next_words.values()
        )[0]
        generated_words.append(next_word)
        current_word = next_word  # Move to the next word

    return " ".join(generated_words)


def download_book_text(url: str = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt") -> str:
    """Download book text from Project Gutenberg."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        book_text = response.text
        
        # Remove Gutenberg header and footer
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

        start_idx = book_text.find(start_marker)
        end_idx = book_text.find(end_marker)

        if start_idx != -1 and end_idx != -1:
            book_text = book_text[start_idx + len(start_marker) : end_idx]
        
        return book_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download book: {str(e)}")


def create_probability_matrix(vocab: List[str], bigram_probs: Dict[str, Dict[str, float]]) -> List[List[float]]:
    """Create a probability matrix from bigram probabilities."""
    matrix = []
    for word1 in vocab:
        row = []
        for word2 in vocab:
            prob = bigram_probs.get(word1, {}).get(word2, 0.0)
            row.append(prob)
        matrix.append(row)
    return matrix
