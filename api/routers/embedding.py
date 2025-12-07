"""
Embedding API Router
Handles all text generation, bigram models, and word embedding endpoints.
"""

from fastapi import APIRouter, HTTPException
from help_lib.text_processing import (
    analyze_bigrams, 
    download_book_text, 
    create_probability_matrix,
    generate_text
)
from help_lib.embeddings import embedding_manager
from models.bigram_model import BigramModel

from models.requests import (
    TextGenerationRequest,
    BigramAnalysisRequest,
    WordEmbeddingRequest,
    WordSimilarityRequest,
    SentenceSimilarityRequest,
    WordAlgebraRequest
)
from models.responses import (
    BigramAnalysisResponse,
    WordEmbeddingResponse,
    WordSimilarityResponse,
    SentenceSimilarityResponse,
    WordAlgebraResponse,
    BookTextGenerationResponse,
    BigramMatrixResponse,
    VocabularyResponse,
    HealthResponse
)

# Create router
router = APIRouter(prefix="/embedding", tags=["Embedding & Text Processing"])

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus, frequency_threshold=1)

@router.post("/generate", response_model=dict)
def generate_text_api(request: TextGenerationRequest):
    """Generate text using the bigram model."""
    if request.start_word.lower() not in bigram_model.vocab:
        return {"error": f"Start word '{request.start_word}' not in vocabulary."}
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@router.post("/analyze-bigrams", response_model=BigramAnalysisResponse)
def analyze_bigrams_api(request: BigramAnalysisRequest):
    """Analyze text to compute bigram probabilities."""
    vocab, bigram_probs = analyze_bigrams(request.text, request.frequency_threshold)
    return BigramAnalysisResponse(vocab=vocab, bigram_probabilities=bigram_probs)

@router.get("/bigram-matrix", response_model=BigramMatrixResponse)
def get_bigram_matrix():
    """Get the bigram probability matrix for the current model."""
    vocab = bigram_model.vocab
    bigram_probs = bigram_model.bigram_probs
    matrix = create_probability_matrix(vocab, bigram_probs)
    return BigramMatrixResponse(
        vocab=vocab,
        probability_matrix=matrix
    )

@router.post("/generate-from-book", response_model=BookTextGenerationResponse)
def generate_from_book(request: TextGenerationRequest):
    """Generate text using bigram model trained on a book from Project Gutenberg."""
    try:
        book_text = download_book_text()
        vocab, bigram_probs = analyze_bigrams(book_text, frequency_threshold=5)
        if request.start_word.lower() not in vocab:
            raise HTTPException(
                status_code=400,
                detail=f"Start word '{request.start_word}' not in book vocabulary."
            )
        generated_text = generate_text(bigram_probs, request.start_word, request.length)
        return BookTextGenerationResponse(
            generated_text=generated_text,
            vocab_size=len(vocab),
            source="The Count of Monte Cristo"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/word-embedding", response_model=WordEmbeddingResponse)
def get_word_embedding(request: WordEmbeddingRequest):
    """Get word embedding for a given word."""
    if not embedding_manager.is_available():
        raise HTTPException(status_code=503, detail="spaCy model not available")
    try:
        embedding = embedding_manager.get_word_embedding(request.word)
        return WordEmbeddingResponse(word=request.word, embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating embedding: {str(e)}")

@router.post("/word-similarity", response_model=WordSimilarityResponse)
def calculate_word_similarity_api(request: WordSimilarityRequest):
    """Calculate similarity between two words using their embeddings."""
    if not embedding_manager.is_available():
        raise HTTPException(status_code=503, detail="spaCy model not available")
    try:
        similarity = embedding_manager.calculate_word_similarity(request.word1, request.word2)
        return WordSimilarityResponse(
            word1=request.word1,
            word2=request.word2,
            similarity=similarity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")

@router.post("/sentence-similarity", response_model=SentenceSimilarityResponse)
def calculate_sentence_similarity_api(request: SentenceSimilarityRequest):
    """Calculate similarity between two sentences using their embeddings."""
    if not embedding_manager.is_available():
        raise HTTPException(status_code=503, detail="spaCy model not available")
    try:
        similarity = embedding_manager.calculate_sentence_similarity(request.sentence1, request.sentence2)
        return SentenceSimilarityResponse(
            sentence1=request.sentence1,
            sentence2=request.sentence2,
            similarity=similarity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")

@router.post("/word-algebra", response_model=WordAlgebraResponse)
def calculate_word_algebra_api(request: WordAlgebraRequest):
    """Calculate word algebra: word1 + word2 - word3, compared to word4."""
    if not embedding_manager.is_available():
        raise HTTPException(status_code=503, detail="spaCy model not available")
    try:
        similarity = embedding_manager.calculate_word_algebra(
            request.word1, request.word2, request.word3, request.word4
        )
        expression = f"{request.word1} + {request.word2} - {request.word3}"
        return WordAlgebraResponse(
            expression=expression,
            comparison_word=request.word4,
            similarity=similarity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating word algebra: {str(e)}")

@router.get("/vocabulary", response_model=VocabularyResponse)
def get_vocabulary():
    """Get the vocabulary of the current bigram model."""
    return VocabularyResponse(
        vocabulary=bigram_model.vocab,
        size=len(bigram_model.vocab)
    )

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        spacy_available=embedding_manager.is_available(),
        bigram_model_vocab_size=len(bigram_model.vocab) if bigram_model.vocab else 0
    )