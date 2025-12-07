"""
Probability and information theory calculations.
Implementation of entropy, cross-entropy, KL divergence, and information content.
"""

import math
import numpy as np
from typing import List


def information(prob: float) -> float:
    """Calculate information content of a probability."""
    if prob <= 0:
        return float('inf')
    return -math.log2(prob)


def calculate_entropy(probabilities: List[float]) -> float:
    """Calculate entropy of a probability distribution."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy += -p * math.log2(p)
    return entropy


def calculate_cross_entropy(true_dist: List[float], pred_dist: List[float]) -> float:
    """Calculate cross-entropy between true and predicted distributions."""
    if len(true_dist) != len(pred_dist):
        raise ValueError("Distributions must have the same length")
    
    cross_entropy = 0.0
    for p, q in zip(true_dist, pred_dist):
        if p > 0 and q > 0:
            cross_entropy += -p * math.log2(q)
        elif p > 0 and q <= 0:
            return float('inf')
    return cross_entropy


def calculate_kl_divergence(p_dist: List[float], q_dist: List[float]) -> float:
    """Calculate KL divergence between two distributions."""
    if len(p_dist) != len(q_dist):
        raise ValueError("Distributions must have the same length")
    
    # Convert to numpy arrays for easier computation
    p = np.array(p_dist)
    q = np.array(q_dist)
    
    # Handle zero probabilities
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    
    kl_div = np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    return float(kl_div)


def calculate_independent_events(p_a: float, p_b: float) -> dict:
    """
    Calculate intersection and union probabilities for independent events.
    
    Args:
        p_a: Probability of event A
        p_b: Probability of event B
    
    Returns:
        Dictionary with P(A ∩ B) and P(A ∪ B)
    """
    p_intersection = p_a * p_b
    p_union = p_a + p_b - p_intersection
    
    return {
        'intersection': p_intersection,
        'union': p_union
    }


def check_independence(p_a: float, p_a_given_b: float, tolerance: float = 1e-10) -> dict:
    """
    Check if two events are independent.
    
    Args:
        p_a: P(A) - probability of event A
        p_a_given_b: P(A|B) - conditional probability of A given B
        tolerance: numerical tolerance for comparison
    
    Returns:
        Dictionary with independence result and explanation
    """
    is_independent = abs(p_a_given_b - p_a) < tolerance
    
    return {
        'is_independent': is_independent,
        'p_a': p_a,
        'p_a_given_b': p_a_given_b,
        'explanation': "Events are independent if P(A|B) = P(A)"
    }


def calculate_bayes_posterior(p_a: float, p_b_given_a: float, p_b_given_a_complement: float) -> dict:
    """
    Calculate P(A|B) using Bayes' rule.
    
    Args:
        p_a: P(A) - prior probability
        p_b_given_a: P(B|A) - likelihood
        p_b_given_a_complement: P(B|A') - likelihood given complement
    
    Returns:
        Dictionary with P(B) and P(A|B)
    """
    p_a_complement = 1 - p_a
    p_b = p_b_given_a * p_a + p_b_given_a_complement * p_a_complement
    p_a_given_b = (p_b_given_a * p_a) / p_b
    
    return {
        'p_b': p_b,
        'p_a_given_b': p_a_given_b
    }


def calculate_medical_test_probability(sensitivity: float, specificity: float, prevalence: float) -> dict:
    """
    Calculate probability of having disease given positive test using Bayes' rule.
    
    Args:
        sensitivity: P(positive|disease) - true positive rate
        specificity: P(negative|no disease) - true negative rate
        prevalence: P(disease) - disease prevalence in population
    
    Returns:
        Dictionary with test probabilities
    """
    p_positive_given_disease = sensitivity
    p_positive_given_no_disease = 1 - specificity
    p_disease = prevalence
    p_no_disease = 1 - prevalence
    
    # Law of total probability
    p_positive = (p_positive_given_disease * p_disease + 
                 p_positive_given_no_disease * p_no_disease)
    
    # Bayes' rule
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    return {
        'p_positive': p_positive,
        'p_disease_given_positive': p_disease_given_positive
    }


def calculate_distribution_statistics(values: List[float], probabilities: List[float], sample: List[float] = None) -> dict:
    """
    Calculate expected value, variance, and sample statistics.
    
    Args:
        values: List of possible values
        probabilities: List of corresponding probabilities
        sample: Optional sample data for comparison
    
    Returns:
        Dictionary with statistical measures
    """
    if len(values) != len(probabilities):
        raise ValueError("Values and probabilities must have the same length")
    
    values_arr = np.array(values)
    probs_arr = np.array(probabilities)
    
    expected_value = np.sum(values_arr * probs_arr)
    expected_value_sq = np.sum((values_arr ** 2) * probs_arr)
    variance = expected_value_sq - expected_value ** 2
    
    result = {
        'expected_value': expected_value,
        'variance': variance,
        'standard_deviation': math.sqrt(variance)
    }
    
    if sample is not None:
        sample_arr = np.array(sample)
        sample_mean = np.mean(sample_arr)
        sample_variance = np.var(sample_arr, ddof=1)  # Sample variance with Bessel's correction
        
        result.update({
            'sample_mean': sample_mean,
            'sample_variance': sample_variance,
            'sample_matches_expected': abs(sample_mean - expected_value) < 1e-10
        })
    
    return result


def compare_entropy_distributions(original_probs: List[float], equal_probs: List[float] = None) -> dict:
    """
    Compare entropy of original distribution with equal probability distribution.
    
    Args:
        original_probs: Original probability distribution
        equal_probs: Equal probability distribution (optional, will be calculated if not provided)
    
    Returns:
        Dictionary with entropy values
    """
    entropy_original = calculate_entropy(original_probs)
    
    if equal_probs is None:
        n = len(original_probs)
        equal_probs = [1.0 / n] * n
    
    entropy_equal = calculate_entropy(equal_probs)
    
    return {
        'original_entropy': entropy_original,
        'equal_entropy': entropy_equal,
        'entropy_difference': entropy_equal - entropy_original
    }


def calculate_expected_value(values: List[float], probabilities: List[float]) -> float:
    """Calculate expected value of a discrete random variable."""
    if len(values) != len(probabilities):
        raise ValueError("Values and probabilities must have the same length")
    
    return sum(v * p for v, p in zip(values, probabilities))


def calculate_variance(values: List[float], probabilities: List[float]) -> float:
    """Calculate variance of a discrete random variable."""
    if len(values) != len(probabilities):
        raise ValueError("Values and probabilities must have the same length")
    
    expected_val = calculate_expected_value(values, probabilities)
    expected_val_sq = sum(v * v * p for v, p in zip(values, probabilities))
    
    return expected_val_sq - expected_val ** 2


def bayes_rule(prior: float, likelihood: float, evidence: float) -> float:
    """
    Apply Bayes' rule: P(A|B) = P(B|A) * P(A) / P(B)
    
    Args:
        prior: P(A) - prior probability
        likelihood: P(B|A) - likelihood
        evidence: P(B) - evidence
    
    Returns:
        P(A|B) - posterior probability
    """
    if evidence == 0:
        raise ValueError("Evidence probability cannot be zero")
    
    return (likelihood * prior) / evidence


def test_independence(p_a: float, p_a_given_b: float, tolerance: float = 1e-10) -> bool:
    """
    Test if two events are independent.
    Events A and B are independent if P(A|B) = P(A).
    
    Args:
        p_a: P(A) - probability of event A
        p_a_given_b: P(A|B) - conditional probability of A given B
        tolerance: numerical tolerance for comparison
    
    Returns:
        True if events are independent, False otherwise
    """
    return abs(p_a_given_b - p_a) < tolerance
