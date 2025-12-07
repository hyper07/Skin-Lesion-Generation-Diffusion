"""
Probability API Router
Handles all probability and statistical calculation endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from help_lib.probability import (
    information,
    calculate_entropy,
    calculate_cross_entropy,
    calculate_kl_divergence,
    calculate_independent_events,
    check_independence,
    calculate_bayes_posterior,
    calculate_medical_test_probability,
    calculate_distribution_statistics,
    compare_entropy_distributions,
    calculate_expected_value,
    calculate_variance,
    bayes_rule,
    test_independence
)


# Probability API Models
class InformationRequest(BaseModel):
    prob: float

class InformationResponse(BaseModel):
    information: float

class IndependentEventsRequest(BaseModel):
    p_a: float
    p_b: float

class IndependentEventsResponse(BaseModel):
    intersection: float
    union: float

class IndependenceCheckRequest(BaseModel):
    p_a: float
    p_a_given_b: float
    tolerance: Optional[float] = 1e-10

class IndependenceCheckResponse(BaseModel):
    is_independent: bool
    p_a: float
    p_a_given_b: float
    explanation: str

class BayesPosteriorRequest(BaseModel):
    p_a: float
    p_b_given_a: float
    p_b_given_a_complement: float

class BayesPosteriorResponse(BaseModel):
    p_b: float
    p_a_given_b: float

class MedicalTestProbabilityRequest(BaseModel):
    sensitivity: float
    specificity: float
    prevalence: float

class MedicalTestProbabilityResponse(BaseModel):
    p_positive: float
    p_disease_given_positive: float

class DistributionStatisticsRequest(BaseModel):
    values: List[float]
    probabilities: List[float]
    sample: Optional[List[float]] = None

class DistributionStatisticsResponse(BaseModel):
    expected_value: float
    variance: float
    standard_deviation: float
    sample_mean: Optional[float] = None
    sample_variance: Optional[float] = None
    sample_matches_expected: Optional[bool] = None

class CompareEntropyRequest(BaseModel):
    original_probs: List[float]
    equal_probs: Optional[List[float]] = None

class CompareEntropyResponse(BaseModel):
    original_entropy: float
    equal_entropy: float
    entropy_difference: float

class ExpectedValueRequest(BaseModel):
    values: List[float]
    probabilities: List[float]

class ExpectedValueResponse(BaseModel):
    expected_value: float

class VarianceRequest(BaseModel):
    values: List[float]
    probabilities: List[float]

class VarianceResponse(BaseModel):
    variance: float

class BayesRuleRequest(BaseModel):
    prior: float
    likelihood: float
    evidence: float

class BayesRuleResponse(BaseModel):
    posterior: float

class TestIndependenceRequest(BaseModel):
    p_a: float
    p_a_given_b: float
    tolerance: Optional[float] = 1e-10

class TestIndependenceResponse(BaseModel):
    is_independent: bool

# Import models from models module
from models.requests import ProbabilityDistribution, CrossEntropyRequest, KLDivergenceRequest
from models.responses import EntropyResponse, CrossEntropyResponse, KLDivergenceResponse

# Create router
router = APIRouter(prefix="/probability", tags=["Probability"])

@router.post("/information", response_model=InformationResponse)
def information_api(request: InformationRequest):
    """Calculate information content of a probability."""
    info = information(request.prob)
    return InformationResponse(information=info)

@router.post("/independent-events", response_model=IndependentEventsResponse)
def independent_events_api(request: IndependentEventsRequest):
    """Calculate intersection and union probabilities for independent events."""
    result = calculate_independent_events(request.p_a, request.p_b)
    return IndependentEventsResponse(**result)

@router.post("/check-independence", response_model=IndependenceCheckResponse)
def check_independence_api(request: IndependenceCheckRequest):
    """Check if two events are independent."""
    result = check_independence(request.p_a, request.p_a_given_b, request.tolerance)
    return IndependenceCheckResponse(**result)

@router.post("/bayes-posterior", response_model=BayesPosteriorResponse)
def bayes_posterior_api(request: BayesPosteriorRequest):
    """Calculate P(A|B) using Bayes' rule."""
    result = calculate_bayes_posterior(request.p_a, request.p_b_given_a, request.p_b_given_a_complement)
    return BayesPosteriorResponse(**result)

@router.post("/medical-test-probability", response_model=MedicalTestProbabilityResponse)
def medical_test_probability_api(request: MedicalTestProbabilityRequest):
    """Calculate probability of having disease given positive test using Bayes' rule."""
    result = calculate_medical_test_probability(request.sensitivity, request.specificity, request.prevalence)
    return MedicalTestProbabilityResponse(**result)

@router.post("/distribution-statistics", response_model=DistributionStatisticsResponse)
def distribution_statistics_api(request: DistributionStatisticsRequest):
    """Calculate expected value, variance, and sample statistics."""
    result = calculate_distribution_statistics(request.values, request.probabilities, request.sample)
    return DistributionStatisticsResponse(**result)

@router.post("/compare-entropy", response_model=CompareEntropyResponse)
def compare_entropy_api(request: CompareEntropyRequest):
    """Compare entropy of original distribution with equal probability distribution."""
    result = compare_entropy_distributions(request.original_probs, request.equal_probs)
    return CompareEntropyResponse(**result)

@router.post("/expected-value", response_model=ExpectedValueResponse)
def expected_value_api(request: ExpectedValueRequest):
    """Calculate expected value of a discrete random variable."""
    val = calculate_expected_value(request.values, request.probabilities)
    return ExpectedValueResponse(expected_value=val)

@router.post("/variance", response_model=VarianceResponse)
def variance_api(request: VarianceRequest):
    """Calculate variance of a discrete random variable."""
    var = calculate_variance(request.values, request.probabilities)
    return VarianceResponse(variance=var)

@router.post("/bayes-rule", response_model=BayesRuleResponse)
def bayes_rule_api(request: BayesRuleRequest):
    """Apply Bayes' rule: P(A|B) = P(B|A) * P(A) / P(B)."""
    posterior = bayes_rule(request.prior, request.likelihood, request.evidence)
    return BayesRuleResponse(posterior=posterior)

@router.post("/test-independence", response_model=TestIndependenceResponse)
def test_independence_api(request: TestIndependenceRequest):
    """Test if two events are independent."""
    result = test_independence(request.p_a, request.p_a_given_b, request.tolerance)
    return TestIndependenceResponse(is_independent=result)

@router.post("/entropy", response_model=EntropyResponse)
def calculate_entropy_api(request: ProbabilityDistribution):
    """Calculate entropy of a probability distribution."""
    prob_sum = sum(request.probabilities)
    if abs(prob_sum - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail="Probabilities must sum to 1.0")
    entropy = calculate_entropy(request.probabilities)
    return EntropyResponse(entropy=entropy)

@router.post("/cross-entropy", response_model=CrossEntropyResponse)
def calculate_cross_entropy_api(request: CrossEntropyRequest):
    """Calculate cross-entropy between true and predicted distributions."""
    cross_entropy = calculate_cross_entropy(request.true_distribution, request.predicted_distribution)
    return CrossEntropyResponse(cross_entropy=cross_entropy)

@router.post("/kl-divergence", response_model=KLDivergenceResponse)
def calculate_kl_divergence_api(request: KLDivergenceRequest):
    """Calculate KL divergence between two distributions."""
    kl_div = calculate_kl_divergence(request.p_distribution, request.q_distribution)
    return KLDivergenceResponse(kl_divergence=kl_div)