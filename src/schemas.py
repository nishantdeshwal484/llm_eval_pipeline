from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class WinnerEnum(str, Enum):
    MODEL_A = "Model A"
    MODEL_B = "Model B"
    TIE = "Tie"


class CriterionScore(BaseModel):
    criterion_name: str = Field(..., description="Name of the evaluated dimension (e.g., Correctness, Faithfulness)")
    score: int = Field(..., ge=1, le=5, description="Integer score between 1 and 5")
    rationale: str = Field(..., description="Evidence-backed justification referencing specific sentences")


class PairwiseVerdict(BaseModel):
    winner: WinnerEnum = Field(..., description="Selected winner: 'Model A', 'Model B', or 'Tie'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Judge certainty score between 0.0 and 1.0")
    criteria_scores: List[CriterionScore] = Field(..., description="Per-criterion granular evaluation")
    overall_rationale: str = Field(..., description="Comprehensive verdict explanation")

    @field_validator("criteria_scores")
    def validate_non_empty(cls, v):
        if not v:
            raise ValueError("Criteria scores list cannot be empty")
        return v


class TestCase(BaseModel):
    test_id: str
    input_prompt: str
    system_prompt: Optional[str] = None
    expected_output: Optional[str] = None
    model_a_output: str
    model_b_output: str
    gold_winner: Optional[WinnerEnum] = None
    is_adversarial_probe: bool = False


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage"):
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens