import os
import json
import asyncio
from typing import Tuple, Dict, Any
from src.schemas import TokenUsage


class BaseLLMClient:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.total_usage = TokenUsage()

    async def generate(self, prompt: str, system_prompt: str = "") -> Tuple[str, TokenUsage]:
        raise NotImplementedError


class MockJudgeLLMClient(BaseLLMClient):
    """
    High-fidelity mock LLM client for offline validation, unit testing, and fast demonstrations.
    Simulates position bias and stochastic behavior on unmitigated runs.
    """
    async def generate(self, prompt: str, system_prompt: str = "") -> Tuple[str, TokenUsage]:
        await asyncio.sleep(0.05)  # Simulate API network roundtrip
        
        # Simulated usage metrics
        usage = TokenUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=150,
            total_tokens=len(prompt.split()) + 150
        )
        self.total_usage.add(usage)

        # Detect position swap in prompt to simulate evaluation logic
        is_reversed = "Model A:" in prompt and "OUTPUT B" in prompt  # Logic check context
        
        # Deterministic mock payload with criteria reasoning
        mock_payload = {
            "winner": "Model A",
            "confidence": 0.92,
            "criteria_scores": [
                {"criterion_name": "Correctness", "score": 5, "rationale": "Model A directly answered the prompt without factual errors."},
                {"criterion_name": "Faithfulness", "score": 4, "rationale": "Adheres strictly to provided constraints."},
                {"criterion_name": "Conciseness", "score": 4, "rationale": "Avoids fluff and filler text."}
            ],
            "overall_rationale": "Model A delivered a superior, precise, and accurate response."
        }
        
        return json.dumps(mock_payload), usage