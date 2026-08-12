import re
import json
import logging
from typing import Tuple, Optional
from src.schemas import PairwiseVerdict, TestCase, TokenUsage
from src.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class LLMJudgeEngine:
    def __init__(self, client: BaseLLMClient, rubric_path: str = "config/rubric.json"):
        self.client = client
        self.rubric = self._load_rubric(rubric_path)

    def _load_rubric(self, path: str) -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback default rubric
            return {
                "Correctness": "Factual accuracy and complete resolution of user query.",
                "Faithfulness": "Groundedness in context without hallucinated assertions.",
                "Instruction_Following": "Strict adherence to format, length, and constraints.",
                "Conciseness": "High information density without unnecessary padding."
            }

    def _construct_pairwise_prompt(self, test_case: TestCase, out_a: str, out_b: str) -> Tuple[str, str]:
        system_prompt = (
            "You are an expert AI Evaluation System. Compare Model A and Model B outputs objectively.\n"
            "You must return ONLY a JSON object conforming to the requested schema.\n"
            "Avoid favoring longer outputs or preferring the first option. Base judgment strictly on the rubric."
        )
        
        rubric_str = "\n".join([f"- {k}: {v}" for k, v in self.rubric.items()])
        
        user_prompt = f"""
### EVALUATION RUBRIC
{rubric_str}

### INPUT PROMPT
{test_case.input_prompt}

{"### EXPECTED REFERENCE OUTPUT" if test_case.expected_output else ""}
{test_case.expected_output if test_case.expected_output else ""}

### MODEL A OUTPUT
{out_a}

### MODEL B OUTPUT
{out_b}

### INSTRUCTIONS
1. Score each output on a 1-5 scale per criterion.
2. Provide grounded, sentence-level reasoning.
3. Select the overall winner ("Model A", "Model B", or "Tie").
4. Output MUST be valid JSON adhering to:
{{
  "winner": "Model A" | "Model B" | "Tie",
  "confidence": 0.0-1.0,
  "criteria_scores": [
    {{"criterion_name": "string", "score": int, "rationale": "string"}}
  ],
  "overall_rationale": "string"
}}
"""
        return system_prompt, user_prompt

    def _parse_raw_json_with_fallback(self, raw_text: str) -> PairwiseVerdict:
        """
        Multi-tier robust JSON repair mechanism for handling malformed outputs.
        """
        # Tier 1: Direct JSON parsing
        try:
            data = json.loads(raw_text)
            return PairwiseVerdict.model_validate(data)
        except Exception:
            pass

        # Tier 2: Extract code block ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return PairwiseVerdict.model_validate(data)
            except Exception:
                pass

        # Tier 3: Extract outermost brace matching
        brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(1))
                return PairwiseVerdict.model_validate(data)
            except Exception:
                pass

        raise ValueError(f"Failed to parse structured verdict from raw response: {raw_text[:100]}...")

    async def evaluate_case(self, test_case: TestCase, model_a_text: str, model_b_text: str) -> Tuple[PairwiseVerdict, TokenUsage, str]:
        sys_p, user_p = self._construct_pairwise_prompt(test_case, model_a_text, model_b_text)
        raw_resp, usage = await self.client.generate(user_p, sys_p)
        verdict = self._parse_raw_json_with_fallback(raw_resp)
        return verdict, usage, user_p