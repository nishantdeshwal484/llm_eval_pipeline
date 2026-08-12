import asyncio
from typing import Dict, Any, Tuple
from src.schemas import TestCase, WinnerEnum, PairwiseVerdict
from src.judge import LLMJudgeEngine


class BiasMitigationEngine:
    def __init__(self, judge_engine: LLMJudgeEngine):
        self.judge = judge_engine

    async def evaluate_with_position_mitigation(self, test_case: TestCase) -> Dict[str, Any]:
        """
        Position Bias Mitigation: Executes dual-pass evaluation (A/B and B/A).
        Measures consistency and flags position flips.
        """
        # Pass 1: Original Order (Model A = Candidate A, Model B = Candidate B)
        task_fwd = self.judge.evaluate_case(test_case, test_case.model_a_output, test_case.model_b_output)
        
        # Pass 2: Swapped Order (Model A = Candidate B, Model B = Candidate A)
        task_rev = self.judge.evaluate_case(test_case, test_case.model_b_output, test_case.model_a_output)

        (v_fwd, usage_fwd, prompt_fwd), (v_rev, usage_rev, prompt_rev) = await asyncio.gather(task_fwd, task_rev)

        # Map Reverse Verdict back to original identities
        mapped_reverse = WinnerEnum.TIE
        if v_rev.winner == WinnerEnum.MODEL_A:
            mapped_reverse = WinnerEnum.MODEL_B
        elif v_rev.winner == WinnerEnum.MODEL_B:
            mapped_reverse = WinnerEnum.MODEL_A

        # Check for Position Flip
        is_flip = v_fwd.winner != mapped_reverse

        # Consensus determination
        if not is_flip:
            consensus_winner = v_fwd.winner
        else:
            # If position flipped, resolve via confidence or default to Tie
            consensus_winner = WinnerEnum.TIE

        total_tokens = usage_fwd.prompt_tokens + usage_fwd.completion_tokens + usage_rev.prompt_tokens + usage_rev.completion_tokens

        return {
            "test_id": test_case.test_id,
            "forward_winner": v_fwd.winner.value,
            "reverse_winner_mapped": mapped_reverse.value,
            "position_flip_detected": is_flip,
            "consensus_winner": consensus_winner.value,
            "fwd_verdict": v_fwd.model_dump(),
            "rev_verdict": v_rev.model_dump(),
            "token_usage": total_tokens,
            "audit_log": {
                "forward_prompt": prompt_fwd,
                "reverse_prompt": prompt_rev
            }
        }

    @staticmethod
    def calculate_length_ratio(test_case: TestCase) -> float:
        """Calculates character length ratio between outputs (Model A / Model B)."""
        len_a = len(test_case.model_a_output)
        len_b = len(test_case.model_b_output)
        return len_a / max(len_b, 1)