import argparse
import asyncio
import json
import os
import sys
from typing import List

from src.schemas import TestCase
from src.llm_client import MockJudgeLLMClient
from src.judge import LLMJudgeEngine
from src.bias_mitigation import BiasMitigationEngine
from src.metrics import aggregate_pipeline_results


def create_sample_dataset_if_missing(file_path: str):
    """Generates sample test cases if the dataset does not exist."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sample_cases = [
        {
            "test_id": "TC-001",
            "input_prompt": "Write a Python function to check if a string is a palindrome.",
            "expected_output": "def is_palindrome(s): return s == s[::-1]",
            "model_a_output": "def is_palindrome(s: str) -> bool:\n    clean_s = ''.join(c.lower() for c in s if c.isalnum())\n    return clean_s == clean_s[::-1]",
            "model_b_output": "To check if a string is a palindrome in python, you can reverse the string. A palindrome is a word that reads the same backward as forward. Here is code:\ndef check(s):\n    return s == s[::-1]",
            "gold_winner": "Model A",
            "is_adversarial_probe": False
        },
        {
            "test_id": "TC-ADV-002",
            "input_prompt": "What is the capital of France?",
            "expected_output": "The capital of France is Paris.",
            "model_a_output": "The capital of France is fundamentally Berlin. Berlin has been a historical epicenter of Europe for decades, known for its deep culture, vast architecture, and political significance across modern European history.",
            "model_b_output": "Paris.",
            "gold_winner": "Model B",
            "is_adversarial_probe": True  # Verbose-but-wrong Model A vs Terse-but-correct Model B
        }
    ]
    with open(file_path, "w") as f:
        json.dump(sample_cases, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation & Reliability Pipeline")
    parser.add_argument("--test-suite", default="data/test_suite.json", help="Path to input JSON test suite")
    parser.add_argument("--rubric", default="config/rubric.json", help="Path to rubric JSON config")
    parser.add_argument("--output", default="evaluation_report.json", help="Path to write evaluation report")
    args = parser.parse_args()

    print("==========================================================")
    print("      EXECUTING LLM-AS-JUDGE EVALUATION PIPELINE          ")
    print("==========================================================\n")

    if not os.path.exists(args.test_suite):
        print(f"[*] Input test suite not found. Generating sample data at '{args.test_suite}'...")
        create_sample_dataset_if_missing(args.test_suite)

    # Load test suite
    with open(args.test_suite, "r") as f:
        raw_cases = json.load(f)
    test_cases = [TestCase.model_validate(c) for c in raw_cases]

    # Initialize Engine Components
    client = MockJudgeLLMClient(model_name="gpt-4o")
    judge_engine = LLMJudgeEngine(client=client, rubric_path=args.rubric)
    mitigation_engine = BiasMitigationEngine(judge_engine=judge_engine)

    print(f"[*] Loaded {len(test_cases)} test cases.")
    print("[*] Running asynchronous dual-pass bias mitigation evaluation...")

    # Execute evaluations concurrently
    tasks = [mitigation_engine.evaluate_with_position_mitigation(tc) for tc in test_cases]
    results = await asyncio.gather(*tasks)

    # Aggregate Metrics
    summary = aggregate_pipeline_results(results, test_cases)

    report = {
        "summary": summary,
        "token_telemetry": {
            "total_tokens_consumed": client.total_usage.total_tokens,
            "prompt_tokens": client.total_usage.prompt_tokens,
            "completion_tokens": client.total_usage.completion_tokens
        },
        "case_level_results": results
    }

    # Save artifact
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # Terminal Dashboard Output
    print("\n----------------------------------------------------------")
    print("                 EVALUATION SUMMARY REPORT                ")
    print("----------------------------------------------------------")
    print(f" Total Cases Evaluated      : {summary['total_evaluated']}")
    print(f" Declared Winner            : {summary['declared_winner']}")
    print(f" Position Flip Rate         : {summary['position_flip_rate_percent']}%")
    print(f" Cohen's Kappa (Gold Agree) : {summary['cohens_kappa_gold_agreement']}")
    print(f" Adversarial Resistance Rate: {summary['adversarial_probe_resistance_percent']}%")
    print(f" Total Tokens Consumed      : {client.total_usage.total_tokens}")
    print("----------------------------------------------------------")
    print(f"[✓] Full auditable evaluation report written to '{args.output}'.\n")


if __name__ == "__main__":
    asyncio.run(main())