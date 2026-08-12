## Overview
This repository provides an enterprise-grade evaluation pipeline that uses an LLM-as-Judge to evaluate language model outputs. It addresses common judge vulnerabilities—including **position bias**, **verbosity bias**, and **format instability**—by using structured parsing, dual-pass permutation testing, and statistical validation[cite: 1].

---

## Key Features & Mitigations

### 1. Robust Structured Output & JSON Repair
- **Pydantic V2 Schemas**: Enforces non-empty criteria list, valid 1–5 ranges, and structured reasoning[cite: 1].
- **Fallback JSON Extractor**: Uses regex-based pattern matching to extract JSON structures if markdown code blocks or conversational text are returned[cite: 1].

### 2. Bias Mitigation Strategies
- **Position Bias Mitigation**: Executes dual-pass evaluation ($A/B$ and $B/A$ swaps)[cite: 1]. If the judge's verdict flips, the result is flagged and declared a `Tie` to avoid false wins[cite: 1].
- **Verbosity Control**: The rubric penalizes long, low-information responses and measures word-count ratios across candidates[cite: 1].
- **Adversarial Probing**: Runs test cases with verbose-but-wrong vs. terse-but-correct responses to test judge accuracy[cite: 1].

### 3. Statistical Validation Metrics
- **Cohen’s Kappa ($\kappa$)**: Measures agreement between the automated judge and human gold-standard annotations[cite: 1].
- **Position Flip Rate**: Measures how often position swaps change the judge's decision[cite: 1].
- **Audit Logging**: Saves all prompts, completion responses, and token counts for reproducibility[cite: 1].

---

## Architectural Analysis & Release-Gating

### Would I let this pipeline gate a production release?
**Yes, with specific guardrails.**

1. **Gating Conditions**:
   - The test suite must achieve **Cohen’s Kappa $\kappa \ge 0.70$** against human gold labels[cite: 1].
   - Position Flip Rate must remain **$< 8.0\%$** across the benchmark suite[cite: 1].
   - Adversarial Probe Resistance must achieve **$\ge 90\%$** accuracy on verbose-wrong test probes[cite: 1].

2. **When to fall back to Human Review**:
   - Any evaluation where $A/B$ dual-pass position checks yield a **Position Flip**[cite: 1].
   - Cases where judge confidence drops below $0.75$[cite: 1].

---

## How to Run
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install pydantic

# 2. Run Pipeline & Generate Audit Report
python pipeline.py --test-suite data/test_suite.json --output evaluation_report.json