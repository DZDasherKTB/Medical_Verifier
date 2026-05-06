# Medical Verifier

LLM-based framework for verifying medical hypotheses using:

* Free-form reasoning traces
* Structured propositions

Uses Groq-hosted models for inference.

---

## Structure

```bash
.
├── data
│   └── manual_data
├── testing
│   └── test_verifiers.py
├── verification
│   ├── proposition_verifier.py
│   ├── reasoning_verifier.py
│   └── verification_pipeline.py
├── requirements.txt
└── verification_results_1_50.xlsx
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_api_key
```

---

## Run

```bash
python verification/verification_pipeline.py
```

The pipeline:

* Loads reasoning traces, propositions, and hypotheses from Excel
* Runs:

  * Reasoning Verifier
  * Proposition Verifier
* Saves results into a new Excel file

---

## Dataset Format

### Hypothesis Sheet

| qn_id | hyp_id | hypothesis | answer |

### Propositions Sheet

| ReH | prop_id | proposition |

### Reasoning Sheet

| qn_id | reasoning_trace |

---

## Models

Currently uses:

* `llama-3.3-70b-versatile`
* Groq API

---

## Goal

Medical reasoning verification through structured proposition-based evaluation and reasoning-trace verification.
