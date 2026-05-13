from datasets import load_dataset
from united_mcq_verification import ReasoningVerifier

import pandas as pd
import json
import re
import os

from tqdm import tqdm

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


def parse_options(options_text: str):

    """
    Converts:

    Answer Choices:
    A. xxx
    B. yyy
    C. zzz
    D. ppp

    into:
    {
        "A": "...",
        "B": "...",
        ...
    }
    """

    pattern = r"([A-D])\.\s*(.*?)(?=\n[A-D]\.|$)"

    matches = re.findall(
        pattern,
        options_text,
        flags=re.DOTALL
    )

    parsed = {}

    for key, value in matches:
        parsed[key] = value.strip()

    return parsed


def clean_excel_text(value):
    """
    Remove illegal Excel characters.
    """

    if isinstance(value, str):

        value = ILLEGAL_CHARACTERS_RE.sub("", value)

    return value


# Load dataset
dataset = load_dataset("UCSC-VLAA/MedReason")

train_data = dataset["train"]


# Initialize verifier
verifier = ReasoningVerifier()


rows = []


# Output path
output_file = (
    "MCQ_Verification_Results/"
    "united_option_verification_results.xlsx"
)

# Create directory if missing
os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)


# Process samples
for idx in tqdm(range(1000)):

    sample = train_data[idx]

    try:

        dataset_id = sample["id_in_dataset"]

        question = sample["question"]

        answer = sample["answer"]

        reasoning_gt = sample["reasoning"]

        raw_options = sample["options"]

        # Parse options string
        options_dict = parse_options(raw_options)

        options = [
            options_dict.get("A", ""),
            options_dict.get("B", ""),
            options_dict.get("C", ""),
            options_dict.get("D", "")
        ]

        # Find ground truth answer letter
        ground_truth_answer_letter = ""

        answer_clean = answer.lower()

        if "explanation:" in answer_clean:
            answer_clean = answer_clean.split("explanation:")[0]

        answer_clean = answer_clean.strip().rstrip(".")

        for key, value in options_dict.items():

            option_clean = value.lower().strip().rstrip(".")

            if (
                option_clean in answer_clean
                or answer_clean in option_clean
            ):

                ground_truth_answer_letter = key
                break

        # Run model
        response = verifier.solve_mcq(
            question=question,
            options=options
        )

        options_analysis = response.get(
            "options_analysis",
            {}
        )

        row = {

            "id_in_dataset": dataset_id,

            "question": question,

            "ground_truth_answer": answer,

            "ground_truth_answer_letter":
                ground_truth_answer_letter,

            "ground_truth_reasoning": reasoning_gt,

            "option_A": options_dict.get("A", ""),
            "option_B": options_dict.get("B", ""),
            "option_C": options_dict.get("C", ""),
            "option_D": options_dict.get("D", ""),

            "selected_answer_letter": response.get(
                "selected_answer",
                ""
            ),

            "selected_option_text": response.get(
                "selected_option_text",
                ""
            ),

            "reasoning_trace": json.dumps(
                response.get(
                    "reasoning_trace",
                    []
                ),
                ensure_ascii=False,
                indent=2
            ),

            "A_label": options_analysis.get(
                "A",
                {}
            ).get("label", ""),

            "A_reason": options_analysis.get(
                "A",
                {}
            ).get("reason", ""),

            "B_label": options_analysis.get(
                "B",
                {}
            ).get("label", ""),

            "B_reason": options_analysis.get(
                "B",
                {}
            ).get("reason", ""),

            "C_label": options_analysis.get(
                "C",
                {}
            ).get("label", ""),

            "C_reason": options_analysis.get(
                "C",
                {}
            ).get("reason", ""),

            "D_label": options_analysis.get(
                "D",
                {}
            ).get("label", ""),

            "D_reason": options_analysis.get(
                "D",
                {}
            ).get("reason", "")
        }

        rows.append(row)

        print(f"[{dataset_id}] DONE")

    except Exception as e:

        print(f"[ERROR] {e}")

        rows.append({

            "id_in_dataset": sample.get(
                "id_in_dataset",
                "unknown"
            ),

            "error": str(e)
        })

    # SAVE AFTER EVERY RESPONSE

    try:

        df = pd.DataFrame(rows)

        # Remove illegal Excel characters
        df = df.applymap(clean_excel_text)

        df.to_excel(
            output_file,
            index=False
        )

    except Exception as save_error:

        print(f"[SAVE ERROR] {save_error}")


print(f"\nSaved to {output_file}")