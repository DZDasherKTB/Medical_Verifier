from datasets import load_dataset
from indipendent_option_verification import ReasoningVerifier

import pandas as pd
import json
import re

from tqdm import tqdm


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
        "C": "...",
        "D": "..."
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

def clean_text(text):
    """
    Normalize text for robust comparison.
    """

    text = text.lower()

    # remove explanation section
    text = text.split("explanation:")[0]

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_correct_letter(answer_text, options_dict):
    """
    Match answer text to the correct MCQ option letter.
    """

    answer_clean = clean_text(answer_text)

    for letter, option in options_dict.items():

        option_clean = clean_text(option)

        # exact match
        if option_clean == answer_clean:
            return letter

    return None

# Load dataset
dataset = load_dataset("UCSC-VLAA/MedReason")

train_data = dataset["train"]


# Initialize verifier
verifier = ReasoningVerifier()


rows = []


# Process first 30 samples
for idx in tqdm(range(30)):

    sample = train_data[idx]

    try:

        dataset_id = sample["id_in_dataset"]

        question = sample["question"]

        raw_options = sample["options"]
        
        # Parse options
        options_dict = parse_options(raw_options)
        
        ground_truth_answer = find_correct_letter(sample["answer"],options_dict)

        ground_truth_reasoning = sample["reasoning"]

        # Store per-option outputs
        option_outputs = {}

        # Run verifier independently for each option
        for option_letter, option_text in options_dict.items():

            result = verifier.verify_option(
                question=question,
                option_letter=option_letter,
                option_text=option_text
            )

            option_outputs[option_letter] = result

        # Determine predicted YES option(s)
        predicted_yes = []

        for letter, output in option_outputs.items():

            if output.get("label", "").upper() == "YES":
                predicted_yes.append(letter)

        row = {

            "id_in_dataset": dataset_id,

            "question": question,

            "ground_truth_answer": ground_truth_answer,

            "ground_truth_reasoning": ground_truth_reasoning,

            "option_A": options_dict.get("A", ""),
            "option_B": options_dict.get("B", ""),
            "option_C": options_dict.get("C", ""),
            "option_D": options_dict.get("D", ""),

            "predicted_yes_options": ", ".join(predicted_yes),

            # A
            "A_label": option_outputs.get(
                "A",
                {}
            ).get("label", ""),

            "A_reason": option_outputs.get(
                "A",
                {}
            ).get("reason", ""),

            "A_reasoning_trace": json.dumps(
                option_outputs.get(
                    "A",
                    {}
                ).get("reasoning_trace", []),
                ensure_ascii=False,
                indent=2
            ),

            # B
            "B_label": option_outputs.get(
                "B",
                {}
            ).get("label", ""),

            "B_reason": option_outputs.get(
                "B",
                {}
            ).get("reason", ""),

            "B_reasoning_trace": json.dumps(
                option_outputs.get(
                    "B",
                    {}
                ).get("reasoning_trace", []),
                ensure_ascii=False,
                indent=2
            ),

            # C
            "C_label": option_outputs.get(
                "C",
                {}
            ).get("label", ""),

            "C_reason": option_outputs.get(
                "C",
                {}
            ).get("reason", ""),

            "C_reasoning_trace": json.dumps(
                option_outputs.get(
                    "C",
                    {}
                ).get("reasoning_trace", []),
                ensure_ascii=False,
                indent=2
            ),

            # D
            "D_label": option_outputs.get(
                "D",
                {}
            ).get("label", ""),

            "D_reason": option_outputs.get(
                "D",
                {}
            ).get("reason", ""),

            "D_reasoning_trace": json.dumps(
                option_outputs.get(
                    "D",
                    {}
                ).get("reasoning_trace", []),
                ensure_ascii=False,
                indent=2
            )
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


# Create dataframe
df = pd.DataFrame(rows)


# Save Excel
output_file = "~/Medical_Verifier/mcq_solving/independent_outputs.xlsx"

df.to_excel(
    output_file,
    index=False
)

print(f"\nSaved to {output_file}")