import os
import json
import re
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from datasets import load_dataset
from triplet_verification import TripletVerifier 


def parse_options(options_text):
    """
    Extract MCQ options into dictionary:
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
    return {
        key: value.strip()
        for key, value in matches
    }


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


# =========================
# LOAD DATASET
# =========================

dataset = load_dataset("UCSC-VLAA/MedReason")
train_data = dataset["train"]
verifier = TripletVerifier()
rows = []

# =========================
# MAIN EVALUATION LOOP
# =========================

for idx in tqdm(range(1000)):
    sample = train_data[idx]

    try:
        dataset_id = sample["id_in_dataset"]
        question = sample["question"]
        raw_answer = sample["answer"]

        answer_text = raw_answer.split("Explanation:")[0].strip()
        options_text = sample["options"]
        
        options_dict = parse_options(options_text)

        # DEBUG
        print("\n====================")
        print(f"QUESTION {idx}")
        print("====================")

        print("\nANSWER TEXT:")
        print(answer_text)

        print("\nOPTIONS:")
        print(options_dict)

        gt_letter = find_correct_letter(
            answer_text,
            options_dict
        )

        if gt_letter is None:
            print(f"\nCould not match answer at index {idx}")
            continue

        print(f"\nMATCHED GT LETTER: {gt_letter}")

        gt_text = options_dict[gt_letter]

        distractor_letters = [
            l for l in options_dict.keys()
            if l != gt_letter
        ]

        if len(distractor_letters) < 2:
            print(f"\nNot enough distractors to form a triplet for index {idx}. Skipping.")
            continue

        print("DISTRACTORS:", distractor_letters)

        row = {
            "id": dataset_id,
            "question": question,
            "ground_truth_letter": gt_letter,
            "ground_truth_text": gt_text
        }

        # =========================
        # TRIPLET COMPARISONS
        # =========================
        # Generate all combinations of 2 distractors + 1 GT
        
        distractor_pairs = list(combinations(distractor_letters, 2))

        for dist_pair in distractor_pairs:
            dist1_letter, dist2_letter = dist_pair
            dist1_text = options_dict[dist1_letter]
            dist2_text = options_dict[dist2_letter]

            print(f"\nRunning Triplet: {gt_letter} vs {dist1_letter} vs {dist2_letter}")

            result = verifier.verify_triplet(
                question,
                gt_letter, gt_text,
                dist1_letter, dist1_text,
                dist2_letter, dist2_text
            )

            col_prefix = f"vs_{dist1_letter}_{dist2_letter}"

            selected = result.get("selected_letter")
            is_correct = (selected == gt_letter)

            row[f"{col_prefix}_selected"] = selected
            row[f"{col_prefix}_is_correct"] = is_correct
            row[f"{col_prefix}_trace"] = json.dumps(
                result.get("reasoning_trace"),
                indent=2
            )
            row[f"{col_prefix}_justification"] = result.get("justification")

        rows.append(row)

    except Exception as e:
        print(f"\nError at index {idx}: {e}")

# =========================
# CREATE DATAFRAME
# =========================

df = pd.DataFrame(rows)

# =========================
# TRIPLET ACCURACY
# =========================

triplet_cols = [
    c for c in df.columns
    if c.endswith("_is_correct")
]

correct = 0
total = 0

for col in triplet_cols:
    valid_values = df[col].dropna()
    correct += valid_values.sum()
    total += len(valid_values)

triplet_accuracy = (
    correct / total
    if total > 0 else 0
)

print("\n====================")
print("TRIPLET RESULTS")
print("====================")
print(f"Total Triplet Comparisons: {total}")
print(f"Correct Triplet Decisions: {correct}")
print(f"Triplet Accuracy: {triplet_accuracy:.4f}")

# =========================
# QUESTION-LEVEL ACCURACY
# =========================

full_correct = 0

for _, row in df.iterrows():
    valid_cols = [
        c for c in triplet_cols
        if pd.notna(row[c])
    ]

    if len(valid_cols) == 0:
        continue

    # A question is fully correct only if the GT wins in EVERY triplet combination
    if all(row[c] == True for c in valid_cols):
        full_correct += 1

question_accuracy = (
    full_correct / len(df)
    if len(df) > 0 else 0
)

print("\n====================")
print("QUESTION-LEVEL RESULTS")
print("====================")
print(f"Questions Fully Correct: {full_correct}/{len(df)}")
print(f"Question-Level Accuracy: {question_accuracy:.4f}")

# =========================
# SAVE EXCEL
# =========================

output_path = os.path.expanduser(
    "~/Medical_Verifier/mcq_solving/triplet_results_independent_style.xlsx"
)

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_excel(
    output_path,
    index=False
)

print("\nEvaluation complete.")
print(f"Results saved to:\n{output_path}")