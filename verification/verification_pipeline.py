# run_verification_pipeline.py

import pandas as pd
import json

from verification.reasoning_verifier import ReasoningVerifier
from verification.proposition_verifier import PropositionVerifier

import re

def extract_qid_number(qid):

    if pd.isna(qid):
        return None

    qid = str(qid)

    match = re.search(r"(\d+)", qid)

    if match:
        return match.group(1)

    return None
# =====================================
# LOAD EXCEL
# =====================================

EXCEL_PATH = "data/manual_data/manually_extracted_propositions_51_100.xlsx"

hypothesis_df = pd.read_excel(
    EXCEL_PATH,
    sheet_name="Hypothesis"
)

proposition_df = pd.read_excel(
    EXCEL_PATH,
    sheet_name="Propositions"
)

reasoning_df = pd.read_excel(
    EXCEL_PATH,
    sheet_name="Reasoning"
)


# =====================================
# FIX MERGED CELLS / EMPTY qn_ids
# =====================================
# Normalize IDs

hypothesis_df["qid_num"] = (
    hypothesis_df["qn_id"]
    .apply(extract_qid_number)
)

proposition_df["qid_num"] = (
    proposition_df["ReH"]
    .apply(extract_qid_number)
)

reasoning_df["qid_num"] = (
    reasoning_df["qn_id"]
    .apply(extract_qid_number)
)


# =====================================
# INIT VERIFIERS
# =====================================

reasoning_verifier = ReasoningVerifier()
proposition_verifier = PropositionVerifier()


# =====================================
# STORAGE
# =====================================

results = []


# =====================================
# PROCESS QUESTION BY QUESTION
# =====================================

unique_qns = hypothesis_df["qn_id"].unique()

for qn_id in unique_qns:

    print(f"\nProcessing {qn_id}")

    # ---------------------------------
    # GET REASONING TRACE
    # ---------------------------------

    reasoning_rows = reasoning_df[
        reasoning_df["qn_id"] == qn_id
    ]

    if len(reasoning_rows) == 0:
        print(f"Skipping {qn_id} - no reasoning trace")
        continue

    reasoning_trace = str(
        reasoning_rows.iloc[0]["reasoning_trace"]
    )

    # ---------------------------------
    # GET ALL HYPOTHESES
    # ---------------------------------

    hyp_rows = hypothesis_df[
        hypothesis_df["qn_id"] == qn_id
    ]

    hypotheses = []

    gt_answers = {}

    for _, row in hyp_rows.iterrows():

        hyp_id = str(row["hyp_id"]).strip()
        hyp_text = str(row["hypothesis"]).strip()

        hypotheses.append(
            f"{hyp_id}: {hyp_text}"
        )

        gt_answers[hyp_id] = row["answer"]

    # ---------------------------------
    # GET ALL PROPOSITIONS
    # ---------------------------------

    prop_rows = proposition_df[
        proposition_df["ReH"] == qn_id
    ]

    proposition_lines = []

    for _, row in prop_rows.iterrows():

        prop_id = str(row["prop_id"]).strip()
        prop_text = str(row["proposition"]).strip()

        proposition_lines.append(
            f"{prop_id} {prop_text}"
        )

    propositions = "\n".join(proposition_lines)

    # =================================
    # RUN REASONING VERIFIER
    # =================================

    reasoning_result = reasoning_verifier.verify(
        reasoning_trace=reasoning_trace,
        hypotheses=hypotheses
    )

    # =================================
    # RUN PROPOSITION VERIFIER
    # =================================

    proposition_result = proposition_verifier.verify(
        propositions=propositions,
        hypotheses=hypotheses
    )

    # =================================
    # STORE RESULTS
    # =================================

    for hyp in hypotheses:

        hyp_id = hyp.split(":")[0]

        reasoning_output = reasoning_result.get(
            hyp_id,
            {}
        )

        proposition_output = proposition_result.get(
            hyp_id,
            {}
        )

        results.append({

            "qn_id": qn_id,

            "hyp_id": hyp_id,

            "hypothesis": hyp,

            "ground_truth":
                gt_answers.get(hyp_id),

            # -------------------------
            # Reasoning verifier
            # -------------------------

            "reasoning_label":
                reasoning_output.get("label"),

            "reasoning_evidence":
                json.dumps(
                    reasoning_output.get(
                        "evidence",
                        []
                    )
                ),

            # -------------------------
            # Proposition verifier
            # -------------------------

            "proposition_label":
                proposition_output.get("label"),

            "proposition_evidence":
                json.dumps(
                    proposition_output.get(
                        "evidence",
                        []
                    )
                )
        })

    print(f"Done {qn_id}")


# =====================================
# SAVE OUTPUT
# =====================================

results_df = pd.DataFrame(results)

OUTPUT_PATH = "verification_results_51_100.xlsx"

results_df.to_excel(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved to {OUTPUT_PATH}")