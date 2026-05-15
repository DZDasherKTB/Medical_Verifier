"""
triplet_evaluation.py  —  Trio / 3-Option Comparative Verification Evaluator
=============================================================================
Experiment formulation:
    (question, [GT + 2 distractors]) -> winner (original MCQ letter)

Schema (confirmed from real data):
    Slot A  = ALWAYS the GT option text     (ground_truth_text)
    Slot B  = Distractor 1 text             (distractor1_text)
    Slot C  = Distractor 2 text             (distractor2_text)
    predicted_letter = original MCQ letter (A/B/C/D), not the slot letter
    is_correct       = (predicted_letter == ground_truth_letter)
    A_label / B_label / C_label = model YES/NO label per slot

Each MCQ with 4 options generates 3 triplet rows (GT vs every pair of distractors):
    [GT, D1, D2],  [GT, D1, D3],  [GT, D2, D3]

Binary normalization per row (3 decisions per row):
    Slot A (GT)  -> gt_binary=1, pred_binary = 1 if predicted == gt_letter  else 0
    Slot B (D1)  -> gt_binary=0, pred_binary = 1 if predicted == d1_letter  else 0
    Slot C (D2)  -> gt_binary=0, pred_binary = 1 if predicted == d2_letter  else 0

Special cases handled:
    - NaN predicted_letter   (API failures / multi-answer refusals -> 18 rows)
    - Multi-answer raw responses (e.g. "A and B", "A,B,C")
    - Questions with 1 or 2 triplet rows (incomplete coverage)
    - A_option != gt_text edge cases (minor typo variants)

MCQ reconstruction:
    Strict  = all triplet rows for this question won by GT (is_correct all True)
    Relaxed = at least 1 triplet row won by GT

Usage
-----
    python triplet_evaluation.py [--input FILE] [--output FILE] [--quiet]
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(description="Triplet 3-option comparative evaluation")
parser.add_argument(
    "--input",
    default="./MCQ_Verification_Results/trio_option_experiment.xlsx",
    help="Path to the input Excel file",
)
parser.add_argument(
    "--output",
    default="triplet_evaluation_results.xlsx",
    help="Path to the output Excel file",
)
parser.add_argument("--quiet", action="store_true", help="Suppress console report")
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output
QUIET = args.quiet

# ============================================================
# COLUMN NAMES
# ============================================================

DATASET_ID_COL    = "dataset_id"
QUESTION_COL      = "question"
GT_LETTER_COL     = "ground_truth_letter"
GT_TEXT_COL       = "ground_truth_text"
D1_LETTER_COL     = "distractor1_letter"
D1_TEXT_COL       = "distractor1_text"
D2_LETTER_COL     = "distractor2_letter"
D2_TEXT_COL       = "distractor2_text"
VISIBLE_COL       = "visible_options"
PRED_LETTER_COL   = "predicted_letter"
RAW_ANSWER_COL    = "raw_selected_answer"
SEL_TEXT_COL      = "selected_option_text"
IS_CORRECT_COL    = "is_correct"
REASONING_COL     = "reasoning_trace"

# Slot columns  (A=GT, B=D1, C=D2 — always, regardless of original MCQ letter)
A_OPTION_COL = "A_option"
A_LABEL_COL  = "A_label"
A_REASON_COL = "A_reason"
B_OPTION_COL = "B_option"
B_LABEL_COL  = "B_label"
B_REASON_COL = "B_reason"
C_OPTION_COL = "C_option"
C_LABEL_COL  = "C_label"
C_REASON_COL = "C_reason"

# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_letter(x):
    v = clean(x).upper()
    return v if v in ("A", "B", "C", "D") else None


def clean_label(x):
    v = clean(x).upper()
    return v if v in ("YES", "NO") else None


def safe_div(num, den):
    return num / den if den else float("nan")


def is_multi_answer(raw):
    """True if raw_selected_answer looks like a multi-selection."""
    s = clean(raw)
    return bool(s and (
        "," in s or
        " and " in s.lower() or
        " & " in s
    ))


def binary_metrics(y_true, y_pred):
    """Return full binary classification metric dict."""
    if len(set(y_true)) < 2:
        return {"error": "single class — metrics undefined"}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tp + tn + fp + fn
    po = (tp + tn) / total if total else float("nan")
    pe = (
        ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total ** 2)
        if total else float("nan")
    )
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "binary_accuracy":            accuracy_score(y_true, y_pred),
        "precision":                  precision_score(y_true, y_pred, zero_division=0),
        "recall":                     recall_score(y_true, y_pred, zero_division=0),
        "f1":                         f1_score(y_true, y_pred, zero_division=0),
        "specificity":                safe_div(tn, tn + fp),
        "false_positive_rate":        safe_div(fp, fp + tn),
        "false_negative_rate":        safe_div(fn, fn + tp),
        "negative_predictive_value":  safe_div(tn, tn + fn),
        "balanced_accuracy":          0.5 * (safe_div(tp, tp + fn) + safe_div(tn, tn + fp)),
        "matthews_corrcoef":          safe_div(tp * tn - fp * fn, mcc_denom),
        "cohens_kappa":               safe_div(po - pe, 1 - pe),
    }

# ============================================================
# LOAD & VALIDATE
# ============================================================

print(f"Loading {INPUT_FILE} ...")
try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    sys.exit(f"[ERROR] File not found: {INPUT_FILE}")

required_cols = {
    QUESTION_COL, DATASET_ID_COL,
    GT_LETTER_COL, GT_TEXT_COL,
    D1_LETTER_COL, D2_LETTER_COL,
    PRED_LETTER_COL, IS_CORRECT_COL,
    A_OPTION_COL, A_LABEL_COL,
    B_OPTION_COL, B_LABEL_COL,
    C_OPTION_COL, C_LABEL_COL,
}
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"[ERROR] Missing columns: {missing}")

raw_rows = len(df)

# ============================================================
# CLEAN
# ============================================================

for col in [GT_LETTER_COL, D1_LETTER_COL, D2_LETTER_COL]:
    df[col] = df[col].apply(clean_letter)

df[PRED_LETTER_COL] = df[PRED_LETTER_COL].apply(clean_letter)  # None if NaN or multi

for col in [GT_TEXT_COL, D1_TEXT_COL, D2_TEXT_COL,
            A_OPTION_COL, B_OPTION_COL, C_OPTION_COL]:
    if col in df.columns:
        df[col] = df[col].apply(clean)

for col in [A_LABEL_COL, B_LABEL_COL, C_LABEL_COL]:
    df[col] = df[col].apply(clean_label)

df[IS_CORRECT_COL] = df[IS_CORRECT_COL].apply(
    lambda x: str(x).strip().lower() in ("true", "1", "yes")
)

# Detect multi-answer rows (raw_selected_answer has comma/and)
if RAW_ANSWER_COL in df.columns:
    df["is_multi_answer"] = df[RAW_ANSWER_COL].apply(is_multi_answer)
else:
    df["is_multi_answer"] = False

# ============================================================
# CATEGORISE ROWS
# ============================================================

# Error row: predicted_letter is None (NaN or unparseable)
# This includes multi-answer rows (model refused to pick one)
df_error    = df[df[PRED_LETTER_COL].isna()].copy()
df_valid    = df[df[PRED_LETTER_COL].notna()].copy()

error_rows  = len(df_error)
multi_rows  = int(df_error["is_multi_answer"].sum()) if "is_multi_answer" in df_error.columns else 0
nan_rows    = error_rows - multi_rows

# Additional check: GT/D1/D2 letters must all be valid
mask_valid_letters = (
    df_valid[GT_LETTER_COL].notna() &
    df_valid[D1_LETTER_COL].notna() &
    df_valid[D2_LETTER_COL].notna()
)
df_letter_err = df_valid[~mask_valid_letters].copy()
df_valid      = df_valid[mask_valid_letters].copy()

letter_err_rows = len(df_letter_err)
if letter_err_rows:
    print(f"[WARN] {letter_err_rows} rows dropped: missing GT/D1/D2 letter.")

valid_rows    = len(df_valid)
total_questions = df_valid[DATASET_ID_COL].nunique()

print(f"  {raw_rows} total rows | {error_rows} errors ({multi_rows} multi-answer, {nan_rows} NaN pred) | {valid_rows} valid")

# ============================================================
# FLATTEN TO BINARY ROWS
#
# Schema confirmed:
#   Slot A = GT option  (gt_binary = 1)
#   Slot B = Distractor 1  (gt_binary = 0)
#   Slot C = Distractor 2  (gt_binary = 0)
#   predicted_letter is original MCQ letter (A/B/C/D)
#   pred_binary for each slot = 1 if predicted_letter matches that slot's letter
# ============================================================

binary_rows = []

for _, row in df_valid.iterrows():
    gt_letter = row[GT_LETTER_COL]
    d1_letter = row[D1_LETTER_COL]
    d2_letter = row[D2_LETTER_COL]
    pred      = row[PRED_LETTER_COL]

    for slot, letter, gt_bin, opt_col, lbl_col in [
        ("A", gt_letter, 1, A_OPTION_COL, A_LABEL_COL),
        ("B", d1_letter, 0, B_OPTION_COL, B_LABEL_COL),
        ("C", d2_letter, 0, C_OPTION_COL, C_LABEL_COL),
    ]:
        pred_bin = 1 if pred == letter else 0

        if gt_bin == 1 and pred_bin == 1:
            outcome = "TP"
        elif gt_bin == 0 and pred_bin == 0:
            outcome = "TN"
        elif gt_bin == 0 and pred_bin == 1:
            outcome = "FP"
        else:
            outcome = "FN"

        # Label from model's slot-level response (for cross-check)
        slot_label = row[lbl_col]

        binary_rows.append({
            "dataset_id":     row[DATASET_ID_COL],
            "question":       row[QUESTION_COL],
            "option_letter":  letter,
            "option_text":    row[opt_col],
            "slot":           slot,
            "role":           "GT" if gt_bin else ("D1" if slot == "B" else "D2"),
            "gt_letter":      gt_letter,
            "d1_letter":      d1_letter,
            "d2_letter":      d2_letter,
            "pred_letter":    pred,
            "ground_truth":   "YES" if gt_bin else "NO",
            "prediction":     "YES" if pred_bin else "NO",
            "slot_label":     slot_label,   # model's label for this slot
            "binary_result":  outcome,
            "gt_binary":      gt_bin,
            "pred_binary":    pred_bin,
            "triplet_correct": row[IS_CORRECT_COL],
        })

binary_df = pd.DataFrame(binary_rows)

# ============================================================
# LABEL CONSISTENCY CHECK
# Compare pred_binary (from predicted_letter) vs slot_label (YES/NO on slot)
# They should agree. Inconsistencies flag model self-contradiction.
# ============================================================

binary_df["label_pred_binary"] = binary_df["slot_label"].apply(
    lambda x: 1 if x == "YES" else (0 if x == "NO" else None)
)
consistent_mask = binary_df["label_pred_binary"].notna()
n_checkable = consistent_mask.sum()
n_consistent = (
    binary_df.loc[consistent_mask, "pred_binary"] ==
    binary_df.loc[consistent_mask, "label_pred_binary"]
).sum()
label_inconsistencies = n_checkable - n_consistent

# ============================================================
# 1. TRIPLET NATIVE METRICS
# ============================================================

triplet_accuracy   = df_valid[IS_CORRECT_COL].mean()
total_triplets     = len(df_valid)

q_all_correct = df_valid.groupby(DATASET_ID_COL)[IS_CORRECT_COL].all()
q_any_correct = df_valid.groupby(DATASET_ID_COL)[IS_CORRECT_COL].any()
question_consistency = q_all_correct.mean()

# Pairs won per question
pairs_won_per_q = df_valid.groupby(DATASET_ID_COL)[IS_CORRECT_COL].sum()
pairs_won_dist  = pairs_won_per_q.value_counts().sort_index()

# GT dominance rate = triplet_accuracy (proportion of triplets won)
gt_dominance_rate = triplet_accuracy

# Distractor defeat rates
d1_wins   = df_valid[df_valid[IS_CORRECT_COL]].groupby(D1_LETTER_COL).size()
d1_total  = df_valid.groupby(D1_LETTER_COL).size()
d1_defeat = (d1_wins / d1_total).rename("D1_defeat_rate")

d2_wins   = df_valid[df_valid[IS_CORRECT_COL]].groupby(D2_LETTER_COL).size()
d2_total  = df_valid.groupby(D2_LETTER_COL).size()
d2_defeat = (d2_wins / d2_total).rename("D2_defeat_rate")

# Combined distractor defeat (letter appears in any distractor slot)
dist_all = pd.concat([
    df_valid[[D1_LETTER_COL, IS_CORRECT_COL]].rename(columns={D1_LETTER_COL: "dist_letter"}),
    df_valid[[D2_LETTER_COL, IS_CORRECT_COL]].rename(columns={D2_LETTER_COL: "dist_letter"}),
])
combined_defeat = (
    dist_all.groupby("dist_letter")[IS_CORRECT_COL]
    .agg(correct="sum", total="count")
    .assign(defeat_rate=lambda x: x["correct"] / x["total"])
)

# Distractor difficulty (errors caused)
distractor_errors_d1 = df_valid[~df_valid[IS_CORRECT_COL]].groupby(D1_LETTER_COL).size()
distractor_errors_d2 = df_valid[~df_valid[IS_CORRECT_COL]].groupby(D2_LETTER_COL).size()

# ============================================================
# 2. FLATTENED BINARY METRICS
# ============================================================

y_true = binary_df["gt_binary"].tolist()
y_pred = binary_df["pred_binary"].tolist()

global_bin      = binary_metrics(y_true, y_pred)
total_decisions = len(binary_df)

# Per original-option-letter
per_letter_metrics = {}
for letter in "ABCD":
    sub = binary_df[binary_df["option_letter"] == letter]
    if len(sub) < 2:
        continue
    per_letter_metrics[letter] = binary_metrics(
        sub["gt_binary"].tolist(), sub["pred_binary"].tolist()
    )

# Per role (GT / D1 / D2)
# NOTE: each role is a single class (GT always gt_binary=1; D1/D2 always gt_binary=0),
# so standard binary_metrics is undefined. Instead we compute role-specific rates:
#   GT  -> correct_rate = recall  (% of GT slots where model predicted YES)
#   D1  -> correct_rate = TNR     (% of D1 slots where model correctly predicted NO)
#   D2  -> correct_rate = TNR     (% of D2 slots where model correctly predicted NO)
#   pred_YES_rate = how often model said YES for that role
per_role_metrics = {}
for role in ["GT", "D1", "D2"]:
    sub = binary_df[binary_df["role"] == role]
    if len(sub) == 0:
        continue
    correct_rate  = (sub["prediction"] == sub["ground_truth"]).mean()
    pred_yes_rate = (sub["prediction"] == "YES").mean()
    per_role_metrics[role] = {
        "n":             len(sub),
        "correct_rate":  correct_rate,
        "pred_YES_rate": pred_yes_rate,
        "label":         "Recall (GT predicted YES)" if role == "GT" else "Specificity (distractor predicted NO)",
    }

# ============================================================
# 3. MCQ RECONSTRUCTION
# ============================================================

strict_correct  = int(q_all_correct.sum())
relaxed_correct = int(q_any_correct.sum())
strict_accuracy  = safe_div(strict_correct,  total_questions)
relaxed_accuracy = safe_div(relaxed_correct, total_questions)

question_level_records = []
for qid in df_valid[DATASET_ID_COL].unique():
    sub        = df_valid[df_valid[DATASET_ID_COL] == qid]
    n_triplets = len(sub)
    n_won      = int(sub[IS_CORRECT_COL].sum())
    gt_letter  = sub[GT_LETTER_COL].iloc[0]
    d1_letters = sorted(sub[D1_LETTER_COL].unique())
    d2_letters = sorted(sub[D2_LETTER_COL].unique())
    question_level_records.append({
        "dataset_id":   qid,
        "question":     sub[QUESTION_COL].iloc[0],
        "gt_letter":    gt_letter,
        "d1_letters":   ",".join(d1_letters),
        "d2_letters":   ",".join(d2_letters),
        "n_triplets":   n_triplets,
        "triplets_won": n_won,
        "all_correct":  n_won == n_triplets,
        "any_correct":  n_won > 0,
        "win_rate":     safe_div(n_won, n_triplets),
    })

question_level_df = pd.DataFrame(question_level_records)

# ============================================================
# 4. MULTI-YES / NONE ANALYSIS  (from slot labels A/B/C)
# ============================================================

# Per-triplet-row: count how many slots got YES from model
df_valid["n_yes_slots"] = (
    (df_valid[A_LABEL_COL] == "YES").astype(int) +
    (df_valid[B_LABEL_COL] == "YES").astype(int) +
    (df_valid[C_LABEL_COL] == "YES").astype(int)
)
yes_slot_dist   = df_valid["n_yes_slots"].value_counts().sort_index()
multi_yes_count = int((df_valid["n_yes_slots"] > 1).sum())
none_count      = int((df_valid["n_yes_slots"] == 0).sum())
single_count    = int((df_valid["n_yes_slots"] == 1).sum())

# ============================================================
# 5. CALIBRATION
# ============================================================

calibration_rows = []
for letter in "ABCD":
    rows = binary_df[binary_df["option_letter"] == letter]
    if len(rows) == 0:
        continue
    yes_pred = int(rows["pred_binary"].sum())
    yes_gt   = int(rows["gt_binary"].sum())
    n        = len(rows)
    calibration_rows.append({
        "option_letter":    letter,
        "appearances":      n,
        "times_as_GT":      yes_gt,
        "times_predicted":  yes_pred,
        "GT_rate":          safe_div(yes_gt,   n),
        "pred_rate":        safe_div(yes_pred, n),
        "calibration_error": abs(safe_div(yes_pred, n) - safe_div(yes_gt, n)),
    })
calibration_df = pd.DataFrame(calibration_rows)

# ============================================================
# 6. REASONING LENGTH
# ============================================================

avg_reasoning_len    = None
median_reasoning_len = None
if REASONING_COL in df_valid.columns:
    lens = df_valid[REASONING_COL].dropna().apply(lambda x: len(str(x)))
    if len(lens):
        avg_reasoning_len    = lens.mean()
        median_reasoning_len = lens.median()

# ============================================================
# 7. GT / PRED LETTER DISTRIBUTIONS
# ============================================================

gt_letter_dist   = df_valid[GT_LETTER_COL].value_counts()
pred_letter_dist = df_valid[PRED_LETTER_COL].value_counts()

# ============================================================
# PRINT REPORT
# ============================================================

SEP = "=" * 70
sep = "-" * 70

def pline(label, value, fmt=".4f"):
    if isinstance(value, float):
        print(f"  {label:<46} {value:{fmt}}")
    else:
        print(f"  {label:<46} {value}")

if not QUIET:
    print(f"\n{SEP}")
    print("  TRIPLET / 3-OPTION EVALUATION REPORT")
    print(SEP)

    print(f"\n[DATASET]")
    print(sep)
    pline("Input file",                  INPUT_FILE, "s")
    pline("Total rows (raw)",            raw_rows, "d")
    pline("Error rows (NaN predicted)",  nan_rows, "d")
    pline("Multi-answer rows",           multi_rows, "d")
    pline("Invalid letter rows",         letter_err_rows, "d")
    pline("Valid rows (triplets)",        valid_rows, "d")
    pline("Unique questions",            total_questions, "d")
    pline("Total binary decisions",      total_decisions, "d")
    print(f"\n  Layout: Slot A = GT option, Slot B = Distractor 1, Slot C = Distractor 2")
    print(f"  predicted_letter = original MCQ letter (A/B/C/D)")

    print(f"\n[1. TRIPLET NATIVE METRICS]")
    print(sep)
    pline("Triplet Accuracy",            triplet_accuracy)
    pline("GT Dominance Rate",           gt_dominance_rate)
    pline("Question Consistency (all won)", question_consistency)

    print(f"\n  Pairs Won per Question:")
    for k, v in pairs_won_dist.items():
        tag = "all correct" if k == 3 else ("all wrong" if k == 0 else "partial")
        print(f"    {int(k)}/3 pairs won: {v} questions  [{tag}]")

    print(f"\n  Combined Distractor Defeat Rate (any slot):")
    for letter, row_s in combined_defeat.iterrows():
        print(f"    {letter}  correct={int(row_s['correct'])}/{int(row_s['total'])}  "
              f"defeat_rate={row_s['defeat_rate']:.4f}")

    print(f"\n  Distractor Errors by Letter:")
    err_combined = dist_all[~dist_all[IS_CORRECT_COL]].groupby("dist_letter").size()
    for letter, cnt in err_combined.items():
        print(f"    {letter}: {int(cnt)} losses as distractor")

    print(f"\n[2. FLATTENED BINARY METRICS]")
    print(sep)
    print(f"  Each triplet row -> 3 binary decisions (GT + D1 + D2)")
    pline("Total binary decisions",      total_decisions, "d")
    pline("TP",  global_bin["TP"], "d")
    pline("TN",  global_bin["TN"], "d")
    pline("FP",  global_bin["FP"], "d")
    pline("FN",  global_bin["FN"], "d")
    print()
    pline("Binary Accuracy",             global_bin["binary_accuracy"])
    pline("Precision",                   global_bin["precision"])
    pline("Recall (Sensitivity / TPR)",  global_bin["recall"])
    pline("Specificity (TNR)",           global_bin["specificity"])
    pline("F1 Score",                    global_bin["f1"])
    pline("Balanced Accuracy",           global_bin["balanced_accuracy"])
    pline("MCC",                         global_bin["matthews_corrcoef"])
    pline("Cohen's Kappa",               global_bin["cohens_kappa"])
    pline("NPV",                         global_bin["negative_predictive_value"])
    pline("False Positive Rate",         global_bin["false_positive_rate"])
    pline("False Negative Rate",         global_bin["false_negative_rate"])

    print(f"\n  Per-Role Correct Rate (GT / D1 / D2):")
    print(f"    (Each role is a single class, so standard binary metrics don't apply)")
    print(f"    GT  correct_rate = Recall:       did the model pick GT when it was correct?")
    print(f"    D1/D2 correct_rate = Specificity: did the model reject the distractor?")
    for role in ["GT", "D1", "D2"]:
        m = per_role_metrics.get(role, {})
        if not m:
            continue
        print(f"    {role}: n={m['n']}  correct_rate={m['correct_rate']:.4f}  pred_YES_rate={m['pred_YES_rate']:.4f}")

    print(f"\n  Per-Option-Letter Binary Metrics:")
    for letter in "ABCD":
        m = per_letter_metrics.get(letter, {})
        if not m or "error" in m:
            continue
        print(f"    {letter}: "
              f"Acc={m['binary_accuracy']:.4f}  "
              f"Prec={m['precision']:.4f}  "
              f"Rec={m['recall']:.4f}  "
              f"Spec={m['specificity']:.4f}  "
              f"F1={m['f1']:.4f}")

    print(f"\n[3. MCQ RECONSTRUCTION METRICS]")
    print(sep)
    print(f"  Strict  = all 3 triplet comparisons won by GT")
    print(f"  Relaxed = at least 1 comparison won by GT")
    pline("Total questions",             total_questions, "d")
    pline("Strict Correct  (all 3 won)", strict_correct, "d")
    pline("Relaxed Correct (>=1 won)",   relaxed_correct, "d")
    pline("Strict Accuracy",             strict_accuracy)
    pline("Relaxed Accuracy",            relaxed_accuracy)

    print(f"\n  Triplets Won Distribution (per question):")
    for k, v in pairs_won_dist.items():
        print(f"    {int(k)}/3 won: {v} questions")

    print(f"\n[4. SLOT LABEL ANALYSIS]")
    print(sep)
    pline("Label inconsistencies (pred_letter vs slot_label)", label_inconsistencies, "d")
    print(f"\n  YES Slot Distribution per triplet row:")
    for k, v in yes_slot_dist.items():
        print(f"    {int(k)} slot(s) YES: {v} rows")
    pline("Single-YES rows",             single_count, "d")
    pline("Multi-YES rows",              multi_yes_count, "d")
    pline("NONE-YES rows",               none_count, "d")

    print(f"\n[5. CALIBRATION  (by original MCQ option letter)]")
    print(sep)
    print(f"  {'Letter':<8} {'As GT':>7} {'Predicted':>10} {'GT Rate':>9} "
          f"{'Pred Rate':>10} {'Cal Error':>10}")
    for _, r in calibration_df.iterrows():
        print(f"  {r['option_letter']:<8} {r['times_as_GT']:>7d} "
              f"{r['times_predicted']:>10d} {r['GT_rate']:>9.4f} "
              f"{r['pred_rate']:>10.4f} {r['calibration_error']:>10.4f}")

    print(f"\n[6. LETTER DISTRIBUTIONS]")
    print(sep)
    print(f"  GT Distribution:")
    for l, c in gt_letter_dist.items():
        print(f"    {l}: {c}")
    print(f"  Prediction Distribution:")
    for l, c in pred_letter_dist.items():
        print(f"    {l}: {c}")

    if avg_reasoning_len is not None:
        print(f"\n[7. REASONING]")
        print(sep)
        pline("Avg reasoning length (chars)",    avg_reasoning_len)
        pline("Median reasoning length (chars)", median_reasoning_len)

    print(f"\n[SUMMARY]")
    print(SEP)
    pline("Triplet Accuracy",            triplet_accuracy)
    pline("GT Dominance Rate",           gt_dominance_rate)
    pline("Question Consistency",        question_consistency)
    pline("Binary Accuracy",             global_bin["binary_accuracy"])
    pline("Strict MCQ Acc (all 3 won)",  strict_accuracy)
    pline("Relaxed MCQ Acc (>=1 won)",   relaxed_accuracy)
    pline("F1",                          global_bin["f1"])
    pline("MCC",                         global_bin["matthews_corrcoef"])
    pline("Cohen's Kappa",               global_bin["cohens_kappa"])
    pline("Multi-YES rows",              multi_yes_count, "d")
    pline("NONE-YES rows",               none_count, "d")
    pline("Error / unparsed rows",       error_rows, "d")
    print(SEP)

# ============================================================
# SAVE OUTPUT
# ============================================================

mistakes_df = df_valid[~df_valid[IS_CORRECT_COL]].copy()

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_valid.to_excel(writer,  sheet_name="raw_triplets",      index=False)
    if len(df_error):
        df_error.to_excel(writer, sheet_name="error_rows",     index=False)
    if letter_err_rows:
        df_letter_err.to_excel(writer, sheet_name="letter_err_rows", index=False)
    binary_df.to_excel(writer, sheet_name="flattened_binary",  index=False)
    question_level_df.to_excel(writer, sheet_name="question_level", index=False)
    calibration_df.to_excel(writer, sheet_name="calibration",  index=False)
    mistakes_df.to_excel(writer, sheet_name="mistakes",        index=False)

    # Per-letter metrics sheet
    plr = []
    for letter in "ABCD":
        m = per_letter_metrics.get(letter, {})
        if m and "error" not in m:
            plr.append({"option_letter": letter, **m})
    pd.DataFrame(plr).to_excel(writer, sheet_name="per_letter_metrics", index=False)

    # Per-role rates sheet
    prr = []
    for role in ["GT", "D1", "D2"]:
        m = per_role_metrics.get(role, {})
        if m:
            prr.append({
                "role":          role,
                "n":             m["n"],
                "correct_rate":  m["correct_rate"],
                "pred_YES_rate": m["pred_YES_rate"],
                "interpretation": m["label"],
            })
    pd.DataFrame(prr).to_excel(writer, sheet_name="per_role_metrics", index=False)

    # Combined distractor defeat
    combined_defeat.to_excel(writer, sheet_name="distractor_analysis")

    # Summary
    summary_data = {
        "Metric": [
            "Total Rows (raw)", "Error Rows (NaN pred)", "Multi-Answer Rows",
            "Invalid Letter Rows", "Valid Rows (triplets)", "Unique Questions",
            "Total Binary Decisions",
            "-- TRIPLET --",
            "Triplet Accuracy", "GT Dominance Rate", "Question Consistency",
            "-- BINARY --",
            "TP", "TN", "FP", "FN",
            "Binary Accuracy", "Precision", "Recall", "Specificity",
            "F1", "Balanced Accuracy", "MCC", "Cohen's Kappa",
            "NPV", "FPR", "FNR",
            "-- MCQ RECONSTRUCTION --",
            "Strict Accuracy (all 3 won)",
            "Relaxed Accuracy (>=1 won)",
            "Strict Correct", "Relaxed Correct",
            "-- SLOT LABEL --",
            "Label Inconsistencies",
            "Multi-YES Rows", "NONE-YES Rows",
        ],
        "Value": [
            raw_rows, nan_rows, multi_rows,
            letter_err_rows, valid_rows, total_questions,
            total_decisions,
            "",
            round(triplet_accuracy,       4),
            round(gt_dominance_rate,      4),
            round(question_consistency,   4),
            "",
            global_bin["TP"], global_bin["TN"], global_bin["FP"], global_bin["FN"],
            round(global_bin["binary_accuracy"],           4),
            round(global_bin["precision"],                 4),
            round(global_bin["recall"],                    4),
            round(global_bin["specificity"],               4),
            round(global_bin["f1"],                        4),
            round(global_bin["balanced_accuracy"],         4),
            round(global_bin["matthews_corrcoef"],         4),
            round(global_bin["cohens_kappa"],              4),
            round(global_bin["negative_predictive_value"], 4),
            round(global_bin["false_positive_rate"],       4),
            round(global_bin["false_negative_rate"],       4),
            "",
            round(strict_accuracy,  4),
            round(relaxed_accuracy, 4),
            strict_correct, relaxed_correct,
            "",
            label_inconsistencies,
            multi_yes_count, none_count,
        ],
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)

print(f"\n✓ Results saved -> {OUTPUT_FILE}")