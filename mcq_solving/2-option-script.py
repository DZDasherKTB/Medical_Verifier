"""
duo_evaluation.py — Robust Duo / 2-Option Pairwise Verification Evaluator
==========================================================================
Experiment formulation:
    (question, [GT + 1 distractor]) -> winner (original MCQ letter)

Each MCQ with 4 options generates 3 pairwise rows:
    GT vs distractor_A, GT vs distractor_B, GT vs distractor_D  (whichever 3)

Data layout (your schema):
    A_option  = GT option text      (ALWAYS the GT, not an MCQ letter slot)
    B_option  = Distractor text     (ALWAYS the distractor, not an MCQ letter slot)
    predicted_letter = the original MCQ letter (A/B/C/D) the model chose

Binary normalization per row:
    GT option    -> gt_binary=1, pred_binary = 1 if predicted==gt_letter   else 0
    Distractor   -> gt_binary=0, pred_binary = 1 if predicted==dist_letter else 0

MCQ reconstruction:
    Strict  = all 3 pairs won by GT  (is_correct True for every row of that question)
    Relaxed = at least 1 pair won by GT

Usage
-----
    python duo_evaluation.py [--input FILE] [--output FILE] [--quiet]
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

parser = argparse.ArgumentParser(description="Duo pairwise evaluation")
parser.add_argument(
    "--input",
    default="./MCQ_Verification_Results/duo_option_experiment_verification_results.xlsx",
    help="Path to the input Excel file",
)
parser.add_argument(
    "--output",
    default="duo_evaluation_results.xlsx",
    help="Path to the output Excel file",
)
parser.add_argument("--quiet", action="store_true", help="Suppress console report")
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output
QUIET = args.quiet

# ============================================================
# COLUMN NAMES  (edit here if schema changes)
# ============================================================

DATASET_ID_COL        = "dataset_id"
QUESTION_COL          = "question"
GT_LETTER_COL         = "ground_truth_letter"    # original MCQ letter of the GT
GT_TEXT_COL           = "ground_truth_text"
DISTRACTOR_LETTER_COL = "distractor_letter"       # original MCQ letter of distractor
DISTRACTOR_TEXT_COL   = "distractor_text"
PRED_LETTER_COL       = "predicted_letter"        # original MCQ letter model chose
IS_CORRECT_COL        = "is_correct"              # predicted_letter == gt_letter
REASONING_COL         = "reasoning_trace"

# These columns label slots, NOT original MCQ letters.
# A_option is ALWAYS the GT text; B_option is ALWAYS the distractor text.
GT_OPTION_COL   = "A_option"
DIST_OPTION_COL = "B_option"
GT_LABEL_COL    = "A_label"
DIST_LABEL_COL  = "B_label"

# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_div(num, den):
    return num / den if den else float("nan")


def binary_metrics(y_true, y_pred):
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
    QUESTION_COL, GT_LETTER_COL, DISTRACTOR_LETTER_COL,
    PRED_LETTER_COL, IS_CORRECT_COL,
}
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"[ERROR] Missing columns: {missing}")

# ============================================================
# CLEAN
# ============================================================

raw_rows = len(df)

for col in [GT_LETTER_COL, DISTRACTOR_LETTER_COL, PRED_LETTER_COL]:
    df[col] = df[col].apply(clean).str.upper()

for col in [GT_TEXT_COL, DISTRACTOR_TEXT_COL, GT_OPTION_COL, DIST_OPTION_COL]:
    if col in df.columns:
        df[col] = df[col].apply(clean)

df[IS_CORRECT_COL] = df[IS_CORRECT_COL].apply(
    lambda x: str(x).strip().lower() in ("true", "1", "yes")
)

# ============================================================
# FILTER INVALID ROWS
# ============================================================

valid_letters = {"A", "B", "C", "D"}
mask_ok = (
    df[GT_LETTER_COL].isin(valid_letters) &
    df[DISTRACTOR_LETTER_COL].isin(valid_letters) &
    df[PRED_LETTER_COL].isin(valid_letters)
)
df_invalid = df[~mask_ok].copy()
df_valid   = df[mask_ok].copy()
error_count = len(df_invalid)

if error_count:
    print(f"[WARN] Dropped {error_count} rows with missing/invalid letters.")

# ============================================================
# FLATTEN TO BINARY ROWS
#
# Each pairwise row -> 2 binary decisions:
#   1. GT option row      (gt_binary=1)
#   2. Distractor row     (gt_binary=0)
#
# pred_binary is determined by predicted_letter (original MCQ letter),
# not by slot position (A_label/B_label).
# ============================================================

binary_rows = []

for _, row in df_valid.iterrows():
    gt_letter   = row[GT_LETTER_COL]
    dist_letter = row[DISTRACTOR_LETTER_COL]
    pred_letter = row[PRED_LETTER_COL]

    gt_text   = row.get(GT_TEXT_COL,       "") if GT_TEXT_COL   in df_valid.columns else ""
    dist_text = row.get(DISTRACTOR_TEXT_COL,"") if DISTRACTOR_TEXT_COL in df_valid.columns else ""

    # GT option
    pred_gt = 1 if pred_letter == gt_letter else 0
    binary_rows.append({
        "question":          row[QUESTION_COL],
        "option_letter":     gt_letter,
        "option_text":       gt_text,
        "role":              "GT",
        "paired_against":    dist_letter,
        "pred_letter":       pred_letter,
        "ground_truth":      "YES",
        "prediction":        "YES" if pred_gt else "NO",
        "binary_result":     "TP" if pred_gt else "FN",
        "gt_binary":         1,
        "pred_binary":       pred_gt,
        "pair_correct":      row[IS_CORRECT_COL],
    })

    # Distractor option
    pred_dist = 1 if pred_letter == dist_letter else 0
    binary_rows.append({
        "question":          row[QUESTION_COL],
        "option_letter":     dist_letter,
        "option_text":       dist_text,
        "role":              "DISTRACTOR",
        "paired_against":    gt_letter,
        "pred_letter":       pred_letter,
        "ground_truth":      "NO",
        "prediction":        "YES" if pred_dist else "NO",
        "binary_result":     "FP" if pred_dist else "TN",
        "gt_binary":         0,
        "pred_binary":       pred_dist,
        "pair_correct":      row[IS_CORRECT_COL],
    })

binary_df = pd.DataFrame(binary_rows)

# ============================================================
# 1. PAIRWISE NATIVE METRICS
# ============================================================

pairwise_accuracy = df_valid[IS_CORRECT_COL].mean()
total_pairs       = len(df_valid)
total_questions   = df_valid[QUESTION_COL].nunique()

q_all_correct = df_valid.groupby(QUESTION_COL)[IS_CORRECT_COL].all()
q_any_correct = df_valid.groupby(QUESTION_COL)[IS_CORRECT_COL].any()
question_consistency = q_all_correct.mean()

# How many pairs won per question
pairs_won_per_q = df_valid.groupby(QUESTION_COL)[IS_CORRECT_COL].sum()
pairs_won_dist  = pairs_won_per_q.value_counts().sort_index()

# Distractor defeat rate by original MCQ letter
distractor_stats = (
    df_valid.groupby(DISTRACTOR_LETTER_COL)[IS_CORRECT_COL]
    .agg(correct="sum", total="count")
    .assign(defeat_rate=lambda x: x["correct"] / x["total"])
)

distractor_errors = (
    df_valid[~df_valid[IS_CORRECT_COL]]
    .groupby(DISTRACTOR_LETTER_COL)
    .size()
    .rename("errors")
)

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
    if len(sub) == 0:
        continue
    per_letter_metrics[letter] = binary_metrics(
        sub["gt_binary"].tolist(), sub["pred_binary"].tolist()
    )

# ============================================================
# 3. MCQ RECONSTRUCTION
# ============================================================

strict_correct  = int(q_all_correct.sum())
relaxed_correct = int(q_any_correct.sum())
strict_accuracy  = safe_div(strict_correct,  total_questions)
relaxed_accuracy = safe_div(relaxed_correct, total_questions)

question_level_records = []
for q in df_valid[QUESTION_COL].unique():
    sub = df_valid[df_valid[QUESTION_COL] == q]
    n_pairs      = len(sub)
    n_won        = int(sub[IS_CORRECT_COL].sum())
    gt_letter    = sub[GT_LETTER_COL].iloc[0]
    distractors  = sorted(sub[DISTRACTOR_LETTER_COL].unique())
    question_level_records.append({
        "question":       q,
        "gt_letter":      gt_letter,
        "distractors":    ",".join(distractors),
        "total_pairs":    n_pairs,
        "pairs_won":      n_won,
        "all_correct":    n_won == n_pairs,
        "any_correct":    n_won > 0,
        "win_rate":       safe_div(n_won, n_pairs),
    })

question_level_df = pd.DataFrame(question_level_records)

# ============================================================
# 4. CALIBRATION
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
        "GT_rate":          safe_div(yes_gt,  n),
        "pred_rate":        safe_div(yes_pred, n),
        "calibration_error": abs(safe_div(yes_pred, n) - safe_div(yes_gt, n)),
    })
calibration_df = pd.DataFrame(calibration_rows)

# ============================================================
# 5. REASONING LENGTH
# ============================================================

avg_reasoning_len    = None
median_reasoning_len = None
if REASONING_COL in df_valid.columns:
    lens = df_valid[REASONING_COL].dropna().apply(lambda x: len(str(x)))
    if len(lens):
        avg_reasoning_len    = lens.mean()
        median_reasoning_len = lens.median()

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
    print("  DUO / PAIRWISE EVALUATION REPORT")
    print(SEP)

    print(f"\n[DATASET]")
    print(sep)
    pline("Input file",                 INPUT_FILE, "s")
    pline("Total rows (raw)",           raw_rows, "d")
    pline("Rows with errors/missing",   error_count, "d")
    pline("Valid rows (pairs)",         len(df_valid), "d")
    pline("Unique questions",           total_questions, "d")

    print(f"\n[1. PAIRWISE NATIVE METRICS]")
    print(sep)
    pline("Pairwise Accuracy",          pairwise_accuracy)
    pline("Question Consistency (all pairs won)", question_consistency)

    print(f"\n  Distractor Defeat Rate by Option Letter:")
    for letter, row_s in distractor_stats.iterrows():
        print(f"    {letter}  correct={int(row_s['correct'])}/{int(row_s['total'])}  "
              f"defeat_rate={row_s['defeat_rate']:.4f}")

    print(f"\n  Pairs Won per Question:")
    for k, v in pairs_won_dist.items():
        tag = "all correct" if k == 3 else ("all wrong" if k == 0 else "partial")
        print(f"    {int(k)} pairs won: {v} questions  [{tag}]")

    print(f"\n  Distractor Errors (losses when this letter was distractor):")
    for letter, cnt in distractor_errors.items():
        print(f"    {letter}: {int(cnt)}")

    print(f"\n[2. FLATTENED BINARY METRICS]")
    print(sep)
    print(f"  Each pair -> 2 binary decisions (GT option + distractor option)")
    pline("Total binary decisions",     total_decisions, "d")
    pline("TP", global_bin["TP"], "d")
    pline("TN", global_bin["TN"], "d")
    pline("FP", global_bin["FP"], "d")
    pline("FN", global_bin["FN"], "d")
    print()
    pline("Binary Accuracy",            global_bin["binary_accuracy"])
    pline("Precision",                  global_bin["precision"])
    pline("Recall (Sensitivity)",       global_bin["recall"])
    pline("Specificity",                global_bin["specificity"])
    pline("F1 Score",                   global_bin["f1"])
    pline("Balanced Accuracy",          global_bin["balanced_accuracy"])
    pline("MCC",                        global_bin["matthews_corrcoef"])
    pline("Cohen's Kappa",              global_bin["cohens_kappa"])
    pline("NPV",                        global_bin["negative_predictive_value"])
    pline("False Positive Rate",        global_bin["false_positive_rate"])
    pline("False Negative Rate",        global_bin["false_negative_rate"])

    print(f"\n  Per-Option-Letter Binary Metrics:")
    for letter in "ABCD":
        m = per_letter_metrics.get(letter, {})
        if not m or "error" in m:
            continue
        print(f"    {letter}:  "
              f"Acc={m['binary_accuracy']:.4f}  "
              f"Prec={m['precision']:.4f}  "
              f"Rec={m['recall']:.4f}  "
              f"Spec={m['specificity']:.4f}  "
              f"F1={m['f1']:.4f}")

    print(f"\n[3. MCQ RECONSTRUCTION METRICS]")
    print(sep)
    print(f"  Strict  = all pairwise comparisons won by GT")
    print(f"  Relaxed = at least 1 comparison won by GT")
    pline("Total questions",            total_questions, "d")
    pline("Strict Correct  (all won)",  strict_correct, "d")
    pline("Relaxed Correct (>=1 won)", relaxed_correct, "d")
    pline("Strict Accuracy",            strict_accuracy)
    pline("Relaxed Accuracy",           relaxed_accuracy)

    print(f"\n[4. CALIBRATION  (by original MCQ option letter)]")
    print(sep)
    print(f"  {'Letter':<8} {'As GT':>7} {'Predicted':>10} {'GT Rate':>9} "
          f"{'Pred Rate':>10} {'Cal Error':>10}")
    for _, r in calibration_df.iterrows():
        print(f"  {r['option_letter']:<8} {r['times_as_GT']:>7d} "
              f"{r['times_predicted']:>10d} {r['GT_rate']:>9.4f} "
              f"{r['pred_rate']:>10.4f} {r['calibration_error']:>10.4f}")

    if avg_reasoning_len is not None:
        print(f"\n[5. REASONING]")
        print(sep)
        pline("Avg reasoning length (chars)",    avg_reasoning_len)
        pline("Median reasoning length (chars)", median_reasoning_len)

    print(f"\n[SUMMARY]")
    print(SEP)
    pline("Pairwise Accuracy",           pairwise_accuracy)
    pline("Binary Accuracy",             global_bin["binary_accuracy"])
    pline("Strict MCQ Acc (all won)",    strict_accuracy)
    pline("Relaxed MCQ Acc (>=1 won)",   relaxed_accuracy)
    pline("F1",                          global_bin["f1"])
    pline("MCC",                         global_bin["matthews_corrcoef"])
    pline("Cohen's Kappa",               global_bin["cohens_kappa"])
    print(SEP)

# ============================================================
# SAVE OUTPUT
# ============================================================

mistakes_df = df_valid[~df_valid[IS_CORRECT_COL]].copy()

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_valid.to_excel(writer, sheet_name="raw_pairwise", index=False)
    if len(df_invalid):
        df_invalid.to_excel(writer, sheet_name="invalid_rows", index=False)
    binary_df.to_excel(writer, sheet_name="flattened_binary", index=False)
    question_level_df.to_excel(writer, sheet_name="question_level", index=False)
    calibration_df.to_excel(writer, sheet_name="calibration", index=False)
    distractor_stats.to_excel(writer, sheet_name="distractor_analysis")
    mistakes_df.to_excel(writer, sheet_name="mistakes", index=False)

    per_letter_records = []
    for letter in "ABCD":
        m = per_letter_metrics.get(letter, {})
        if m and "error" not in m:
            per_letter_records.append({"option_letter": letter, **m})
    pd.DataFrame(per_letter_records).to_excel(
        writer, sheet_name="per_letter_metrics", index=False
    )

    summary_data = {
        "Metric": [
            "Total Rows (raw)", "Error/Missing Rows", "Valid Rows (pairs)",
            "Unique Questions",
            "-- PAIRWISE --",
            "Pairwise Accuracy", "Question Consistency",
            "-- BINARY --",
            "Total Binary Decisions",
            "TP", "TN", "FP", "FN",
            "Binary Accuracy", "Precision", "Recall", "Specificity",
            "F1", "Balanced Accuracy", "MCC", "Cohen's Kappa",
            "NPV", "FPR", "FNR",
            "-- MCQ RECONSTRUCTION --",
            "Strict Accuracy (all pairs won)",
            "Relaxed Accuracy (>=1 won)",
            "Strict Correct", "Relaxed Correct",
        ],
        "Value": [
            raw_rows, error_count, len(df_valid), total_questions,
            "",
            round(pairwise_accuracy,    4),
            round(question_consistency, 4),
            "",
            total_decisions,
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
            strict_correct,
            relaxed_correct,
        ],
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)

print(f"\n Results saved -> {OUTPUT_FILE}")