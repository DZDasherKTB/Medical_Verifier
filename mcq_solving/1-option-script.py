"""
solo_evaluation.py — Robust Solo / Independent Option Verification Evaluator
=============================================================================
Experiment formulation:
    (question, single_option) -> YES / NO   (4 separate calls per MCQ)

This script handles:
  - Missing/error rows (NaN labels from API failures)
  - Multi-YES, NONE, and single-YES prediction types
  - Flattened binary metrics (TP/TN/FP/FN) using the letter-based GT
  - Per-option (A/B/C/D) breakdown
  - MCQ-level strict / relaxed / none / multi analysis
  - Extended metrics: MCC, Cohen's Kappa, NPV, Balanced Accuracy

Usage
-----
    python solo_evaluation.py [--input FILE] [--output FILE] [--quiet]
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

parser = argparse.ArgumentParser(description="Solo independent-option evaluation")
parser.add_argument(
    "--input",
    default="./MCQ_Verification_Results/indipendent_option_verification_results.xlsx",
    help="Path to the input Excel file",
)
parser.add_argument(
    "--output",
    default="solo_evaluation_results.xlsx",
    help="Path to the output Excel file",
)
parser.add_argument(
    "--quiet", action="store_true", help="Suppress console report"
)
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output
QUIET = args.quiet

# ============================================================
# COLUMN NAMES  (edit if schema changes)
# ============================================================

DATASET_ID_COL = "id_in_dataset"
QUESTION_COL   = "question"
GT_LETTER_COL  = "ground_truth_answer_letter"
GT_REASON_COL  = "ground_truth_reasoning"
ERROR_COL      = "error"

OPTION_COLS = {
    "A": "option_A",
    "B": "option_B",
    "C": "option_C",
    "D": "option_D",
}
LABEL_COLS = {
    "A": "A_label",
    "B": "B_label",
    "C": "C_label",
    "D": "D_label",
}
REASON_COLS = {
    "A": "A_reason",
    "B": "B_reason",
    "C": "C_reason",
    "D": "D_reason",
}
TRACE_COLS = {
    "A": "A_reasoning_trace",
    "B": "B_reasoning_trace",
    "C": "C_reasoning_trace",
    "D": "D_reasoning_trace",
}

# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_label(x):
    v = clean(x).upper()
    return v if v in ("YES", "NO") else None   # None = abstention / error


def safe_div(num, den):
    return num / den if den else float("nan")


def binary_metrics_from_lists(y_true, y_pred):
    """Full binary classification metric dict."""
    if len(set(y_true)) < 2:
        return {"error": "single class in y_true — metrics undefined"}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tp + tn + fp + fn
    po = (tp + tn) / total if total else float("nan")
    pe = (
        ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total ** 2)
        if total else float("nan")
    )
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "TP":  int(tp),  "TN": int(tn),
        "FP":  int(fp),  "FN": int(fn),
        "binary_accuracy":         accuracy_score(y_true, y_pred),
        "precision":               precision_score(y_true, y_pred, zero_division=0),
        "recall":                  recall_score(y_true, y_pred, zero_division=0),
        "f1":                      f1_score(y_true, y_pred, zero_division=0),
        "specificity":             safe_div(tn, tn + fp),
        "false_positive_rate":     safe_div(fp, fp + tn),
        "false_negative_rate":     safe_div(fn, fn + tp),
        "negative_predictive_value": safe_div(tn, tn + fn),
        "balanced_accuracy":       0.5 * (safe_div(tp, tp + fn) + safe_div(tn, tn + fp)),
        "matthews_corrcoef":       safe_div(tp * tn - fp * fn, mcc_denom),
        "cohens_kappa":            safe_div(po - pe, 1 - pe),
    }

# ============================================================
# LOAD & VALIDATE
# ============================================================

print(f"Loading {INPUT_FILE} …")
try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    sys.exit(f"[ERROR] File not found: {INPUT_FILE}")

required_cols = {QUESTION_COL, GT_LETTER_COL} | set(OPTION_COLS.values()) | set(LABEL_COLS.values())
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"[ERROR] Missing columns: {missing}")

# ============================================================
# CLEAN & CATEGORISE ROWS
# ============================================================

df[GT_LETTER_COL] = df[GT_LETTER_COL].apply(clean).str.upper()

for letter in "ABCD":
    df[OPTION_COLS[letter]] = df[OPTION_COLS[letter]].apply(clean)
    df[LABEL_COLS[letter]]  = df[LABEL_COLS[letter]].apply(clean_label)

has_error = (ERROR_COL in df.columns) and df[ERROR_COL].notna().any()
if has_error:
    error_df  = df[df[ERROR_COL].notna()].copy()
    df_valid  = df[df[ERROR_COL].isna()].copy()
    print(f"[WARN] {len(error_df)} rows had API errors and are excluded from metrics.")
else:
    error_df = pd.DataFrame()
    df_valid = df.copy()

# Rows where any label is None (unexpected abstention)
mask_incomplete = df_valid[[LABEL_COLS[l] for l in "ABCD"]].isna().any(axis=1)
incomplete_df = df_valid[mask_incomplete].copy()
df_valid = df_valid[~mask_incomplete].copy()

if len(incomplete_df):
    print(f"[WARN] {len(incomplete_df)} rows had missing/null labels (partial abstention) — excluded.")

raw_rows       = len(df)
valid_rows     = len(df_valid)
error_rows     = len(error_df)
incomplete_rows = len(incomplete_df)

# ============================================================
# BUILD FLATTENED BINARY ROWS
# ============================================================

binary_rows = []

for _, row in df_valid.iterrows():
    gt_letter = row[GT_LETTER_COL]

    for letter in "ABCD":
        label  = row[LABEL_COLS[letter]]   # "YES" or "NO"
        is_gt  = (letter == gt_letter)

        gt_label   = "YES" if is_gt else "NO"
        pred_label = label  # model's explicit decision

        if gt_label == "YES" and pred_label == "YES":
            outcome = "TP"
        elif gt_label == "NO" and pred_label == "NO":
            outcome = "TN"
        elif gt_label == "NO" and pred_label == "YES":
            outcome = "FP"
        else:
            outcome = "FN"

        binary_rows.append({
            "question":      row[QUESTION_COL],
            "letter":        letter,
            "option_text":   row[OPTION_COLS[letter]],
            "gt_letter":     gt_letter,
            "ground_truth":  gt_label,
            "prediction":    pred_label,
            "binary_result": outcome,
            "gt_binary":     1 if is_gt       else 0,
            "pred_binary":   1 if pred_label == "YES" else 0,
        })

binary_df = pd.DataFrame(binary_rows)

# ============================================================
# 1. GLOBAL BINARY METRICS
# ============================================================

y_true = binary_df["gt_binary"].tolist()
y_pred = binary_df["pred_binary"].tolist()

global_bin = binary_metrics_from_lists(y_true, y_pred)
total_decisions = len(binary_df)

# ============================================================
# 2. PER-LETTER METRICS
# ============================================================

per_letter_metrics = {}
for letter in "ABCD":
    sub = binary_df[binary_df["letter"] == letter]
    per_letter_metrics[letter] = binary_metrics_from_lists(
        sub["gt_binary"].tolist(), sub["pred_binary"].tolist()
    )

# ============================================================
# 3. MCQ RECONSTRUCTION METRICS
# ============================================================

q_to_rows = defaultdict(list)
for _, r in binary_df.iterrows():
    q_to_rows[r["question"]].append(r)

strict_correct  = 0
relaxed_correct = 0
none_count      = 0
multi_yes_count = 0
contradiction_count = 0
fully_correct_questions = 0    # all 4 options correct
yes_distribution = Counter()
pred_type_counter = Counter()
question_level_records = []

for q, rows in q_to_rows.items():
    gt_letter   = rows[0]["gt_letter"]
    gt_opts     = [r["letter"] for r in rows if r["ground_truth"] == "YES"]
    pred_opts   = [r["letter"] for r in rows if r["prediction"]   == "YES"]
    all_correct = all(r["binary_result"] in ("TP", "TN") for r in rows)

    n_yes = len(pred_opts)
    yes_distribution[n_yes] += 1

    if n_yes == 0:
        pred_type_counter["NONE"] += 1
        none_count += 1
    elif n_yes == 1:
        pred_type_counter["SINGLE"] += 1
    else:
        pred_type_counter[f"MULTI_{n_yes}"] += 1
        multi_yes_count += 1

    # A contradiction here would be multi-YES that includes a TN being flipped
    # (that IS multi-YES — track separately)
    if n_yes > 1:
        contradiction_count += 1

    if all_correct:
        fully_correct_questions += 1

    gt_letter_in_pred = gt_letter in pred_opts
    strict_ok  = (n_yes == 1 and pred_opts[0] == gt_letter)
    relaxed_ok = gt_letter_in_pred

    if strict_ok:
        strict_correct += 1
    if relaxed_ok:
        relaxed_correct += 1

    question_level_records.append({
        "question":        q,
        "gt_letter":       gt_letter,
        "pred_letters":    ",".join(pred_opts),
        "n_yes":           n_yes,
        "strict_correct":  strict_ok,
        "relaxed_correct": relaxed_ok,
        "all_option_correct": all_correct,
    })

question_level_df = pd.DataFrame(question_level_records)
total_questions   = len(q_to_rows)

strict_accuracy   = safe_div(strict_correct,  total_questions)
relaxed_accuracy  = safe_div(relaxed_correct, total_questions)
full_q_accuracy   = safe_div(fully_correct_questions, total_questions)

# ============================================================
# 4. LETTER YES COUNT (how often each letter predicted YES)
# ============================================================

option_yes_dist = (
    binary_df[binary_df["prediction"] == "YES"]["letter"].value_counts()
)

gt_letter_dist = df_valid[GT_LETTER_COL].value_counts()

# ============================================================
# 5. CALIBRATION: per-letter YES rate vs GT rate
# ============================================================

calibration_rows = []
for letter in "ABCD":
    sub      = binary_df[binary_df["letter"] == letter]
    yes_pred = (sub["pred_binary"] == 1).sum()
    yes_gt   = (sub["gt_binary"]   == 1).sum()
    calibration_rows.append({
        "letter": letter,
        "predicted_YES": int(yes_pred),
        "ground_truth_YES": int(yes_gt),
        "pred_rate": safe_div(yes_pred, len(sub)),
        "gt_rate":   safe_div(yes_gt,   len(sub)),
        "calibration_error": abs(safe_div(yes_pred, len(sub)) - safe_div(yes_gt, len(sub))),
    })
calibration_df = pd.DataFrame(calibration_rows)

# ============================================================
# 6. ERROR ROW ANALYSIS
# ============================================================

error_summary = {}
if has_error and len(error_df):
    error_summary = error_df[ERROR_COL].value_counts().to_dict()

# ============================================================
# PRINT REPORT
# ============================================================

SEP  = "=" * 70
sep  = "-" * 70

def pline(label, value, fmt=".4f"):
    if isinstance(value, float):
        print(f"  {label:<42} {value:{fmt}}")
    else:
        print(f"  {label:<42} {value}")

if not QUIET:
    print(f"\n{SEP}")
    print("  SOLO / INDEPENDENT OPTION EVALUATION REPORT")
    print(SEP)

    print(f"\n[DATASET]")
    print(sep)
    pline("Input file",              INPUT_FILE, "s")
    pline("Total rows (raw)",        raw_rows, "d")
    pline("Error rows (API failure)", error_rows, "d")
    pline("Incomplete label rows",   incomplete_rows, "d")
    pline("Valid rows",              valid_rows, "d")
    pline("Total questions",         total_questions, "d")
    pline("Total verifier decisions", total_decisions, "d")
    if error_summary:
        print("  Error breakdown:")
        for err, cnt in error_summary.items():
            print(f"    {err}: {cnt}")

    print(f"\n[1. FLATTENED BINARY METRICS]")
    print(sep)
    pline("TP",  global_bin["TP"], "d")
    pline("TN",  global_bin["TN"], "d")
    pline("FP",  global_bin["FP"], "d")
    pline("FN",  global_bin["FN"], "d")
    print()
    pline("Binary Accuracy",         global_bin["binary_accuracy"])
    pline("Precision",               global_bin["precision"])
    pline("Recall (Sensitivity)",    global_bin["recall"])
    pline("Specificity",             global_bin["specificity"])
    pline("F1 Score",                global_bin["f1"])
    pline("Balanced Accuracy",       global_bin["balanced_accuracy"])
    pline("MCC",                     global_bin["matthews_corrcoef"])
    pline("Cohen's Kappa",           global_bin["cohens_kappa"])
    pline("NPV",                     global_bin["negative_predictive_value"])
    pline("False Positive Rate",     global_bin["false_positive_rate"])
    pline("False Negative Rate",     global_bin["false_negative_rate"])

    print(f"\n  Per-Letter Metrics (A/B/C/D):")
    for letter in "ABCD":
        m = per_letter_metrics[letter]
        if "error" in m:
            print(f"    {letter}: {m['error']}")
            continue
        print(f"    {letter}: "
              f"Acc={m['binary_accuracy']:.4f}  "
              f"Prec={m['precision']:.4f}  "
              f"Rec={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"Spec={m['specificity']:.4f}")

    print(f"\n[2. MCQ RECONSTRUCTION METRICS]")
    print(sep)
    pline("Strict Accuracy",         strict_accuracy)
    pline("Relaxed Accuracy",        relaxed_accuracy)
    pline("Full Q Accuracy (all 4)", full_q_accuracy)
    pline("Strict Correct",          strict_correct, "d")
    pline("Relaxed Correct",         relaxed_correct, "d")
    pline("None Predictions",        none_count, "d")
    pline("Multi-YES",               multi_yes_count, "d")

    print(f"\n  YES Distribution (n predicted YES per question):")
    for k in sorted(yes_distribution):
        print(f"    {k} YES: {yes_distribution[k]}")

    print(f"\n  Prediction Type:")
    for k, v in sorted(pred_type_counter.items()):
        print(f"    {k}: {v}")

    print(f"\n[3. CALIBRATION (predicted YES rate vs GT YES rate)]")
    print(sep)
    print(f"  {'Letter':<8} {'Pred YES':>10} {'GT YES':>8} {'Pred Rate':>10} {'GT Rate':>8} {'Cal Error':>10}")
    for _, r in calibration_df.iterrows():
        print(f"  {r['letter']:<8} {r['predicted_YES']:>10d} {r['ground_truth_YES']:>8d} "
              f"{r['pred_rate']:>10.4f} {r['gt_rate']:>8.4f} {r['calibration_error']:>10.4f}")

    print(f"\n[4. GROUND TRUTH DISTRIBUTION]")
    print(sep)
    for letter, cnt in gt_letter_dist.items():
        print(f"    {letter}: {cnt}")

    print(f"\n[SUMMARY]")
    print(SEP)
    pline("Binary Accuracy",     global_bin["binary_accuracy"])
    pline("Strict MCQ Acc",      strict_accuracy)
    pline("Relaxed MCQ Acc",     relaxed_accuracy)
    pline("Full-Q Accuracy",     full_q_accuracy)
    pline("F1",                  global_bin["f1"])
    pline("MCC",                 global_bin["matthews_corrcoef"])
    pline("Cohen's Kappa",       global_bin["cohens_kappa"])
    pline("None Predictions",    none_count, "d")
    pline("Multi-YES",           multi_yes_count, "d")
    print(SEP)

# ============================================================
# SAVE OUTPUT
# ============================================================

mistakes_df = question_level_df[~question_level_df["strict_correct"]].copy()

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_valid.to_excel(writer, sheet_name="raw_valid", index=False)
    if len(error_df):
        error_df.to_excel(writer, sheet_name="error_rows", index=False)
    if len(incomplete_df):
        incomplete_df.to_excel(writer, sheet_name="incomplete_rows", index=False)
    binary_df.to_excel(writer, sheet_name="flattened_binary", index=False)
    question_level_df.to_excel(writer, sheet_name="question_level", index=False)
    calibration_df.to_excel(writer, sheet_name="calibration", index=False)
    mistakes_df.to_excel(writer, sheet_name="mistakes", index=False)

    # Per-letter metrics sheet
    per_letter_records = []
    for letter in "ABCD":
        m = per_letter_metrics[letter]
        if "error" not in m:
            rec = {"letter": letter}
            rec.update(m)
            per_letter_records.append(rec)
    pd.DataFrame(per_letter_records).to_excel(
        writer, sheet_name="per_letter_metrics", index=False
    )

    # Summary sheet
    summary_data = {
        "Metric": [
            "Total Rows (raw)", "Error Rows", "Incomplete Rows", "Valid Rows",
            "Total Questions", "Total Decisions",
            "── BINARY ──",
            "TP", "TN", "FP", "FN",
            "Binary Accuracy", "Precision", "Recall", "Specificity",
            "F1", "Balanced Accuracy", "MCC", "Cohen's Kappa", "NPV",
            "FPR", "FNR",
            "── MCQ RECONSTRUCTION ──",
            "Strict Accuracy", "Relaxed Accuracy", "Full-Q Accuracy",
            "Strict Correct", "Relaxed Correct",
            "None Predictions", "Multi-YES",
        ],
        "Value": [
            raw_rows, error_rows, incomplete_rows, valid_rows,
            total_questions, total_decisions,
            "",
            global_bin["TP"], global_bin["TN"], global_bin["FP"], global_bin["FN"],
            round(global_bin["binary_accuracy"],         4),
            round(global_bin["precision"],               4),
            round(global_bin["recall"],                  4),
            round(global_bin["specificity"],             4),
            round(global_bin["f1"],                      4),
            round(global_bin["balanced_accuracy"],       4),
            round(global_bin["matthews_corrcoef"],       4),
            round(global_bin["cohens_kappa"],            4),
            round(global_bin["negative_predictive_value"], 4),
            round(global_bin["false_positive_rate"],     4),
            round(global_bin["false_negative_rate"],     4),
            "",
            round(strict_accuracy,  4), round(relaxed_accuracy, 4),
            round(full_q_accuracy,  4),
            strict_correct, relaxed_correct,
            none_count, multi_yes_count,
        ],
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)

print(f"\n✓ Results saved → {OUTPUT_FILE}")