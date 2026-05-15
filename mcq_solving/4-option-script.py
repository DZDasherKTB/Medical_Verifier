"""
united_evaluation.py — Robust United / Full MCQ Solving Evaluator
==================================================================
Experiment formulation:
    (question, all_options) -> selected_answer + YES/NO for each option

Normalization to binary:
    selected_answer=C  becomes  A=NO, B=NO, C=YES, D=NO

This script handles:
  - Multi-YES edge cases from label columns
  - Discrepancies between selected_answer_letter and label columns
  - Per-option, per-letter breakdown
  - Extended metrics: MCC, Cohen's Kappa, Balanced Accuracy, NPV
  - Calibration analysis

Usage
-----
    python united_evaluation.py [--input FILE] [--output FILE] [--quiet]
"""

import argparse
import sys
from collections import Counter, defaultdict

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

parser = argparse.ArgumentParser(description="United full-MCQ evaluation")
parser.add_argument(
    "--input",
    default="./MCQ_Verification_Results/united_option_verification_results.xlsx",
    help="Path to the input Excel file",
)
parser.add_argument(
    "--output",
    default="united_evaluation_results.xlsx",
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

DATASET_ID_COL     = "id_in_dataset"
QUESTION_COL       = "question"
GT_LETTER_COL      = "ground_truth_answer_letter"
GT_TEXT_COL        = "ground_truth_answer"
GT_REASON_COL      = "ground_truth_reasoning"
SEL_LETTER_COL     = "selected_answer_letter"
SEL_TEXT_COL       = "selected_option_text"
REASONING_COL      = "reasoning_trace"

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

# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_label(x):
    v = clean(x).upper()
    return v if v in ("YES", "NO") else None


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
        "TP":  int(tp), "TN": int(tn),
        "FP":  int(fp), "FN": int(fn),
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

print(f"Loading {INPUT_FILE} …")
try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    sys.exit(f"[ERROR] File not found: {INPUT_FILE}")

required_cols = (
    {QUESTION_COL, GT_LETTER_COL, SEL_LETTER_COL}
    | set(OPTION_COLS.values())
    | set(LABEL_COLS.values())
)
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"[ERROR] Missing columns: {missing}")

# ============================================================
# CLEAN
# ============================================================

df[GT_LETTER_COL]  = df[GT_LETTER_COL].apply(clean).str.upper()
df[SEL_LETTER_COL] = df[SEL_LETTER_COL].apply(clean).str.upper()

for letter in "ABCD":
    df[OPTION_COLS[letter]] = df[OPTION_COLS[letter]].apply(clean)
    df[LABEL_COLS[letter]]  = df[LABEL_COLS[letter]].apply(clean_label)

# ============================================================
# FILTER INVALID ROWS
# ============================================================

raw_rows = len(df)

# Missing GT or selection
mask_ok = (
    df[GT_LETTER_COL].isin(["A", "B", "C", "D"]) &
    df[SEL_LETTER_COL].isin(["A", "B", "C", "D"])
)
df_invalid = df[~mask_ok].copy()
df_valid   = df[mask_ok].copy()

if len(df_invalid):
    print(f"[WARN] {len(df_invalid)} rows dropped (missing GT or selection letter).")

valid_rows = len(df_valid)

# ============================================================
# DETERMINE PREDICTION SOURCE
# We derive pred from sel_letter (authoritative) and check
# consistency with label columns.
# ============================================================

def derive_pred_from_selection(row):
    """Derive YES/NO for each letter from the selected answer letter."""
    sel = row[SEL_LETTER_COL]
    return {l: ("YES" if l == sel else "NO") for l in "ABCD"}


def derive_pred_from_labels(row):
    """Read YES/NO from label columns directly."""
    return {l: row[LABEL_COLS[l]] for l in "ABCD"}


# For United, the primary source is SEL_LETTER_COL (holistic selection).
# We also record label-column predictions for consistency checks.

df_valid["pred_from_sel"] = df_valid.apply(
    lambda r: derive_pred_from_selection(r), axis=1
)
df_valid["pred_from_labels"] = df_valid.apply(
    lambda r: derive_pred_from_labels(r), axis=1
)

# Consistency check: does selected letter match the label column?
def check_consistency(row):
    sel = row[SEL_LETTER_COL]
    labels = row["pred_from_labels"]
    selected_label_says_yes = labels.get(sel) == "YES"
    # Count YES labels (should be exactly 1)
    yes_count = sum(1 for v in labels.values() if v == "YES")
    return pd.Series({
        "label_sel_consistent": selected_label_says_yes,
        "label_yes_count": yes_count,
    })

consistency = df_valid.apply(check_consistency, axis=1)
df_valid = pd.concat([df_valid, consistency], axis=1)

inconsistent_count = (~df_valid["label_sel_consistent"]).sum()
multi_label_count  = (df_valid["label_yes_count"] > 1).sum()
none_label_count   = (df_valid["label_yes_count"] == 0).sum()

# ============================================================
# BUILD FLATTENED BINARY ROWS
# Uses sel_letter as ground truth for prediction (primary source)
# ============================================================

binary_rows = []

for _, row in df_valid.iterrows():
    gt_letter  = row[GT_LETTER_COL]
    sel_letter = row[SEL_LETTER_COL]

    for letter in "ABCD":
        is_gt   = (letter == gt_letter)
        is_pred = (letter == sel_letter)

        gt_label   = "YES" if is_gt   else "NO"
        pred_label = "YES" if is_pred else "NO"

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
            "sel_letter":    sel_letter,
            "ground_truth":  gt_label,
            "prediction":    pred_label,
            "binary_result": outcome,
            "gt_binary":     1 if is_gt   else 0,
            "pred_binary":   1 if is_pred else 0,
            # Also capture label-column prediction for comparison
            "label_pred":    row[LABEL_COLS[letter]],
        })

binary_df = pd.DataFrame(binary_rows)

# ============================================================
# 1. GLOBAL MCQ ACCURACY  (from selected_answer_letter)
# ============================================================

df_valid["is_correct"] = df_valid[GT_LETTER_COL] == df_valid[SEL_LETTER_COL]
mcq_accuracy  = df_valid["is_correct"].mean()
total_questions = len(df_valid)
correct_count = df_valid["is_correct"].sum()
wrong_count   = total_questions - correct_count

# ============================================================
# 2. GLOBAL BINARY METRICS
# ============================================================

y_true = binary_df["gt_binary"].tolist()
y_pred = binary_df["pred_binary"].tolist()

global_bin = binary_metrics_from_lists(y_true, y_pred)
total_decisions = len(binary_df)

# ============================================================
# 3. PER-LETTER METRICS
# ============================================================

per_letter_metrics = {}
for letter in "ABCD":
    sub = binary_df[binary_df["letter"] == letter]
    per_letter_metrics[letter] = binary_metrics_from_lists(
        sub["gt_binary"].tolist(), sub["pred_binary"].tolist()
    )

# ============================================================
# 4. STRICT MCQ ACCURACY
# (Exactly one YES in label columns, and it matches GT)
# ============================================================

def strict_check(row):
    labels = row["pred_from_labels"]
    yes_letters = [l for l, v in labels.items() if v == "YES"]
    if len(yes_letters) == 1 and yes_letters[0] == row[GT_LETTER_COL]:
        return True
    return False

df_valid["strict_correct"] = df_valid.apply(strict_check, axis=1)
strict_accuracy = df_valid["strict_correct"].mean()
strict_correct  = df_valid["strict_correct"].sum()

# Relaxed: GT letter appears among predicted YES labels
def relaxed_check(row):
    labels = row["pred_from_labels"]
    return labels.get(row[GT_LETTER_COL]) == "YES"

df_valid["relaxed_correct"] = df_valid.apply(relaxed_check, axis=1)
relaxed_accuracy = df_valid["relaxed_correct"].mean()
relaxed_correct  = df_valid["relaxed_correct"].sum()

# ============================================================
# 5. YES DISTRIBUTION  (from label columns, not sel)
# ============================================================

yes_distribution  = Counter(df_valid["label_yes_count"].tolist())
pred_type_counter = Counter()
for n in df_valid["label_yes_count"]:
    if n == 0:
        pred_type_counter["NONE"] += 1
    elif n == 1:
        pred_type_counter["SINGLE"] += 1
    else:
        pred_type_counter[f"MULTI_{n}"] += 1

# ============================================================
# 6. CALIBRATION
# ============================================================

calibration_rows = []
for letter in "ABCD":
    sub      = binary_df[binary_df["letter"] == letter]
    yes_pred = (sub["pred_binary"] == 1).sum()
    yes_gt   = (sub["gt_binary"]   == 1).sum()
    calibration_rows.append({
        "letter": letter,
        "predicted_YES":      int(yes_pred),
        "ground_truth_YES":   int(yes_gt),
        "pred_rate":          safe_div(yes_pred, len(sub)),
        "gt_rate":            safe_div(yes_gt,   len(sub)),
        "calibration_error":  abs(safe_div(yes_pred, len(sub)) - safe_div(yes_gt, len(sub))),
    })
calibration_df = pd.DataFrame(calibration_rows)

# ============================================================
# 7. GT LETTER DISTRIBUTION
# ============================================================

gt_letter_dist   = df_valid[GT_LETTER_COL].value_counts()
sel_letter_dist  = df_valid[SEL_LETTER_COL].value_counts()

# ============================================================
# 8. MULTI-YES & NONE ANALYSIS
# ============================================================

multi_yes_df   = df_valid[df_valid["label_yes_count"] > 1].copy()
none_df        = df_valid[df_valid["label_yes_count"] == 0].copy()
multi_yes_correct = multi_yes_df["is_correct"].sum() if len(multi_yes_df) else 0
multi_yes_wrong   = len(multi_yes_df) - multi_yes_correct

# ============================================================
# 9. QUESTION-LEVEL RECORDS
# ============================================================

question_level_records = []
for _, row in df_valid.iterrows():
    labels = row["pred_from_labels"]
    yes_letters = [l for l, v in labels.items() if v == "YES"]
    question_level_records.append({
        "question":          row[QUESTION_COL],
        "gt_letter":         row[GT_LETTER_COL],
        "selected_letter":   row[SEL_LETTER_COL],
        "label_yes_letters": ",".join(yes_letters),
        "n_label_yes":       row["label_yes_count"],
        "is_correct":        row["is_correct"],
        "strict_correct":    row["strict_correct"],
        "relaxed_correct":   row["relaxed_correct"],
        "label_consistent":  row["label_sel_consistent"],
    })
question_level_df = pd.DataFrame(question_level_records)

# ============================================================
# REASONING LENGTH ANALYSIS
# ============================================================

avg_reasoning_len    = None
median_reasoning_len = None
if REASONING_COL in df_valid.columns:
    lens = df_valid[REASONING_COL].dropna().apply(lambda x: len(str(x)))
    avg_reasoning_len    = lens.mean()
    median_reasoning_len = lens.median()

# ============================================================
# PRINT REPORT
# ============================================================

SEP  = "=" * 70
sep  = "-" * 70

def pline(label, value, fmt=".4f"):
    if isinstance(value, float):
        print(f"  {label:<44} {value:{fmt}}")
    else:
        print(f"  {label:<44} {value}")

if not QUIET:
    print(f"\n{SEP}")
    print("  UNITED / FULL MCQ EVALUATION REPORT")
    print(SEP)

    print(f"\n[DATASET]")
    print(sep)
    pline("Input file",                 INPUT_FILE, "s")
    pline("Total rows (raw)",           raw_rows, "d")
    pline("Invalid / dropped rows",     len(df_invalid), "d")
    pline("Valid rows",                 valid_rows, "d")
    pline("Total questions",            total_questions, "d")
    pline("Total verifier decisions",   total_decisions, "d")

    print(f"\n[1. MCQ ACCURACY  (from selected_answer_letter)]")
    print(sep)
    pline("MCQ Accuracy",               mcq_accuracy)
    pline("Correct",                    int(correct_count), "d")
    pline("Wrong",                      int(wrong_count), "d")

    print(f"\n[2. LABEL CONSISTENCY CHECK]")
    print(sep)
    pline("Label-Selection Inconsistencies", int(inconsistent_count), "d")
    pline("Multi-YES from labels",      int(multi_label_count), "d")
    pline("NONE from labels",           int(none_label_count), "d")
    pline("Multi-YES correct",          int(multi_yes_correct), "d")
    pline("Multi-YES wrong",            int(multi_yes_wrong), "d")

    print(f"\n[3. FLATTENED BINARY METRICS  (from selected_answer_letter)]")
    print(sep)
    pline("Total decisions",            total_decisions, "d")
    pline("TP",  global_bin["TP"], "d")
    pline("TN",  global_bin["TN"], "d")
    pline("FP",  global_bin["FP"], "d")
    pline("FN",  global_bin["FN"], "d")
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

    print(f"\n  Per-Letter Metrics:")
    for letter in "ABCD":
        m = per_letter_metrics[letter]
        if "error" in m:
            print(f"    {letter}: {m['error']}")
            continue
        print(f"    {letter}: "
              f"TP={m['TP']} FP={m['FP']} FN={m['FN']}  "
              f"Prec={m['precision']:.4f}  "
              f"Rec={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}")

    print(f"\n[4. MCQ RECONSTRUCTION  (from label columns)]")
    print(sep)
    pline("Strict Accuracy  (label-based)", strict_accuracy)
    pline("Relaxed Accuracy (label-based)", relaxed_accuracy)
    pline("Strict Correct",                int(strict_correct), "d")
    pline("Relaxed Correct",               int(relaxed_correct), "d")

    print(f"\n  YES Distribution (label columns):")
    for k in sorted(yes_distribution):
        print(f"    {k} YES: {yes_distribution[k]}")

    print(f"\n  Prediction Type (label columns):")
    for k, v in sorted(pred_type_counter.items()):
        print(f"    {k}: {v}")

    print(f"\n[5. CALIBRATION]")
    print(sep)
    print(f"  {'Letter':<8} {'Pred YES':>10} {'GT YES':>8} {'Pred Rate':>10} {'GT Rate':>8} {'Cal Error':>10}")
    for _, r in calibration_df.iterrows():
        print(f"  {r['letter']:<8} {r['predicted_YES']:>10d} {r['ground_truth_YES']:>8d} "
              f"{r['pred_rate']:>10.4f} {r['gt_rate']:>8.4f} {r['calibration_error']:>10.4f}")

    print(f"\n[6. LETTER DISTRIBUTIONS]")
    print(sep)
    print("  GT Distribution:")
    for l, c in gt_letter_dist.items():
        print(f"    {l}: {c}")
    print("  Selection Distribution:")
    for l, c in sel_letter_dist.items():
        print(f"    {l}: {c}")

    if avg_reasoning_len is not None:
        print(f"\n[7. REASONING]")
        print(sep)
        pline("Avg reasoning length (chars)",    avg_reasoning_len)
        pline("Median reasoning length (chars)", median_reasoning_len)

    print(f"\n[SUMMARY]")
    print(SEP)
    pline("MCQ Accuracy",                mcq_accuracy)
    pline("Binary Accuracy",             global_bin["binary_accuracy"])
    pline("Strict MCQ Acc (label)",      strict_accuracy)
    pline("Relaxed MCQ Acc (label)",     relaxed_accuracy)
    pline("F1",                          global_bin["f1"])
    pline("MCC",                         global_bin["matthews_corrcoef"])
    pline("Cohen's Kappa",               global_bin["cohens_kappa"])
    pline("Label Inconsistencies",       int(inconsistent_count), "d")
    pline("Multi-YES (labels)",          int(multi_label_count), "d")
    print(SEP)

# ============================================================
# SAVE OUTPUT
# ============================================================

mistakes_df = df_valid[~df_valid["is_correct"]].copy()

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_valid.to_excel(writer, sheet_name="raw_valid", index=False)
    if len(df_invalid):
        df_invalid.to_excel(writer, sheet_name="invalid_rows", index=False)
    binary_df.to_excel(writer, sheet_name="flattened_binary", index=False)
    question_level_df.to_excel(writer, sheet_name="question_level", index=False)
    calibration_df.to_excel(writer, sheet_name="calibration", index=False)
    mistakes_df.to_excel(writer, sheet_name="mistakes", index=False)
    if len(multi_yes_df):
        multi_yes_df.to_excel(writer, sheet_name="multi_yes_rows", index=False)
    if len(none_df):
        none_df.to_excel(writer, sheet_name="none_rows", index=False)

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
            "Total Rows (raw)", "Invalid Rows", "Valid Rows",
            "Total Questions", "Total Decisions",
            "── MCQ ──",
            "MCQ Accuracy", "Correct", "Wrong",
            "── BINARY ──",
            "TP", "TN", "FP", "FN",
            "Binary Accuracy", "Precision", "Recall", "Specificity",
            "F1", "Balanced Accuracy", "MCC", "Cohen's Kappa",
            "NPV", "FPR", "FNR",
            "── LABEL-BASED ──",
            "Strict Accuracy", "Relaxed Accuracy",
            "Strict Correct", "Relaxed Correct",
            "Label Inconsistencies", "Multi-YES (labels)", "NONE (labels)",
        ],
        "Value": [
            raw_rows, len(df_invalid), valid_rows,
            total_questions, total_decisions,
            "",
            round(mcq_accuracy, 4), int(correct_count), int(wrong_count),
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
            round(strict_accuracy,  4), round(relaxed_accuracy, 4),
            int(strict_correct), int(relaxed_correct),
            int(inconsistent_count), int(multi_label_count), int(none_label_count),
        ],
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)

print(f"\n✓ Results saved → {OUTPUT_FILE}")