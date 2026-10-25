"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.

    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.
    

    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    

    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found in DataFrame")
    if id_col not in df.columns:
        raise KeyError(f"id_col column '{id_col}' not found in DataFrame")
    if time_col not in df.columns:
        raise KeyError(f"time_col column '{time_col}' not found in DataFrame")

    n_rows = len(df)

    # Find missing values
    missing = {}
    for col in df.columns:
        is_missing = df[col].isna() | df[col].astype(str).str.strip().eq("")
        missing[col] = float(is_missing.mean())

    # Find duplicate IDs, ignoring missing IDs
    id_missing = df[id_col].isna() | df[id_col].astype(str).str.strip().eq("")
    ids = df.loc[~id_missing, id_col]
    duplicate_ids = sorted(ids[ids.duplicated(keep=False)].unique().tolist())

    # Count rows that repeat an earlier row
    duplicate_rows = int(df.duplicated().sum())

    # Find constant columns, ignoring missing values
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue

        values = df[col]
        is_missing = values.isna() | values.astype(str).str.strip().eq("")
        values = values[~is_missing]

        if values.nunique() <= 1:
            constant_columns.append(col)

    constant_columns = sorted(constant_columns)

    # Find unlabelled rows and calculate positive rate
    target_missing = df[target].isna() | df[target].astype(str).str.strip().eq("")
    labelled = df.loc[~target_missing, target]

    unlabelled_rows = int(target_missing.sum())
    if len(labelled) == 0:
        positive_rate = 0.0
    else:
        positive_rate = float((labelled == 1).mean())

    # Check timestamps
    time_missing = df[time_col].isna() | df[time_col].astype(str).str.strip().eq("")
    parsed_dates = pd.to_datetime(df[time_col], errors="coerce")
    bad_timestamps = int((~time_missing & parsed_dates.isna()).sum())

    valid_dates = parsed_dates.dropna()
    if valid_dates.empty:
        time_range = [None, None]
    else:
        time_range = [
            valid_dates.min().isoformat(),
            valid_dates.max().isoformat()
        ]

    # Look for possible data leakage
    leakage_suspects = []
    target_missing = df[target].isna() | df[target].astype(str).str.strip().eq("")
    data = df.loc[~target_missing]
    y = data[target]

    if len(data) >= 20 and y.nunique() == 2:
        for col in df.columns:
            if col in [target, id_col, time_col]:
                continue

            values = data[col]
            is_missing = values.isna() | values.astype(str).str.strip().eq("")

            # Skip columns with only one value, counting missing as a value
            filled = values.astype(object).where(~is_missing, "__MISSING__")
            if filled.nunique() <= 1:
                continue

            # Check numeric columns
            if (
                pd.api.types.is_numeric_dtype(values)
                or pd.api.types.is_bool_dtype(values)
            ):
                valid = ~is_missing

                if valid.sum() >= 20 and y[valid].nunique() == 2:
                    auc = roc_auc_score(
                        y[valid].astype(int),
                        values[valid].astype(float)
                    )

                    if auc >= 0.98 or auc <= 0.02:
                        leakage_suspects.append(col)
                        continue

            # Check whether missingness reveals the target
            auc = roc_auc_score(y.astype(int), is_missing.astype(int))

            if auc >= 0.98 or auc <= 0.02:
                leakage_suspects.append(col)

    leakage_suspects = sorted(set(leakage_suspects))

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
