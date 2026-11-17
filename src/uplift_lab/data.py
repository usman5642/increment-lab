"""Download a pinned public benchmark and audit without inventing repairs."""

from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split

URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
EXPECTED_SHA256 = "bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f"
EXPECTED_MD5 = "a68a81291f53a14f4e29002629803ba3"
FEATURES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]
ARMS = ["Mens E-Mail", "No E-Mail", "Womens E-Mail"]


def download(root):
    path = Path(root) / "data/raw/hillstrom.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
            raise ValueError(
                "Dataset checksum differs from the scikit-uplift reference"
            )
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Cached dataset checksum failed")
    return path, {
        "download_url": URL,
        "md5": EXPECTED_MD5,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def validate_frame(df):
    required = set(
        FEATURES + ["history_segment", "segment", "visit", "conversion", "spend"]
    )
    if set(df.columns) != required:
        raise ValueError("Unexpected data schema")
    if len(df) != 64000 or df.isna().any().any():
        raise ValueError("Unexpected row count or missing values")
    if set(df.segment.unique()) != set(ARMS):
        raise ValueError("Unexpected experiment arms")
    for col in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Non-binary {col}")
    if (
        not df.recency.between(1, 12).all()
        or (df[["history", "spend"]] < 0).any().any()
    ):
        raise ValueError("Invalid numeric range")
    if not np.isfinite(df.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite numeric value")
    if not df.zip_code.isin(["Urban", "Surburban", "Rural"]).all():
        raise ValueError("Unexpected geography category")
    if not df.channel.isin(["Web", "Phone", "Multichannel"]).all():
        raise ValueError("Unexpected channel category")


def load(root, seed):
    path = Path(root) / "data/raw/hillstrom.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Run the explicit download command first")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError("SHA256 mismatch")
    df = pd.read_csv(path)
    validate_frame(df)
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    counts = df.segment.value_counts().reindex(ARMS)
    audit = {
        "rows": len(df),
        "normalisation": {
            "zip_code": "Surburban -> Suburban; source spelling corrected without changing grouping"
        },
        "sha256": EXPECTED_SHA256,
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rule": "Retain: matching rows are not known duplicate customers without identifiers.",
        "arm_counts": {k: int(v) for k, v in counts.items()},
        "srm_p_value": float(chisquare(counts.to_numpy()).pvalue),
    }
    df["source_row"] = np.arange(len(df))
    retained = df[df.segment.isin(ARMS[:2])].copy()
    retained["treatment"] = (retained.segment == ARMS[0]).astype(int)
    train, test = train_test_split(
        retained, test_size=0.5, random_state=seed, stratify=retained.treatment
    )
    if set(train.source_row) & set(test.source_row):
        raise ValueError("Split overlap")
    audit.update(
        retained_rows=len(retained),
        train_rows=len(train),
        test_rows=len(test),
        split_disjoint=True,
        two_arm_srm_p_value=float(chisquare(retained.treatment.value_counts()).pvalue),
    )
    x = pd.get_dummies(train[FEATURES], columns=CATEGORICAL, dtype=float)
    balance = []
    for col in x:
        a = x.loc[train.treatment == 1, col]
        b = x.loc[train.treatment == 0, col]
        scale = np.sqrt((a.var() + b.var()) / 2)
        balance.append(
            {
                "feature": col,
                "smd": float((a.mean() - b.mean()) / scale) if scale else 0.0,
            }
        )
    audit["training_balance"] = balance
    return {"train": train, "test": test}, audit
