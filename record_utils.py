import json
import os
import pandas as pd


RECORD_FILE = "records.json"


def load_records(file_path=RECORD_FILE):
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []

    if isinstance(data, list):
        return data

    return []


def records_to_dataframe(records):
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    numeric_cols = [
        "accuracy_score",
        "stability_score",
        "avg_abs_cent_error",
        "max_abs_cent_error",
        "best_time_shift",
        "estimated_key_shift",
        "cent_stability_std",
        "highest_note",
        "lowest_note",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def get_profile_summary(df):
    if df.empty:
        return None

    summary = {
        "avg_accuracy": df["accuracy_score"].mean() if "accuracy_score" in df else None,
        "avg_stability": df["stability_score"].mean() if "stability_score" in df else None,
        "avg_key_shift": df["estimated_key_shift"].mean() if "estimated_key_shift" in df else None,
    }

    if "accuracy_score" in df.columns:
        summary["best_accuracy_song"] = df.loc[df["accuracy_score"].idxmax()].to_dict()
        summary["hardest_song"] = df.loc[df["accuracy_score"].idxmin()].to_dict()

    if "stability_score" in df.columns:
        summary["best_stability_song"] = df.loc[df["stability_score"].idxmax()].to_dict()

    return summary