"""Session-level place-cell property comparisons for the GUI."""
from __future__ import annotations

import numpy as np


DRUG_TREATMENTS = ("SCH", "Prazosin", "Propranolol")
METRIC_SPECS = (
    ("place_cell_percent", "Place-cell fraction", "Place cells (%)"),
    ("spatial_information", "Spatial information", "Bits/event"),
    ("field_locations", "Mean field location", "Position (cm)"),
    ("reward_fraction", "Reward-zone fields", "Fields (%)"),
    ("onset_fraction", "Run-onset fields", "Fields (%)"),
    ("field_width", "Field width", "Width (cm)"),
    ("odd_even_stability", "Odd-even stability", "Pearson r"),
    ("consecutive_stability", "Consecutive stability", "Pearson r"),
    ("in_out_ratio", "In/out activity ratio", "Ratio"),
)


def available_trial_types(metric_rows: list[dict]) -> list[str]:
    return sorted({str(row["trial_type"]) for row in metric_rows})


def _session_value(row: dict, metric_key: str) -> float:
    if metric_key == "place_cell_percent":
        return float(row[metric_key])
    if metric_key in {"reward_fraction", "onset_fraction"}:
        return 100.0 * float(row[metric_key])
    values = np.asarray(row[metric_key], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def _trial_rows(metric_rows: list[dict], trial_type: str) -> list[dict]:
    return [row for row in metric_rows if row["trial_type"] == trial_type]


def _drug_allowed(row: dict, drug_family: str) -> bool:
    return (row["treatment"] in DRUG_TREATMENTS
            and (drug_family == "Any drug" or row["treatment"] == drug_family))


def _saline_allowed(row: dict, drug_family: str) -> bool:
    if row["treatment"] != "Saline":
        return False
    return (drug_family == "Any drug"
            or drug_family in tuple(row.get("drug_families", ())))


def build_property_comparison(
    metric_rows: list[dict], comparison_mode: str,
    drug_family: str = "Any drug", session_trial_type: str = "",
    trial_type_a: str = "", trial_type_b: str = "",
) -> dict:
    """Build session-level paired or unpaired arrays for all property plots."""
    paired = comparison_mode != "drug_saline"
    pairs: list[tuple[dict, dict, str]] = []
    group_a: list[dict] = []
    group_b: list[dict] = []

    if comparison_mode == "baseline_drug":
        rows = _trial_rows(metric_rows, session_trial_type)
        by_day: dict[str, dict[str, list[dict]]] = {}
        for row in rows:
            day_id = row["rec_id"].rsplit("-", 1)[0]
            bucket = by_day.setdefault(day_id, {"baseline": [], "drug": []})
            if row["treatment"] == "Baseline":
                bucket["baseline"].append(row)
            elif _drug_allowed(row, drug_family):
                bucket["drug"].append(row)
        for day_id, bucket in sorted(by_day.items()):
            if len(bucket["baseline"]) == 1 and len(bucket["drug"]) == 1:
                pairs.append((bucket["baseline"][0], bucket["drug"][0], day_id))
        labels = ("Baseline", drug_family if drug_family != "Any drug" else "Drug")
        description = (f"paired recording-day comparison | trial type: {session_trial_type} | "
                       f"drug filter: {drug_family}")
    elif comparison_mode == "drug_saline":
        rows = _trial_rows(metric_rows, session_trial_type)
        group_a = [row for row in rows if _saline_allowed(row, drug_family)]
        group_b = [row for row in rows if _drug_allowed(row, drug_family)]
        labels = ("Saline", drug_family if drug_family != "Any drug" else "Drug")
        description = (f"unpaired session comparison | trial type: {session_trial_type} | "
                       f"drug filter: {drug_family}")
    elif comparison_mode == "trial_types":
        by_session: dict[str, dict[str, dict]] = {}
        for row in metric_rows:
            by_session.setdefault(row["rec_id"], {})[row["trial_type"]] = row
        for rec_id, rows in sorted(by_session.items()):
            if trial_type_a in rows and trial_type_b in rows:
                pairs.append((rows[trial_type_a], rows[trial_type_b], rec_id))
        labels = (trial_type_a, trial_type_b)
        description = "paired within-session trial-type comparison"
    else:
        raise ValueError(f"Unknown property comparison mode: {comparison_mode}")

    results = []
    for metric_key, title, ylabel in METRIC_SPECS:
        if paired:
            values_a, values_b, sample_ids = [], [], []
            for row_a, row_b, sample_id in pairs:
                value_a = _session_value(row_a, metric_key)
                value_b = _session_value(row_b, metric_key)
                if np.isfinite(value_a) and np.isfinite(value_b):
                    values_a.append(value_a); values_b.append(value_b)
                    sample_ids.append(sample_id)
        else:
            values_a = [_session_value(row, metric_key) for row in group_a]
            values_b = [_session_value(row, metric_key) for row in group_b]
            values_a = [value for value in values_a if np.isfinite(value)]
            values_b = [value for value in values_b if np.isfinite(value)]
            sample_ids = ([row["rec_id"] for row in group_a],
                          [row["rec_id"] for row in group_b])
        results.append({
            "key": metric_key, "title": title, "ylabel": ylabel,
            "values_a": np.asarray(values_a, dtype=float),
            "values_b": np.asarray(values_b, dtype=float),
            "sample_ids": sample_ids,
        })
    return {
        "mode": comparison_mode, "paired": paired, "labels": labels,
        "description": description, "metrics": results,
        "n_a": len(pairs) if paired else len(group_a),
        "n_b": len(pairs) if paired else len(group_b),
    }