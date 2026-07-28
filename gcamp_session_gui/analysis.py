"""Data loading, trial grouping, and figure construction for the GCaMP GUI."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_DATA_ROOT = Path(r"Z:\Jingyu\raw_data\gcamp_drug_infusion\processed_data")
TRACK_LENGTH_CM = 180.0
@lru_cache(maxsize=1)
def infusion_session_labels() -> dict[str, str]:
    """Map full recording IDs to normalized infusion-condition labels."""
    from drug_infusion import rec_lst_infusion as recs

    rec_sch = pd.concat((recs.rec_SCH, recs.rec_SCH_ctrl))
    rec_praz = pd.concat((recs.rec_praz, recs.rec_praz_ctrl))
    rec_prop = pd.concat((recs.rec_prop, recs.rec_prop_ctrl))
    rec_all = pd.concat((rec_sch, rec_praz, rec_prop))

    def normalize(raw_label) -> str:
        label = str(raw_label).strip().lower()
        if label == "baseline":
            return "Baseline"
        if label.startswith("ctrl"):
            return "Saline"
        if label.startswith("sch"):
            return "SCH"
        if label.startswith("praz"):
            return "Prazosin"
        if label.startswith("prop"):
            return "Propranolol"
        return str(raw_label).strip() or "Unknown"

    mapping: dict[str, str] = {}
    for day_id, row in rec_all.iterrows():
        sessions = np.atleast_1d(row["session"])
        labels = np.atleast_1d(row["label"])
        for session_number, raw_label in zip(sessions, labels):
            rec_id = f"{day_id}-{str(session_number).zfill(2)}"
            treatment = normalize(raw_label)
            existing = mapping.get(rec_id)
            if existing is not None and existing != treatment:
                raise ValueError(
                    f"Conflicting infusion labels for {rec_id}: "
                    f"{existing!r} versus {treatment!r}"
                )
            mapping[rec_id] = treatment
    return mapping


def session_treatment_label(rec_id: str) -> str:
    return infusion_session_labels().get(rec_id, "Unknown")
@lru_cache(maxsize=1)
def infusion_session_families() -> dict[str, tuple[str, ...]]:
    """Map sessions to the drug experiment(s) they belong to."""
    from drug_infusion import rec_lst_infusion as recs

    sources = (
        (recs.rec_SCH, "SCH"), (recs.rec_SCH_ctrl, "SCH"),
        (recs.rec_praz, "Prazosin"), (recs.rec_praz_ctrl, "Prazosin"),
        (recs.rec_prop, "Propranolol"),
        (recs.rec_prop_ctrl, "Propranolol"),
    )
    mapping: dict[str, set[str]] = {}
    for dataframe, family in sources:
        for day_id, row in dataframe.iterrows():
            for session_number in np.atleast_1d(row["session"]):
                rec_id = f"{day_id}-{str(session_number).zfill(2)}"
                mapping.setdefault(rec_id, set()).add(family)
    return {rec_id: tuple(sorted(families))
            for rec_id, families in mapping.items()}


def session_drug_families(rec_id: str) -> tuple[str, ...]:
    return infusion_session_families().get(rec_id, ())


@dataclass
class SessionData:
    rec_id: str
    dataframe: pd.DataFrame
    behaviour: dict
    tensor: np.ndarray  # cells x laps x position bins
    lap_trial_idx: np.ndarray
    bin_centres: np.ndarray
    parquet_path: Path
    behaviour_path: Path
    treatment: str = "Unknown"
    drug_families: tuple[str, ...] = ()
    fov: "FOVData | None" = None

    @property
    def n_cells(self) -> int:
        return self.tensor.shape[0]

    @property
    def n_laps(self) -> int:
        return self.tensor.shape[1]

    @property
    def display_name(self) -> str:
        return f"{self.rec_id} [{self.treatment}]"

@dataclass
class FOVData:
    mean_image: np.ndarray
    roi_xpix: dict[int, np.ndarray]
    roi_ypix: dict[int, np.ndarray]
    source_description: str


@dataclass
class GroupedMaps:
    session: SessionData
    label_a: str
    label_b: str
    lap_idx_a: np.ndarray
    lap_idx_b: np.ndarray
    maps_a: np.ndarray
    maps_b: np.ndarray
    cell_indices: np.ndarray
    details: str
    place_cell_indices: np.ndarray | None = None
    trial_metrics: list[dict] | None = None


def discover_sessions(root: Path | str = DEFAULT_DATA_ROOT) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        if ((folder / f"{folder.name}_place_cell_dataframe_3rsd.parquet").is_file()
                and (folder / f"{folder.name}.pkl").is_file()):
            found.append(folder.name)
    return sorted(found)


def load_session(rec_id: str, root: Path | str = DEFAULT_DATA_ROOT) -> SessionData:
    folder = Path(root) / rec_id
    parquet = folder / f"{rec_id}_place_cell_dataframe_3rsd.parquet"
    behaviour_path = folder / f"{rec_id}.pkl"
    if not parquet.is_file() or not behaviour_path.is_file():
        raise FileNotFoundError(f"Session {rec_id} does not contain both expected files")

    df = pd.read_parquet(parquet)
    behaviour = pd.read_pickle(behaviour_path)
    if not isinstance(behaviour, dict):
        raise TypeError(f"Expected a dict-like behaviour pickle for {rec_id}")
    if df.empty or "per_lap_profile" not in df or "lap_trial_idx" not in df:
        raise ValueError(f"Place-cell dataframe for {rec_id} is empty or incomplete")

    profiles = []
    for value in df["per_lap_profile"]:
        profile = np.vstack([np.asarray(lap, dtype=float) for lap in value])
        profiles.append(profile)
    shape = profiles[0].shape
    if any(profile.shape != shape for profile in profiles):
        raise ValueError(f"Inconsistent per_lap_profile shapes in {rec_id}")
    tensor = np.asarray(profiles, dtype=float)
    lap_trial_idx = np.asarray(df.iloc[0]["lap_trial_idx"], dtype=int)
    if len(lap_trial_idx) != shape[0]:
        raise ValueError(f"lap_trial_idx length does not match lap count in {rec_id}")
    bin_width = TRACK_LENGTH_CM / shape[1]
    bin_centres = np.arange(shape[1], dtype=float) * bin_width + bin_width / 2
    return SessionData(rec_id, df, behaviour, tensor, lap_trial_idx, bin_centres,
                       parquet, behaviour_path,
                       treatment=session_treatment_label(rec_id),
                       drug_families=session_drug_families(rec_id))


def load_session_fov(session: SessionData, save_cache: bool = True) -> FOVData:
    """Load or create the self-contained FOV cache for a processed session.

    The cache stores the mean image and flattened ROI pixels without pickle
    objects. Dataframe ``cell_id`` indexes the original active-soma list.
    """
    cache_path = session.parquet_path.parent / f"{session.rec_id}_fov_data.npz"
    if cache_path.is_file():
        with np.load(cache_path) as cache:
            mean_image = np.asarray(cache["mean_image"], dtype=float)
            cell_ids = np.asarray(cache["cell_ids"], dtype=int)
            offsets = np.asarray(cache["offsets"], dtype=int)
            xpix_flat = np.asarray(cache["xpix"], dtype=int)
            ypix_flat = np.asarray(cache["ypix"], dtype=int)
        roi_xpix = {int(cell_id): xpix_flat[offsets[i]:offsets[i + 1]]
                    for i, cell_id in enumerate(cell_ids)}
        roi_ypix = {int(cell_id): ypix_flat[offsets[i]:offsets[i + 1]]
                    for i, cell_id in enumerate(cell_ids)}
        session.fov = FOVData(mean_image, roi_xpix, roi_ypix, cache_path.name)
        return session.fov

    anm, date, ss = session.rec_id.split("-")
    recording = Path(r"Z:\Jingyu\2P_Recording") / anm / f"{anm}-{date}" / ss
    ops_candidates = [
        recording / "suite2p" / "plane0" / "ops.npy",
        recording / "suite2p_func_detec" / "plane0" / "ops.npy",
    ]
    ops_path = next((path for path in ops_candidates if path.is_file()), None)
    raw_signals = (Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals")
                   / f"{anm}-{date}")
    stat_path = raw_signals / "gcamp_stats.npy"
    soma_path = raw_signals / "soma_class.npz"
    missing = [str(path) for path in (stat_path, soma_path) if not path.is_file()]
    if ops_path is None:
        missing.append(" or ".join(str(path) for path in ops_candidates))
    if missing:
        raise FileNotFoundError("Missing FOV asset(s): " + "; ".join(missing))

    ops = np.load(ops_path, allow_pickle=True).item()
    mean_image = np.asarray(ops.get("meanImg", ops.get("meanImgE")), dtype=float)
    roi_stat = np.load(stat_path, allow_pickle=True)
    with np.load(soma_path) as soma:
        soma_mask = np.asarray(soma["is_soma"], dtype=bool)
    active_soma_indices = np.flatnonzero(soma_mask)

    roi_xpix: dict[int, np.ndarray] = {}
    roi_ypix: dict[int, np.ndarray] = {}
    for cell_id in pd.to_numeric(session.dataframe["cell_id"], errors="coerce"):
        if not np.isfinite(cell_id):
            continue
        cell_id = int(cell_id)
        if not (0 <= cell_id < len(active_soma_indices)):
            continue
        roi_idx = int(active_soma_indices[cell_id])
        if not (0 <= roi_idx < len(roi_stat)):
            continue
        stat = roi_stat[roi_idx]
        if isinstance(stat, dict) and "xpix" in stat and "ypix" in stat:
            roi_xpix[cell_id] = np.asarray(stat["xpix"], dtype=np.int32)
            roi_ypix[cell_id] = np.asarray(stat["ypix"], dtype=np.int32)

    if save_cache:
        cell_ids = np.asarray(sorted(roi_xpix), dtype=np.int32)
        lengths = np.asarray([len(roi_xpix[int(cell_id)]) for cell_id in cell_ids], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        xpix_flat = (np.concatenate([roi_xpix[int(cell_id)] for cell_id in cell_ids])
                     if len(cell_ids) else np.array([], dtype=np.int32))
        ypix_flat = (np.concatenate([roi_ypix[int(cell_id)] for cell_id in cell_ids])
                     if len(cell_ids) else np.array([], dtype=np.int32))
        np.savez_compressed(cache_path,
                            mean_image=mean_image.astype(np.float32),
                            cell_ids=cell_ids, offsets=offsets,
                            xpix=xpix_flat, ypix=ypix_flat)
    session.fov = FOVData(mean_image, roi_xpix, roi_ypix,
                          cache_path.name if save_cache else
                          f"{ops_path.name} + {stat_path.name} + {soma_path.name}")
    return session.fov

def _first_lick_distances(behaviour: dict) -> np.ndarray:
    """Match common.utils_behaviour.extract_first_licks(distance), with fallback."""
    try:
        from common.utils_behaviour import extract_first_licks
        return np.asarray(extract_first_licks(behaviour, align_by="distance"), dtype=float)
    except (ImportError, KeyError, TypeError, ValueError):
        values = behaviour.get("lick_distances_aligned", [])
        result = []
        for licks in values:
            arr = np.asarray(licks if licks is not None else [], dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            result.append(arr[0] if len(arr) else np.nan)
        return np.asarray(result, dtype=float)


def _valid_trials(behaviour: dict, n_trials: int, time_thresh: float = 15000) -> np.ndarray:
    try:
        from common.trial_selection import seperate_valid_trial
        mask = np.asarray(seperate_valid_trial(behaviour, time_thresh=time_thresh), dtype=bool)
        if len(mask) == n_trials:
            return mask
    except (ImportError, KeyError, TypeError, ValueError):
        pass
    return np.ones(n_trials, dtype=bool)


def _stim_control_trials(behaviour: dict, max_pulse_delay: float = 500) -> tuple[np.ndarray, np.ndarray, str]:
    """Return trial-space masks using timestamps, or aligned pulse descriptions.

    Older drug-infusion pickles in this dataset have no pulse start/end arrays,
    but do have one ``pulse_descriptions`` entry per trial. A non-empty entry is
    the recorded stimulation command for that trial.
    """
    n_trials = len(behaviour.get("run_onsets", behaviour.get("run_onset_frames", [])))
    if not n_trials:
        raise ValueError("Behaviour data contain no trial index")
    valid = _valid_trials(behaviour, n_trials)

    required = ("pulse_start_times", "pulse_end_times", "run_onsets", "frame_times")
    if all(key in behaviour for key in required):
        # Local import avoids importing the executable analysis script.
        from test_lc_stim_gcamp import align_pulses
        pulse = align_pulses(behaviour, max_pulse_delay=max_pulse_delay)
        stim = valid & pulse["valid_trials"] & pulse["trials_with_stim"]
        control = valid & ~pulse["trials_with_stim"]
        source = "pulse timestamps"
    else:
        descriptions = behaviour.get("pulse_descriptions")
        if descriptions is None or len(descriptions) != n_trials:
            raise ValueError("No aligned pulse timestamps or per-trial pulse descriptions")
        has_command = np.asarray([
            bool(item) and len(item) > 0 for item in descriptions
        ], dtype=bool)
        if not np.any(has_command) or np.all(has_command):
            raise ValueError("Per-trial pulse descriptions do not contain both groups")
        stim = valid & has_command
        control = valid & ~has_command
        source = "per-trial pulse descriptions"
    return stim, control, source


def _selected_cells(
    session: SessionData, significant_only: bool,
    min_peak_dff: float, min_in_out_ratio: float,
    min_width_cm: float, min_transient_fraction: float,
) -> np.ndarray:
    if not significant_only:
        return np.arange(session.n_cells)
    try:
        from place_cell_analysis.place_cell_functions import select_significant_cells
        selected = select_significant_cells(
            session.dataframe.copy(), config=None,
            min_peak_dff=min_peak_dff, min_in_out_ratio=min_in_out_ratio,
            min_width_cm=min_width_cm,
            min_transient_fraction=min_transient_fraction,
        )
        if "cell_id" in selected and "cell_id" in session.dataframe:
            ids = set(selected.loc[selected["is_significant"], "cell_id"])
            return np.flatnonzero(session.dataframe["cell_id"].isin(ids).to_numpy())
    except (ImportError, KeyError, TypeError, ValueError):
        pass
    # Spatial-information shuffle test is the most conservative stored fallback.
    if "spatial_information_bits" in session.dataframe and "shuffled_SI" in session.dataframe:
        keep = []
        for i, row in session.dataframe.iterrows():
            shuffled = np.asarray(row["shuffled_SI"], dtype=float)
            threshold = np.nanpercentile(shuffled, 95) if shuffled.size else np.inf
            if float(row["spatial_information_bits"]) > threshold:
                keep.append(i)
        return np.asarray(keep, dtype=int)
    return np.arange(session.n_cells)


def group_session(
    session: SessionData,
    mode: Literal["all_valid", "lick_median", "lick_threshold", "stim_control"],
    early_threshold: float = 100,
    late_threshold: float = 120,
    significant_only: bool = True,
    min_peak_dff: float = 0.1,
    min_in_out_ratio: float = 2.0,
    min_width_cm: float = 18.0,
    min_transient_fraction: float = 0.1,
    max_pulse_delay: float = 500,
    speed_match_lick: bool = False,
    speed_tolerance: float = 2.0,
) -> GroupedMaps:
    trial_idx = session.lap_trial_idx
    if mode == "all_valid":
        n_trials = len(session.behaviour.get(
            "run_onsets", session.behaviour.get("run_onset_frames", [])
        ))
        if not n_trials:
            raise ValueError("Behaviour data contain no trial index")
        valid_trials = _valid_trials(session.behaviour, n_trials)
        valid_idx = (trial_idx >= 0) & (trial_idx < n_trials)
        safe_trial_idx = np.clip(trial_idx, 0, n_trials - 1)
        idx_a = np.flatnonzero(valid_idx & valid_trials[safe_trial_idx])
        idx_b = np.array([], dtype=int)
        labels = ("All valid trials", "")
        details = "all behaviour-valid trials; no sub-trial split"
    elif mode.startswith("lick_"):
        first_lick = _first_lick_distances(session.behaviour)
        lick_per_lap = np.full(session.n_laps, np.nan)
        valid_idx = (trial_idx >= 0) & (trial_idx < len(first_lick))
        lick_per_lap[valid_idx] = first_lick[trial_idx[valid_idx]]
        finite = np.isfinite(lick_per_lap)
        if mode == "lick_median":
            split = float(np.nanmedian(lick_per_lap))
            idx_a = np.flatnonzero(finite & (lick_per_lap < split))
            idx_b = np.flatnonzero(finite & (lick_per_lap >= split))
            details = f"median first lick = {split:.1f} cm"
        else:
            if early_threshold >= late_threshold:
                raise ValueError("Early threshold must be smaller than late threshold")
            idx_a = np.flatnonzero(finite & (lick_per_lap < early_threshold))
            idx_b = np.flatnonzero(finite & (lick_per_lap > late_threshold))
            details = f"early < {early_threshold:g} cm; late > {late_threshold:g} cm"
        labels = ("Early lick", "Late lick")
        if speed_match_lick:
            from common.utils_behaviour import speed_match
            early_before, late_before = len(idx_a), len(idx_b)
            early_trials = sorted(set(int(v) for v in trial_idx[idx_a]))
            late_trials = sorted(set(int(v) for v in trial_idx[idx_b]))
            try:
                matched = speed_match(
                    session.behaviour, early_trials, late_trials,
                    align_by="distance", tolerance=speed_tolerance,
                    plot_validation=False,
                )
            except Exception as exc:
                raise ValueError(f"Speed matching failed: {exc}") from exc
            if matched is None:
                raise ValueError(
                    "Speed matching found no usable distance-aligned speed trials"
                )
            matched_early_trials, matched_late_trials, p_values = matched
            idx_a = idx_a[np.isin(trial_idx[idx_a], matched_early_trials)]
            idx_b = idx_b[np.isin(trial_idx[idx_b], matched_late_trials)]
            if not len(idx_a) or not len(idx_b):
                raise ValueError(
                    f"Speed matching retained early={len(idx_a)} and late={len(idx_b)} laps"
                )
            p_values = np.asarray(p_values, dtype=float)
            finite_p = p_values[np.isfinite(p_values)]
            p_text = (f"; speed-bin p range={np.min(finite_p):.3g}-"
                      f"{np.max(finite_p):.3g}" if len(finite_p) else "")
            details += (
                f"; distance-speed matched at {speed_tolerance:g} SD: "
                f"early {early_before}->{len(idx_a)}, late {late_before}->{len(idx_b)}"
                f"{p_text}"
            )
            labels = ("Early lick - speed matched", "Late lick - speed matched")
    else:
        stim_trials, control_trials, source = _stim_control_trials(
            session.behaviour, max_pulse_delay=max_pulse_delay)
        valid_idx = (trial_idx >= 0) & (trial_idx < len(stim_trials))
        idx_a = np.flatnonzero(valid_idx & stim_trials[np.clip(trial_idx, 0, len(stim_trials)-1)])
        idx_b = np.flatnonzero(valid_idx & control_trials[np.clip(trial_idx, 0, len(control_trials)-1)])
        labels = ("Stim", "Control")
        details = f"labels from {source}"
    if mode == "all_valid" and not len(idx_a):
        raise ValueError("No behaviour-valid running laps were found")
    if mode.startswith("lick_") and (not len(idx_a) or not len(idx_b)):
        raise ValueError(f"Selected split produced {len(idx_a)} and {len(idx_b)} laps")
    if mode == "stim_control" and not (len(idx_a) or len(idx_b)):
        raise ValueError("No valid stim or control laps were found")

    place_cells = _selected_cells(
        session, True, min_peak_dff, min_in_out_ratio,
        min_width_cm, min_transient_fraction,
    )
    cells = place_cells if significant_only else np.arange(session.n_cells)
    if significant_only and not len(cells):
        raise ValueError("No cells passed the significant-place-cell filter")
    details += (
        f"; PC criteria: peak >= {min_peak_dff:g}, "
        f"in/out >= {min_in_out_ratio:g}, width >= {min_width_cm:g} cm, "
        f"transient fraction >= {min_transient_fraction:g}"
    )
    shape = (len(cells), session.tensor.shape[2])
    maps_a = (np.nanmean(session.tensor[cells][:, idx_a, :], axis=1)
              if len(idx_a) else np.full(shape, np.nan))
    maps_b = (np.nanmean(session.tensor[cells][:, idx_b, :], axis=1)
              if len(idx_b) else np.full(shape, np.nan))
    return GroupedMaps(session, labels[0], labels[1], idx_a, idx_b,
                       maps_a, maps_b, cells, details, place_cells)


def place_cell_metrics(group: GroupedMaps) -> dict:
    """Return the saved place-cell metrics and exact per-session summaries."""
    indices = (group.place_cell_indices if group.place_cell_indices is not None
               else group.cell_indices)
    indices = np.asarray(indices, dtype=int)
    rows = group.session.dataframe.iloc[indices]

    def values(column: str) -> np.ndarray:
        if column not in rows:
            return np.array([], dtype=float)
        result = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
        return result[np.isfinite(result)]

    def primary_field_values(column: str) -> np.ndarray:
        """Scalarize per-field arrays using the pipeline's primary-field rule."""
        result = []
        required = {"tentative_field", "tentative_field_in_out_ratio", column}
        if not required.issubset(rows.columns):
            return np.array([], dtype=float)
        for _, row in rows.iterrows():
            masks = row["tentative_field"]
            if masks is None or len(masks) == 0:
                continue
            primary = None
            peak_bin = row.get("place_field_peak_bin", np.nan)
            try:
                if np.isfinite(float(peak_bin)):
                    peak_bin = int(peak_bin)
                    for field_idx, mask in enumerate(masks):
                        mask = np.asarray(mask, dtype=bool)
                        if 0 <= peak_bin < mask.size and mask[peak_bin]:
                            primary = field_idx
                            break
            except (TypeError, ValueError):
                pass
            if primary is None:
                ratios = np.asarray(row["tentative_field_in_out_ratio"], dtype=float)
                primary = (int(np.nanargmax(ratios))
                           if ratios.size and np.any(np.isfinite(ratios)) else 0)
            field_values = np.asarray(row[column], dtype=float)
            if primary < field_values.size and np.isfinite(field_values[primary]):
                result.append(float(field_values[primary]))
        return np.asarray(result, dtype=float)

    locations = values("place_field_position_cm")
    reward_zone = (150.0, 180.0)
    onset_zone = (0.0, 30.0)
    expected_zone_fraction = (reward_zone[1] - reward_zone[0]) / TRACK_LENGTH_CM

    def zone_fraction(zone: tuple[float, float]) -> float:
        if not len(locations):
            return np.nan
        return float(np.mean((locations >= zone[0]) & (locations < zone[1])))

    reward_fraction = zone_fraction(reward_zone)
    onset_fraction = zone_fraction(onset_zone)
    return {
        "rec_id": group.session.rec_id,
        "treatment": group.session.treatment,
        "drug_families": group.session.drug_families,
        "n_cells": int(group.session.n_cells),
        "n_place_cells": int(len(indices)),
        "place_cell_percent": (100.0 * len(indices) / group.session.n_cells
                               if group.session.n_cells else np.nan),
        "spatial_information": values("spatial_information_bits"),
        "field_locations": locations,
        "reward_fraction": reward_fraction,
        "onset_fraction": onset_fraction,
        "expected_zone_fraction": expected_zone_fraction,
        "reward_enrichment": (reward_fraction / expected_zone_fraction
                              if np.isfinite(reward_fraction) else np.nan),
        "onset_enrichment": (onset_fraction / expected_zone_fraction
                             if np.isfinite(onset_fraction) else np.nan),
        "field_width": primary_field_values("tentative_field_width_cm"),
        "odd_even_stability": values("odd_even_corr"),
        "consecutive_stability": values("consecutive_corr"),
        "in_out_ratio": primary_field_values("tentative_field_in_out_ratio"),
        "reward_zone": reward_zone,
        "onset_zone": onset_zone,
    }

def trial_type_metrics(group: GroupedMaps) -> list[dict]:
    """Recompute paired metrics for each trial type using one fixed PC cohort."""
    if group.trial_metrics is not None:
        return group.trial_metrics
    from place_cell_analysis.place_cell_functions import (
        calculate_spatial_info, detect_tentative_fields,
    )
    from place_cell_analysis.utils_trial_correlation import (
        calculate_trial_correlations_gpu,
    )

    indices = (group.place_cell_indices if group.place_cell_indices is not None
               else group.cell_indices)
    indices = np.asarray(indices, dtype=int)
    n_pc = len(indices)
    conditions = [(group.label_a, group.lap_idx_a)]
    if group.label_b:
        conditions.append((group.label_b, group.lap_idx_b))
    results = []
    expected_zone_fraction = 30.0 / TRACK_LENGTH_CM
    bin_size = TRACK_LENGTH_CM / group.session.tensor.shape[2]

    for label, lap_indices in conditions:
        lap_indices = np.asarray(lap_indices, dtype=int)
        profiles = group.session.tensor[indices][:, lap_indices, :]
        maps = (np.nanmean(profiles, axis=1) if len(lap_indices)
                else np.full((n_pc, group.session.tensor.shape[2]), np.nan))
        with np.errstate(divide="ignore", invalid="ignore"):
            spatial_info = calculate_spatial_info(
                maps, np.ones(group.session.tensor.shape[2]), gpu=False
            )
        all_nan = np.all(~np.isfinite(maps), axis=1)
        spatial_info = np.asarray(spatial_info, dtype=float)
        spatial_info[all_nan] = np.nan
        safe_maps = np.where(np.isfinite(maps), maps, -np.inf)
        peak_bins = np.argmax(safe_maps, axis=1)
        locations = group.session.bin_centres[peak_bins].astype(float)
        locations[all_nan] = np.nan

        widths = np.full(n_pc, np.nan)
        ratios = np.full(n_pc, np.nan)
        tentative_fields = detect_tentative_fields(maps)
        for cell_idx, fields in enumerate(tentative_fields):
            if all_nan[cell_idx] or not fields:
                continue
            primary = next(
                (field for field in fields if field[peak_bins[cell_idx]]),
                fields[0],
            )
            widths[cell_idx] = float(np.sum(primary) * bin_size)
            in_mean = float(np.nanmean(maps[cell_idx, primary]))
            out_mean = float(np.nanmean(maps[cell_idx, ~primary]))
            if out_mean > 0:
                ratios[cell_idx] = in_mean / out_mean

        stability = {
            "odd_even": np.full(n_pc, np.nan),
            "consecutive": np.full(n_pc, np.nan),
        }
        profile_list = [profiles[cell_idx] for cell_idx in range(n_pc)]
        if len(lap_indices) >= 4 and n_pc:
            stability = calculate_trial_correlations_gpu(
                profile_list, methods=["odd_even", "consecutive"], gpu=False,
            )
        elif len(lap_indices) >= 2 and n_pc:
            stability["consecutive"] = calculate_trial_correlations_gpu(
                profile_list, methods="consecutive", gpu=False,
            )
        finite_locations = locations[np.isfinite(locations)]
        reward_fraction = (float(np.mean(
            (finite_locations >= 150) & (finite_locations < 180)
        )) if len(finite_locations) else np.nan)
        onset_fraction = (float(np.mean(
            (finite_locations >= 0) & (finite_locations < 30)
        )) if len(finite_locations) else np.nan)
        results.append({
            "rec_id": group.session.rec_id,
        "treatment": group.session.treatment,
        "drug_families": group.session.drug_families,
            "trial_type": label,
            "lap_count": int(len(lap_indices)),
            "n_cells": int(group.session.n_cells),
            "n_place_cells": int(n_pc),
            "place_cell_percent": (100.0 * n_pc / group.session.n_cells
                                   if group.session.n_cells else np.nan),
            "spatial_information": np.asarray(spatial_info, dtype=float),
            "field_locations": finite_locations,
            "reward_fraction": reward_fraction,
            "onset_fraction": onset_fraction,
            "expected_zone_fraction": expected_zone_fraction,
            "reward_enrichment": (reward_fraction / expected_zone_fraction
                                  if np.isfinite(reward_fraction) else np.nan),
            "onset_enrichment": (onset_fraction / expected_zone_fraction
                                 if np.isfinite(onset_fraction) else np.nan),
            "field_width": widths[np.isfinite(widths)],
            "odd_even_stability": np.asarray(stability["odd_even"], dtype=float),
            "consecutive_stability": np.asarray(stability["consecutive"], dtype=float),
            "in_out_ratio": ratios[np.isfinite(ratios)],
            "reward_zone": (150.0, 180.0),
            "onset_zone": (0.0, 30.0),
        })
    group.trial_metrics = results
    return results

def normalized_sorted_maps(group: GroupedMaps) -> tuple[np.ndarray, np.ndarray]:
    """Jointly normalize each cell across both trial types, then sort once."""
    maps_a = np.asarray(group.maps_a, dtype=float)
    maps_b = np.asarray(group.maps_b, dtype=float)
    stacked = np.stack([maps_a, maps_b])
    count = np.isfinite(stacked).sum(axis=0)
    combined = np.divide(
        np.nansum(stacked, axis=0), count,
        out=np.full_like(maps_a, np.nan), where=count > 0,
    )
    peaks = np.argmax(np.where(np.isfinite(combined), combined, -np.inf), axis=1)
    order = np.argsort(peaks)

    both = np.concatenate([maps_a, maps_b], axis=1)
    finite = np.isfinite(both)
    row_min = np.min(np.where(finite, both, np.inf), axis=1)
    row_max = np.max(np.where(finite, both, -np.inf), axis=1)
    valid_rows = np.isfinite(row_min) & np.isfinite(row_max)
    row_range = row_max - row_min

    def joint_norm(maps: np.ndarray) -> np.ndarray:
        normalized = np.full_like(maps, np.nan, dtype=float)
        varying = valid_rows & (row_range > 0)
        np.divide(
            maps - row_min[:, None], row_range[:, None], out=normalized,
            where=np.isfinite(maps) & varying[:, None],
        )
        constant = np.isfinite(maps) & valid_rows[:, None] & (row_range[:, None] == 0)
        normalized[constant] = 0.0
        return normalized[order]

    return joint_norm(maps_a), joint_norm(maps_b)







