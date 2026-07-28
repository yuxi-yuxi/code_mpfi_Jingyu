# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:32:38 2026

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import base64
import io
from config import CONFIG_TIME, CONFIG_PLACE


def _round_for_json(obj):
    """Recursively round floats to 4 decimal places before JSON serialization."""
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_for_json(v) for v in obj]
    return obj


def _encode_image_base64(img_norm):
    """Encode a [0,1] float image as a base64 PNG data-URI string."""
    try:
        from PIL import Image
        img_uint8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f"data:image/png;base64,{b64}"
    except ImportError:
        return None


def _subsample_sorted(sorted_list, max_n=100):
    """Keep at most max_n evenly-spaced values from a sorted list.

    This preserves percentile accuracy while drastically reducing JSON size
    (e.g. 1000 shuffle values -> 100).
    """
    if sorted_list is None or len(sorted_list) <= max_n:
        return sorted_list
    indices = np.round(np.linspace(0, len(sorted_list) - 1, max_n)).astype(int)
    return [sorted_list[i] for i in indices]


#%% helpers
def get_config(mode='time'):
    return CONFIG_TIME if mode == 'time' else CONFIG_PLACE

def trace_is_valid(trace):
    if trace is None:
        return False
    arr = np.asarray(trace)
    if arr.size == 0:
        return False
    return not np.all(np.isnan(arr))

def normalize_traces(traces):
    traces = np.asarray(traces)
    min_val = np.nanmin(traces)
    max_val = np.nanmax(traces)
    if max_val - min_val == 0:
        return traces - min_val
    return (traces - min_val) / (max_val - min_val)


def select_significant_cells(df, config,
                             info_threshold,
                             shuff_thresh,
                             mean_dff_threshold=0.1,
                             perc_active_laps_threshold=0.1,
                             perc_active_frames_threshold=0.1):
    """Mark significant cells using mode-specific info_bits / shuffled columns."""
    info_col = config['info_bits_col']
    shuffled_col = config['shuffled_info_col']
    label = config['cell_type_label']
    non_label = config['non_cell_type_label']

    df = df.copy()
    if shuff_thresh is not None and shuffled_col in df.columns:
        def _percentile(x):
            if x is None:
                return np.nan
            arr = np.asarray(x, dtype=float)
            if arr.size == 0 or np.all(np.isnan(arr)):
                return np.nan
            return np.nanpercentile(arr, shuff_thresh)
        df['shuffle_thresh'] = df[shuffled_col].apply(_percentile)
        df['is_significant'] = ((df[info_col] > info_threshold) &
                                (df[info_col] > df['shuffle_thresh']))
    else:
        df['is_significant'] = df[info_col] > info_threshold

    if mean_dff_threshold is not None:
        peak_amp_col = config['peak_amplitude_col']
        df['is_significant'] = (df['is_significant']
                                & (df[peak_amp_col] > mean_dff_threshold))

    if perc_active_laps_threshold is not None and 'perc_active_laps' in df.columns:
        df['is_significant'] = (df['is_significant']
                                & (df['perc_active_laps'] > perc_active_laps_threshold))

    if perc_active_frames_threshold is not None and 'perc_active_frames' in df.columns:
        df['is_significant'] = (df['is_significant']
                                & (df['perc_active_frames'] > perc_active_frames_threshold))

    df['cell_type'] = np.where(df['is_significant'], label, non_label)
    return df


def select_time_cell(df, TI_threshold, shuff_TI_thresh):
    return select_significant_cells(df, CONFIG_TIME, TI_threshold, shuff_TI_thresh)


def select_place_cell(df, SI_threshold, shuff_SI_thresh):
    return select_significant_cells(df, CONFIG_PLACE, SI_threshold, shuff_SI_thresh)


def find_shared_cells(df_ss1, df_ss2, key='cell_id'):
    """Return (df_ss1, df_ss2) filtered to cells present in both dataframes."""
    keys_df = df_ss1[[key]].merge(df_ss2[[key]], on=key, how='inner').drop_duplicates()
    df_ss1_shared = df_ss1.merge(keys_df, on=key, how='inner')
    df_ss2_shared = df_ss2.merge(keys_df, on=key, how='inner')
    return df_ss1_shared, df_ss2_shared


def _build_mode_data(df_valid, xaxis, config, roi_stat, active_soma_indices,
                     threshold, shuff_thresh, mean_dff_threshold=0.1,
                     perc_active_laps_threshold=0.1,
                     perc_active_frames_threshold=0.1):
    """Extract one mode+session's data into a JSON-ready dict."""
    cell_id_col = config['cell_id_col']
    field_map_col = config['field_map_col']
    info_bits_col = config['info_bits_col']
    peak_position_col = config['peak_position_col']
    peak_amplitude_col = config['peak_amplitude_col']
    shuffled_col = config.get('shuffled_info_col')

    traces_dict = {}
    per_lap_dict = {}
    cell_types = {}
    cell_info = {}

    for _, row in df_valid.iterrows():
        cell_id = row[cell_id_col]
        key = str(cell_id)
        cell_types[key] = row.get('cell_type', 'Unknown')

        if field_map_col in row and trace_is_valid(row[field_map_col]):
            traces_dict[key] = np.round(np.asarray(row[field_map_col]), 4).tolist()

        if 'per_lap_profile' in row and row['per_lap_profile'] is not None:
            per_lap_raw = row['per_lap_profile']
            per_lap_arr = None
            if isinstance(per_lap_raw, np.ndarray):
                if per_lap_raw.dtype == object:
                    try:
                        per_lap_arr = np.vstack([np.asarray(r) for r in per_lap_raw])
                    except Exception:
                        per_lap_arr = None
                else:
                    per_lap_arr = per_lap_raw
            elif isinstance(per_lap_raw, list):
                try:
                    per_lap_arr = np.array([np.asarray(r) for r in per_lap_raw])
                except Exception:
                    per_lap_arr = None
            if per_lap_arr is not None and per_lap_arr.ndim == 2 and per_lap_arr.size > 0:
                per_lap_dict[key] = np.round(per_lap_arr, 4).tolist()

        shuff_p1 = None
        shuff_p99 = None
        shuff_sorted = None
        if shuffled_col and shuffled_col in row and row[shuffled_col] is not None:
            shuff_arr = np.asarray(row[shuffled_col], dtype=float)
            if len(shuff_arr) > 0 and not np.all(np.isnan(shuff_arr)):
                clean = shuff_arr[~np.isnan(shuff_arr)]
                if len(clean) > 0:
                    shuff_sorted = _subsample_sorted(sorted(clean.tolist()))
                p1_val = np.nanpercentile(shuff_arr, 1)
                p99_val = np.nanpercentile(shuff_arr, 99)
                if not np.isnan(p1_val):
                    shuff_p1 = float(p1_val)
                if not np.isnan(p99_val):
                    shuff_p99 = float(p99_val)

        odd_even_corr = None
        if 'odd_even_corr' in row and pd.notna(row['odd_even_corr']):
            odd_even_corr = float(row['odd_even_corr'])

        perc_active_laps = None
        if 'perc_active_laps' in row and pd.notna(row['perc_active_laps']):
            perc_active_laps = float(row['perc_active_laps'])

        perc_active_frames = None
        if 'perc_active_frames' in row and pd.notna(row['perc_active_frames']):
            perc_active_frames = float(row['perc_active_frames'])

        # Tentative field masks + raw data for Dombeck filtering in JS
        tent_fields = None
        if 'tentative_field' in row and row['tentative_field'] is not None:
            raw_tf = row['tentative_field']
            if isinstance(raw_tf, (list, np.ndarray)) and len(raw_tf) > 0:
                tent_fields = [[bool(b) for b in np.asarray(m, dtype=bool)] for m in raw_tf]

        raw_pf_map = None
        if 'place_field_map' in row and row['place_field_map'] is not None:
            arr = np.asarray(row['place_field_map'], dtype=float)
            if arr.size > 0:
                raw_pf_map = np.round(arr, 4).tolist()

        tcount = None
        if 'transient_count_per_bin' in row and row['transient_count_per_bin'] is not None:
            arr = np.asarray(row['transient_count_per_bin'], dtype=float)
            if arr.size > 0:
                tcount = np.round(arr, 2).tolist()

        occ_per_bin = None
        if 'occupancy_per_bin' in row and row['occupancy_per_bin'] is not None:
            arr = np.asarray(row['occupancy_per_bin'], dtype=float)
            if arr.size > 0:
                occ_per_bin = np.round(arr, 2).tolist()

        cell_info[key] = {
            'info_bits': float(row[info_bits_col]) if pd.notna(row.get(info_bits_col)) else None,
            'peak_position': float(row.get(peak_position_col, np.nan)) if pd.notna(row.get(peak_position_col)) else None,
            'peak_amplitude': float(row.get(peak_amplitude_col, np.nan)) if pd.notna(row.get(peak_amplitude_col)) else None,
            'shuff_p1': shuff_p1,
            'shuff_p99': shuff_p99,
            'shuff_dist': shuff_sorted,
            'odd_even_corr': odd_even_corr,
            'perc_active_laps': perc_active_laps,
            'perc_active_frames': perc_active_frames,
            'tentative_fields': tent_fields,
            'pf_map': raw_pf_map,
            'tcount': tcount,
            'occ': occ_per_bin,
        }

    roi_data = {}
    n_with_coords = 0
    for _, row in df_valid.iterrows():
        cell_id = int(row[cell_id_col])
        key = str(cell_id)
        xpix = []
        ypix = []
        if roi_stat is not None and active_soma_indices is not None:
            if cell_id < len(active_soma_indices):
                original_roi_idx = active_soma_indices[cell_id]
                if original_roi_idx < len(roi_stat):
                    stat_entry = roi_stat[original_roi_idx]
                    if isinstance(stat_entry, dict) and 'xpix' in stat_entry:
                        xpix = stat_entry['xpix'].tolist()
                        ypix = stat_entry['ypix'].tolist()
                        n_with_coords += 1

        roi_data[key] = {
            'info_bits': float(row[info_bits_col]) if pd.notna(row[info_bits_col]) else None,
            'cell_type': row.get('cell_type', 'Unknown'),
            'is_significant': bool(row.get('is_significant', False)),
            'xpix': xpix,
            'ypix': ypix,
        }

    valid_info = df_valid[info_bits_col].dropna().values
    vmax = float(np.percentile(valid_info, 95)) if len(valid_info) > 0 else 1.0
    vmin = 0.0

    n_sig = int(df_valid['is_significant'].sum()) if 'is_significant' in df_valid.columns else 0
    n_valid = int(len(df_valid))
    pct_sig = (100.0 * n_sig / n_valid) if n_valid > 0 else 0.0

    stability_values = df_valid['odd_even_corr'].dropna().tolist() if 'odd_even_corr' in df_valid.columns else []
    stability_median = float(np.nanmedian(stability_values)) if len(stability_values) > 0 else None
    stability_mean = float(np.nanmean(stability_values)) if len(stability_values) > 0 else None

    sig_stability_values = []
    if 'odd_even_corr' in df_valid.columns and 'is_significant' in df_valid.columns:
        sig_stability_values = df_valid[df_valid['is_significant'] == True]['odd_even_corr'].dropna().tolist()
    sig_stability_median = float(np.nanmedian(sig_stability_values)) if len(sig_stability_values) > 0 else None
    sig_stability_mean = float(np.nanmean(sig_stability_values)) if len(sig_stability_values) > 0 else None

    df_sig = df_valid[df_valid['is_significant'] == True].copy()
    sequence_heatmap = None
    sequence_cell_ids = []
    sequence_peak_times = []
    if len(df_sig) > 0:
        df_sig_sorted = df_sig.sort_values(peak_position_col)
        sequence_cell_ids = df_sig_sorted[cell_id_col].tolist()
        sequence_peak_times = df_sig_sorted[peak_position_col].tolist()
        heatmap_rows = []
        for _, row in df_sig_sorted.iterrows():
            field_map = row[field_map_col]
            if trace_is_valid(field_map):
                heatmap_rows.append(np.round(np.asarray(field_map), 4).tolist())
            else:
                heatmap_rows.append([0] * len(xaxis))
        if heatmap_rows:
            sequence_heatmap = heatmap_rows

    xaxis_list = xaxis.tolist() if hasattr(xaxis, 'tolist') else list(xaxis)

    return {
        'cellTypes': cell_types,
        'traces': traces_dict,
        'perLapData': per_lap_dict,
        'cellInfo': cell_info,
        'roiData': roi_data,

        'sequenceHeatmap': sequence_heatmap,
        'sequenceCellIds': sequence_cell_ids,
        'sequencePeakTimes': sequence_peak_times,

        'stabilityValues': stability_values,
        'stabilityMedian': stability_median,
        'stabilityMean': stability_mean,

        'sigStabilityValues': sig_stability_values,
        'sigStabilityMedian': sig_stability_median,
        'sigStabilityMean': sig_stability_mean,

        'xaxis': xaxis_list,
        'signalLabel': config['signal_label'],
        'xLabel': config['x_label'],
        'infoLabel': config['info_label'],
        'thresholdLabel': config['threshold_label'],
        'cellTypeLabel': config['cell_type_label'],
        'nonCellTypeLabel': config['non_cell_type_label'],

        'threshold': float(threshold) if threshold is not None else 0.0,
        'useShuffle': shuff_thresh is not None,
        'shuffThresh': float(shuff_thresh) if shuff_thresh is not None else 99.0,
        'meanDffThreshold': float(mean_dff_threshold) if mean_dff_threshold is not None else 0.0,
        'percActiveLapsThreshold': float(perc_active_laps_threshold) if perc_active_laps_threshold is not None else 0.0,
        'percActiveFramesThreshold': float(perc_active_frames_threshold) if perc_active_frames_threshold is not None else 0.0,
        'binSize': float(xaxis_list[1] - xaxis_list[0]) if len(xaxis_list) > 1 else 4.0,
        'minPeakDff': 0.1,
        'minInOutRatio': 2.0,
        'minWidthCm': 12.0,
        'minTransientFraction': 0.15,
        'maxXLimit': float(config.get('max_x_limit', xaxis_list[-1] if xaxis_list else 1.0)),

        'vmin': vmin,
        'vmax': vmax,
        'nSig': n_sig,
        'nValid': n_valid,
        'pctSig': pct_sig,
    }


def _session_panels_html(session_id, session_label):
    return f'''
        <div id="session-col-{session_id}" class="session-row">
            <div class="session-header">
                <h3 class="session-title">Session {session_label}</h3>
                <div class="stats" id="stats-counter-{session_id}">Significant: -/-</div>
            </div>
            <div id="roi-info-{session_id}" class="info">Click an ROI on the map to view its field map.</div>
            <div class="session-panels">
                <div id="trace-panel-{session_id}" class="panel">
                    <div id="trace-plot-{session_id}"></div>
                </div>
                <div id="per-lap-panel-{session_id}" class="panel">
                    <div id="per-lap-heatmap-{session_id}"></div>
                </div>
                <div id="sequence-panel-{session_id}" class="panel">
                    <div id="sequence-heatmap-{session_id}"></div>
                </div>
                <div id="stability-panel-{session_id}" class="panel">
                    <div id="stability-hist-{session_id}"></div>
                </div>
                <div id="sig-stability-panel-{session_id}" class="panel">
                    <div id="sig-stability-box-{session_id}"></div>
                </div>
            </div>
        </div>
    '''





def load_mean_image(rec_id, config):
    """Load mean reference image for a session-level rec id ('anm-date-ss')."""
    anm, date, ss = rec_id.split('-')
    suite2p_path = Path(config['suite2p_pattern'].format(anm=anm, date=date, ss=ss))

    ops_path = suite2p_path / 'ops.npy'
    if ops_path.exists():
        suite2p_ops = np.load(ops_path, allow_pickle=True).item()
        return suite2p_ops[config['mean_img_key']]

    alt_path = suite2p_path.parent / 'ops.npy'
    if alt_path.exists():
        suite2p_ops = np.load(alt_path, allow_pickle=True).item()
        return suite2p_ops.get(config['mean_img_key'], suite2p_ops.get('meanImg'))

    raise FileNotFoundError(f"Could not find mean image for {rec_id}")


def load_roi_stat(rec_id, config):
    parts = rec_id.split('-')
    anm, date = parts[0], parts[1]

    gcamp_stats_path = Path(config['gcamp_stats_pattern'].format(anm=anm, date=date))
    if not gcamp_stats_path.exists():
        raise FileNotFoundError(f"Could not find gcamp_stats.npy for {rec_id}: {gcamp_stats_path}")
    gcamp_stats = np.load(gcamp_stats_path, allow_pickle=True)

    soma_class_path = Path(config['soma_class_pattern'].format(anm=anm, date=date))
    if not soma_class_path.exists():
        raise FileNotFoundError(f"Could not find soma_class.npz for {rec_id}: {soma_class_path}")
    soma_data = np.load(soma_class_path)
    is_active_soma = soma_data['is_soma']
    active_soma_indices = np.where(is_active_soma)[0]

    return gcamp_stats, active_soma_indices


def load_dataframe(rec_id, config):
    df_subdir = config['df_subdir']
    df_pattern = config['df_pattern']
    df_name = df_pattern.format(rec=rec_id)
    df_path = config['out_dir_raw_data'] / df_subdir / df_name
    if df_path.exists():
        return pd.read_parquet(df_path)
    raise FileNotFoundError(f"DataFrame not found: {df_path}")


def _load_mode_for_both_sessions(rec_ss1, rec_ss2, config, xaxis,
                                  threshold, shuff_thresh,
                                  roi_stat, active_soma_indices,
                                  mean_dff_threshold=0.1,
                                  perc_active_laps_threshold=0.1,
                                  perc_active_frames_threshold=0.1):
    """Load df for ss1 and ss2 of a mode, filter to shared cells, build bundles."""
    try:
        df1 = load_dataframe(rec_ss1, config)
    except FileNotFoundError as e:
        print(f"  [{config['cell_type_label']}] missing ss1: {e}")
        return None
    try:
        df2 = load_dataframe(rec_ss2, config)
    except FileNotFoundError as e:
        print(f"  [{config['cell_type_label']}] missing ss2: {e}")
        return None

    df1 = select_significant_cells(df1, config, threshold, shuff_thresh, mean_dff_threshold, perc_active_laps_threshold, perc_active_frames_threshold)
    df2 = select_significant_cells(df2, config, threshold, shuff_thresh, mean_dff_threshold, perc_active_laps_threshold, perc_active_frames_threshold)

    df1_shared, df2_shared = find_shared_cells(df1, df2, key=config['cell_id_col'])
    n_shared = len(df1_shared)
    print(f"  [{config['cell_type_label']}] shared cells: {n_shared} "
          f"(ss1 had {len(df1)}, ss2 had {len(df2)})")

    if n_shared == 0:
        return None

    bundle_ss1 = _build_mode_data(df1_shared, xaxis, config, roi_stat, active_soma_indices, threshold, shuff_thresh, mean_dff_threshold, perc_active_laps_threshold, perc_active_frames_threshold)
    bundle_ss2 = _build_mode_data(df2_shared, xaxis, config, roi_stat, active_soma_indices, threshold, shuff_thresh, mean_dff_threshold, perc_active_laps_threshold, perc_active_frames_threshold)
    return {'ss1': bundle_ss1, 'ss2': bundle_ss2}


def process_behaviour_data(behaviour_data):
    """Process behaviour pickle into JSON-ready speed/lick data."""
    result = {'ss1': None, 'ss2': None}
    for ss_key, beh in behaviour_data.items():
        if beh is None:
            continue
        ss_result = {}
        try:
            # Lick index (selectivity)
            lick_selectivities = beh.get('lick_selectivities', None)
            if lick_selectivities is not None:
                li = float(np.nanmean(lick_selectivities))
                if not np.isnan(li):
                    ss_result['lickIndex'] = round(li, 4)

            # Speed: already aligned to run onset at 1000Hz, plot 0-4s
            total_ms = 4000
            speed_trials = []
            for tr_speed in beh.get('speed_times_aligned', []):
                if len(tr_speed) > 0:
                    speed_arr = np.vstack(tr_speed)
                    if len(speed_arr) >= total_ms:
                        speed_trials.append(speed_arr[:total_ms, 1])
                    else:
                        padded = np.full(total_ms, np.nan)
                        padded[:len(speed_arr)] = speed_arr[:, 1]
                        speed_trials.append(padded)
            if len(speed_trials) > 0:
                speed_trials = np.vstack(speed_trials)
                ss_result['speedMean'] = np.round(np.nanmean(speed_trials, axis=0), 2).tolist()
                ss_result['speedSem'] = np.round(
                    np.nanstd(speed_trials, axis=0) / np.sqrt(speed_trials.shape[0]), 2
                ).tolist()
                ss_result['speedXaxis'] = np.round(np.arange(total_ms) / 1000, 3).tolist()

            # Lick rate histogram (0-4s, 100ms bins)
            lick_times_all = []
            for licks in beh.get('lick_times_aligned', []):
                if isinstance(licks, (list, np.ndarray)) and len(licks) > 0:
                    lick_times_all.extend(np.asarray(licks).flatten() / 1000)
            if len(lick_times_all) > 0:
                bin_edges = np.arange(0, 4.0 + 0.1, 0.1)
                lick_hist, _ = np.histogram(lick_times_all, bins=bin_edges)
                n_trials = len(beh.get('speed_times_aligned', []))
                if n_trials > 0:
                    ss_result['lickRate'] = np.round(lick_hist / n_trials / 0.1, 2).tolist()
                    ss_result['lickXaxis'] = np.round(bin_edges[:-1] + 0.05, 3).tolist()
        except Exception as e:
            print(f"  Warning: Could not process behaviour data for {ss_key}: {e}")

        if ss_result:
            result[ss_key] = ss_result
    return result


def generate_interactive_html(rec_date, mean_img,
                              time_data=None, place_data=None,
                              save_path=None,
                              session_labels=None,
                              behaviour_data=None):
    """
    Generate interactive HTML for one date (two sessions of the same FOV).

    time_data / place_data : dict, optional
        {'ss1': <bundle from _build_mode_data>, 'ss2': <bundle>}
        Either key may be missing; that session's column shows empty plots.

    session_labels : dict, optional
        {'ss1': '02', 'ss2': '04'} — used for display titles.
    """
    if time_data is None and place_data is None:
        raise ValueError("Must provide at least one of time_data or place_data")

    if session_labels is None:
        session_labels = {'ss1': 'ss1', 'ss2': 'ss2'}

    img_h, img_w = mean_img.shape
    img_norm = (mean_img - np.percentile(mean_img, 1)) / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1))
    img_norm = np.clip(img_norm, 0, 1)

    # initial_mode = 'time' if time_data is not None else 'place'
    initial_mode = 'place' if place_data is not None else 'time'

    has_time = time_data is not None
    has_place = place_data is not None

    def _ensure_both(d):
        return {'ss1': d.get('ss1'), 'ss2': d.get('ss2')}

    mode_data_js = {}
    if has_time:
        mode_data_js['time'] = _ensure_both(time_data)
    if has_place:
        mode_data_js['place'] = _ensure_both(place_data)
    mode_data_json = json.dumps(_round_for_json(mode_data_js))

    # Process behaviour data
    beh_data_js = {'ss1': None, 'ss2': None}
    if behaviour_data is not None:
        beh_data_js = process_behaviour_data(behaviour_data)
    beh_data_json = json.dumps(beh_data_js)

    img_b64 = _encode_image_base64(img_norm)
    img_json = json.dumps(np.round(img_norm, 2).tolist()) if img_b64 is None else None

    if has_time and has_place:
        mode_switch_html = (
            '<div class="mode-switch"><strong>Mode:</strong> '
            f'<input type="radio" name="mode" id="mode-time" value="time"{" checked" if initial_mode == "time" else ""}>'
            '<label for="mode-time">Time Cell</label> '
            f'<input type="radio" name="mode" id="mode-place" value="place"{" checked" if initial_mode == "place" else ""}>'
            '<label for="mode-place">Place Cell</label></div>'
        )
    else:
        mode_switch_html = ''

    session_panels_html = (
        _session_panels_html('ss1', session_labels.get('ss1', 'ss1')) +
        _session_panels_html('ss2', session_labels.get('ss2', 'ss2'))
    )

    init_ref = mode_data_js[initial_mode]['ss1'] or mode_data_js[initial_mode]['ss2']
    init_threshold = init_ref['threshold']
    init_shuff_thresh = init_ref['shuffThresh']
    init_mean_dff_thresh = init_ref['meanDffThreshold']
    init_perc_active_laps_thresh = init_ref['percActiveLapsThreshold']
    init_perc_active_frames_thresh = init_ref['percActiveFramesThreshold']
    init_min_peak_dff = init_ref['minPeakDff']
    init_min_in_out_ratio = init_ref['minInOutRatio']
    init_min_width_cm = init_ref['minWidthCm']
    init_min_transient_frac = init_ref['minTransientFraction']

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{rec_date} - Interactive Time/Place Cell Map</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: stretch; }}
        .panel {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-bottom: 5px; }}
        h3.session-title {{ margin: 5px 0; color: #444; }}
        .stats {{ color: #666; margin-bottom: 8px; font-size: 13px; }}
        .info {{ margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px; font-size: 13px; }}
        .map-column {{ display: flex; flex-direction: column; flex: 0 0 auto; gap: 8px; }}
        #map-panel {{ flex: 0 0 auto; }}
        #map-controls {{
            display: flex; flex-direction: row; align-items: center;
            gap: 15px; flex-wrap: wrap; font-size: 13px;
            padding: 10px; background: #f0f0f0; border-radius: 4px;
            margin-top: 50px; /* align bottom with roi-info for ss1 */
        }}
        #map-controls .legend {{ margin-top: 0; }}
        #response-map {{ width: 550px; height: 40vh; }}
        .beh-panel {{ flex: 1 1 auto; display: flex; flex-direction: column; }}
        .beh-panel #behaviour-plot {{ width: 550px; flex: 1 1 auto; min-height: 20vh; }}
        .sessions-stack {{ display: flex; flex-direction: column; gap: 20px; flex: 1 1 auto; min-width: 0; }}
        .session-row {{ display: flex; flex-direction: column; gap: 8px; }}
        .session-header {{ display: flex; align-items: baseline; gap: 15px; }}
        .session-panels {{ display: flex; flex-direction: row; flex-wrap: wrap; gap: 12px; align-items: flex-start; }}
        .session-panels > .panel {{
            flex: 0 0 auto;
            box-sizing: border-box;
        }}
        /* Per-group plot-div sizes for non-FOV panels */
        .session-panels [id^="trace-plot-"] {{
            width: 15vw;
            height: 30vh;
        }}
        .session-panels [id^="per-lap-heatmap-"],
        .session-panels [id^="sequence-heatmap-"] {{
            width: 13vw;
            height: 30vh;
        }}
        .session-panels [id^="stability-hist-"],
        .session-panels [id^="sig-stability-box-"] {{
            width: 11vw;
            height: 30vh;
        }}
        .legend {{ display: flex; gap: 15px; margin-top: 10px; font-size: 12px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-box {{ width: 15px; height: 15px; }}
        .controls {{
            background: white; padding: 10px 15px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px;
            display: flex; align-items: center; gap: 15px; font-size: 14px; flex-wrap: wrap;
        }}
        .controls input[type="number"] {{ width: 80px; padding: 4px 6px; font-size: 14px; }}
        .mode-switch {{ display: inline-flex; align-items: center; gap: 5px; }}
        .mode-switch label {{ margin-right: 8px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>{rec_date}</h1>
    <div class="controls">
        {mode_switch_html}
        <span>
            <label id="threshold-label" for="threshold-input">Threshold:</label>
            <input type="number" id="threshold-input" step="0.05" min="0" value="{init_threshold}">
        </span>
        <span>
            <label id="shuff-thresh-label" for="shuff-thresh-input">Shuffle Percentile:</label>
            <input type="number" id="shuff-thresh-input" step="1" min="0" max="100" value="{init_shuff_thresh}">
        </span>
        <span>
            <label for="mean-dff-thresh-input">Mean dF/F Threshold:</label>
            <input type="number" id="mean-dff-thresh-input" step="0.01" min="0" value="{init_mean_dff_thresh}">
        </span>
        <span>
            <label for="perc-active-laps-thresh-input">% Active Laps Threshold:</label>
            <input type="number" id="perc-active-laps-thresh-input" step="0.01" min="0" max="1" value="{init_perc_active_laps_thresh}">
        </span>
        <span>
            <label for="perc-active-frames-thresh-input">% Active Frames Threshold:</label>
            <input type="number" id="perc-active-frames-thresh-input" step="0.01" min="0" max="1" value="{init_perc_active_frames_thresh}">
        </span>
        <span id="shuffle-info" style="color: #888;"></span>
        <br>
        <strong style="font-size:13px;">Field criteria:</strong>
        <span>
            <label for="min-peak-dff-input">Min Peak dF/F:</label>
            <input type="number" id="min-peak-dff-input" step="0.01" min="0" value="{init_min_peak_dff}">
        </span>
        <span>
            <label for="min-in-out-ratio-input">Min In/Out Ratio:</label>
            <input type="number" id="min-in-out-ratio-input" step="0.5" min="0" value="{init_min_in_out_ratio}">
        </span>
        <span>
            <label for="min-width-cm-input">Min Width (cm):</label>
            <input type="number" id="min-width-cm-input" step="1" min="0" value="{init_min_width_cm}">
        </span>
        <span>
            <label for="min-transient-frac-input">Min Transient Frac:</label>
            <input type="number" id="min-transient-frac-input" step="0.01" min="0" max="1" value="{init_min_transient_frac}">
        </span>
    </div>

    <div class="container">
        <div class="map-column">
            <div id="map-controls">
                <div class="legend">
                    <div class="legend-item"><div class="legend-box" style="background: #440154;"></div> Low Info</div>
                    <div class="legend-item"><div class="legend-box" style="background: #21918c;"></div> Mid Info</div>
                    <div class="legend-item"><div class="legend-box" style="background: #fde725;"></div> High Info</div>
                </div>
                <div class="fov-color-switch">
                    <strong>Color by:</strong>
                    <input type="radio" name="fov-color" id="fov-ss1" value="ss1" checked>
                    <label for="fov-ss1">ss1</label>
                    <input type="radio" name="fov-color" id="fov-ss2" value="ss2">
                    <label for="fov-ss2">ss2</label>
                    <input type="radio" name="fov-color" id="fov-off" value="off">
                    <label for="fov-off">off</label>
                </div>
            </div>
            <div id="map-panel" class="panel">
                <div id="response-map"></div>
            </div>
            <div class="panel beh-panel">
                <div id="behaviour-plot"></div>
            </div>
        </div>
        <div class="sessions-stack">
        {session_panels_html}
        </div>
    </div>

    <script>
        const modeData = {mode_data_json};
        const imgH = {img_h};
        const imgW = {img_w};
        let refImg = {img_json if img_json is not None else 'null'};
        {'const _refImgB64 = "' + img_b64 + '";' if img_b64 is not None else ''}
        const behData = {beh_data_json};
        const SESSIONS = ['ss1', 'ss2'];

        let currentMode = {json.dumps(initial_mode)};
        let currentThreshold;
        let currentShuffThresh;
        let currentMeanDffThresh;
        let currentPercActiveLapsThresh;
        let currentPercActiveFramesThresh;
        let currentMinPeakDff;
        let currentMinInOutRatio;
        let currentMinWidthCm;
        let currentMinTransientFrac;
        let binSize;

        let signalLabel, xLabel, infoLabel, thresholdLabel, cellTypeLabel, nonCellTypeLabel;
        let xaxis, useShuffle, maxXLimit;
        let vmin, vmax;
        let fovRoiData;

        const state = {{ ss1: null, ss2: null }};
        let selectedRoiId = null;
        let fovColorSource = 'ss1';  // 'ss1' | 'ss2' | 'off'

        // Decode base64 PNG into 2D array for Plotly heatmap (if available)
        function decodeRefImage() {{
            if (refImg !== null) return Promise.resolve();
            if (typeof _refImgB64 === 'undefined') return Promise.resolve();
            return new Promise((resolve) => {{
                const img = new Image();
                img.onload = function() {{
                    const canvas = document.createElement('canvas');
                    canvas.width = imgW;
                    canvas.height = imgH;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    const data = ctx.getImageData(0, 0, imgW, imgH).data;
                    refImg = [];
                    for (let y = 0; y < imgH; y++) {{
                        const row = new Array(imgW);
                        for (let x = 0; x < imgW; x++) {{
                            row[x] = data[(y * imgW + x) * 4] / 255.0;
                        }}
                        refImg.push(row);
                    }}
                    resolve();
                }};
                img.src = _refImgB64;
            }});
        }}

        function computeMedian(arr) {{
            if (!arr || arr.length === 0) return null;
            const s = [...arr].sort((a, b) => a - b);
            const m = Math.floor(s.length / 2);
            return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
        }}
        function computeMean(arr) {{
            if (!arr || arr.length === 0) return null;
            return arr.reduce((a, b) => a + b, 0) / arr.length;
        }}

        // Compute percentile from a pre-sorted array (matches np.nanpercentile linear interp)
        function percentileFromSorted(sorted, pct) {{
            if (!sorted || sorted.length === 0) return null;
            if (sorted.length === 1) return sorted[0];
            const idx = (pct / 100) * (sorted.length - 1);
            const lo = Math.floor(idx);
            const hi = Math.ceil(idx);
            if (lo === hi) return sorted[lo];
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
        }}

        // Dombeck-style field filtering (mirrors pcf.filter_tentative_fields)
        function hasAcceptedField(info) {{
            if (!info.tentative_fields || info.tentative_fields.length === 0) return false;
            if (!info.pf_map || !info.tcount || !info.occ) return false;
            const pfMap = info.pf_map;
            const tcount = info.tcount;
            const occ = info.occ;
            const nBins = pfMap.length;
            for (const mask of info.tentative_fields) {{
                // (i) width
                const fieldBins = mask.reduce((s, v) => s + (v ? 1 : 0), 0);
                if (fieldBins * binSize < currentMinWidthCm) continue;
                // (ii) peak dF/F in field
                let peakVal = -Infinity;
                let inSum = 0, inN = 0, outSum = 0, outN = 0;
                let tIn = 0, occIn = 0;
                for (let b = 0; b < nBins; b++) {{
                    const v = pfMap[b];
                    if (v == null || isNaN(v)) continue;
                    if (mask[b]) {{
                        if (v > peakVal) peakVal = v;
                        inSum += v; inN++;
                        tIn += tcount[b]; occIn += occ[b];
                    }} else {{
                        outSum += v; outN++;
                    }}
                }}
                if (peakVal < currentMinPeakDff) continue;
                // (iii) in/out ratio
                const inMean = inN > 0 ? inSum / inN : 0;
                const outMean = outN > 0 ? outSum / outN : 0;
                if (outMean <= 0 || inMean <= currentMinInOutRatio * outMean) continue;
                // (iv) transient fraction
                if (occIn === 0) continue;
                if (tIn / occIn < currentMinTransientFrac) continue;
                return true;  // at least one field passes
            }}
            return false;
        }}

        // Shared significance check used by recomputeSignificance and updateTrace
        function isCellSignificant(info) {{
            if (!info || info.info_bits === null) return false;
            let isSig = info.info_bits > currentThreshold;
            if (currentShuffThresh > 0 && info.shuff_dist) {{
                const shuffCutoff = percentileFromSorted(info.shuff_dist, currentShuffThresh);
                isSig = isSig && (shuffCutoff !== null) && (info.info_bits > shuffCutoff);
            }}
            if (currentMeanDffThresh > 0) {{
                if (info.peak_amplitude === null || info.peak_amplitude <= currentMeanDffThresh) isSig = false;
            }}
            if (currentPercActiveLapsThresh > 0) {{
                if (info.perc_active_laps === null || info.perc_active_laps <= currentPercActiveLapsThresh) isSig = false;
            }}
            if (currentPercActiveFramesThresh > 0) {{
                if (info.perc_active_frames === null || info.perc_active_frames <= currentPercActiveFramesThresh) isSig = false;
            }}
            // Dombeck field criteria (only applied if tentative_fields data exists)
            if (info.tentative_fields && info.tentative_fields.length > 0) {{
                if (!hasAcceptedField(info)) isSig = false;
            }}
            return isSig;
        }}

        // Mean ± SEM across laps (axis=0), nan-aware.
        // Mirrors common.plotting_functions_Jingyu.plot_mean_trace:
        //   mean = np.nanmean(data, axis=0)
        //   sem  = np.nanstd(data, axis=0) / sqrt(n)
        function computeMeanSem(perLap) {{
            if (!perLap || perLap.length === 0) return null;
            const nLaps = perLap.length;
            const nBins = perLap[0].length;
            const mean = new Array(nBins).fill(NaN);
            const sem = new Array(nBins).fill(0);
            for (let j = 0; j < nBins; j++) {{
                let sum = 0, count = 0;
                for (let i = 0; i < nLaps; i++) {{
                    const v = perLap[i][j];
                    if (v !== null && !Number.isNaN(v)) {{ sum += v; count++; }}
                }}
                if (count === 0) continue;
                const m = sum / count;
                mean[j] = m;
                let sqSum = 0;
                for (let i = 0; i < nLaps; i++) {{
                    const v = perLap[i][j];
                    if (v !== null && !Number.isNaN(v)) sqSum += (v - m) * (v - m);
                }}
                const std = Math.sqrt(sqSum / count);
                sem[j] = std / Math.sqrt(count);
            }}
            return {{ mean, sem }};
        }}

        // Wrap Plotly.newPlot with responsive config so plots autosize on window resize.
        function plot(div, data, layout) {{
            return Plotly.newPlot(div, data, layout, {{ responsive: true }});
        }}

        // Histogram params shared across the stability / sig-stability plots
        const STAB_HIST_NBINS = 20;
        const STAB_HIST_XMIN = -0.5;
        const STAB_HIST_XMAX = 1.0;

        function binCount(values, xMin, xMax, nBins) {{
            const width = (xMax - xMin) / nBins;
            const counts = new Array(nBins).fill(0);
            for (const v of (values || [])) {{
                if (v < xMin || v > xMax) continue;
                let bin = Math.floor((v - xMin) / width);
                if (bin >= nBins) bin = nBins - 1;
                counts[bin]++;
            }}
            return counts;
        }}

        // Compute the max histogram-bin count for sigStabilityValues across both sessions,
        // so both sig-stability hists can share the same y-axis range.
        function computeSharedSigStabYMax() {{
            let m = 0;
            for (const s of SESSIONS) {{
                if (!state[s]) continue;
                const counts = binCount(state[s].sigStabilityValues, STAB_HIST_XMIN, STAB_HIST_XMAX, STAB_HIST_NBINS);
                const cm = Math.max(...counts);
                if (cm > m) m = cm;
            }}
            return Math.max(m, 1);
        }}

        function syncSigStabilityYRange() {{
            const yMax = computeSharedSigStabYMax() * 1.15;
            for (const s of SESSIONS) {{
                if (!state[s]) continue;
                const div = document.getElementById(`sig-stability-box-${{s}}`);
                if (div && div.layout) {{
                    Plotly.relayout(`sig-stability-box-${{s}}`, {{ 'yaxis.range': [0, yMax] }});
                }}
            }}
        }}

        function createBehaviourPlot() {{
            const traces = [];
            const colors = {{ ss1: 'steelblue', ss2: 'coral' }};
            const lickColors = {{ ss1: 'rgba(70,130,180,0.4)', ss2: 'rgba(250,128,114,0.4)' }};
            for (const s of SESSIONS) {{
                const bd = behData[s];
                if (!bd) continue;
                // Speed mean ± SEM
                if (bd.speedMean && bd.speedXaxis) {{
                    const upper = bd.speedMean.map((m, i) => m + (bd.speedSem ? bd.speedSem[i] : 0));
                    const lower = bd.speedMean.map((m, i) => m - (bd.speedSem ? bd.speedSem[i] : 0));
                    traces.push({{
                        x: bd.speedXaxis, y: upper, mode: 'lines',
                        line: {{ color: 'rgba(0,0,0,0)', width: 0 }},
                        hoverinfo: 'skip', showlegend: false, yaxis: 'y'
                    }});
                    traces.push({{
                        x: bd.speedXaxis, y: lower, mode: 'lines',
                        line: {{ color: 'rgba(0,0,0,0)', width: 0 }},
                        fill: 'tonexty', fillcolor: lickColors[s],
                        hoverinfo: 'skip', showlegend: false, yaxis: 'y'
                    }});
                    traces.push({{
                        x: bd.speedXaxis, y: bd.speedMean, mode: 'lines',
                        name: `Speed (${{s}})`, line: {{ color: colors[s], width: 1.5 }},
                        yaxis: 'y'
                    }});
                }}
                // Lick rate on secondary y-axis
                if (bd.lickRate && bd.lickXaxis) {{
                    traces.push({{
                        x: bd.lickXaxis, y: bd.lickRate, type: 'bar',
                        name: `Lick rate (${{s}})`, marker: {{ color: lickColors[s] }},
                        width: 0.08, yaxis: 'y2'
                    }});
                }}
            }}
            const liText = [];
            for (const s of SESSIONS) {{
                if (behData[s] && behData[s].lickIndex != null)
                    liText.push(`${{s}}: ${{behData[s].lickIndex.toFixed(3)}}`);
            }}
            const titleSuffix = liText.length > 0 ? ` | LI: ${{liText.join(', ')}}` : '';
            const layout = {{
                title: `Behaviour${{titleSuffix}}`,
                autosize: true,
                xaxis: {{ title: 'Time (s)', range: [0, 4] }},
                yaxis: {{ title: 'Speed (cm/s)', side: 'left' }},
                yaxis2: {{ title: 'Lick rate (Hz)', side: 'right', overlaying: 'y' }},
                legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.7)', font: {{ size: 10 }} }},
                margin: {{ t: 40, b: 50, l: 60, r: 60 }},
                barmode: 'overlay'
            }};
            Plotly.newPlot('behaviour-plot', traces, layout, {{ responsive: true }});
        }}

        function applyModePointers(mode) {{
            currentMode = mode;
            const ref = modeData[mode].ss1 || modeData[mode].ss2;

            signalLabel   = ref.signalLabel;
            xLabel        = ref.xLabel;
            infoLabel     = ref.infoLabel;
            thresholdLabel = ref.thresholdLabel;
            cellTypeLabel = ref.cellTypeLabel;
            nonCellTypeLabel = ref.nonCellTypeLabel;
            xaxis         = ref.xaxis;
            useShuffle    = ref.useShuffle;
            maxXLimit     = ref.maxXLimit;
            currentThreshold = ref.threshold;
            currentShuffThresh = ref.shuffThresh;
            currentMeanDffThresh = ref.meanDffThreshold;
            currentPercActiveLapsThresh = ref.percActiveLapsThreshold;
            currentPercActiveFramesThresh = ref.percActiveFramesThreshold;
            currentMinPeakDff = ref.minPeakDff;
            currentMinInOutRatio = ref.minInOutRatio;
            currentMinWidthCm = ref.minWidthCm;
            currentMinTransientFrac = ref.minTransientFraction;
            binSize = ref.binSize;
            vmin          = ref.vmin;
            vmax          = Math.max(
                modeData[mode].ss1 ? modeData[mode].ss1.vmax : 0,
                modeData[mode].ss2 ? modeData[mode].ss2.vmax : 0
            );
            fovRoiData = (modeData[mode].ss1 || modeData[mode].ss2).roiData;

            for (const s of SESSIONS) {{
                const d = modeData[mode][s];
                state[s] = d ? {{
                    cellInfo: d.cellInfo,
                    traces: d.traces,
                    roiData: d.roiData,
                    perLapData: d.perLapData,
                    sequenceHeatmap: d.sequenceHeatmap,
                    sequenceCellIds: d.sequenceCellIds,
                    sequencePeakTimes: d.sequencePeakTimes,
                    stabilityValues: d.stabilityValues,
                    stabilityMedian: d.stabilityMedian,
                    stabilityMean: d.stabilityMean,
                    sigStabilityValues: d.sigStabilityValues,
                    sigStabilityMedian: d.sigStabilityMedian,
                    sigStabilityMean: d.sigStabilityMean,
                    totalCells: Object.keys(d.cellInfo).length,
                }} : null;
            }}
        }}

        function createResponseMap() {{
            const refImgTrace = {{
                z: refImg, type: 'heatmap', colorscale: 'Greys',
                showscale: false, hoverinfo: 'skip', zmin: 0, zmax: 1
            }};

            // Pick the session whose info_bits colors the overlay.
            // Fall back to ss1 for roiIdMap (click target) regardless of what's selected,
            // since ROI coords are shared across sessions.
            let effectiveSource = fovColorSource;
            if (effectiveSource !== 'off' && !modeData[currentMode][effectiveSource]) {{
                effectiveSource = 'ss1';
            }}
            const coordSource = modeData[currentMode][effectiveSource === 'off' ? 'ss1' : effectiveSource]
                             || modeData[currentMode].ss1
                             || modeData[currentMode].ss2;
            const colorSrcData = coordSource.roiData;
            const colorSrcVmax = coordSource.vmax;

            const roiMaskMap = [];
            const roiIdMap = [];
            for (let py = 0; py < imgH; py++) {{
                roiMaskMap.push(new Array(imgW).fill(null));
                roiIdMap.push(new Array(imgW).fill(null));
            }}
            Object.keys(colorSrcData).forEach((roiId) => {{
                const d = colorSrcData[roiId];
                const xp = d.xpix, yp = d.ypix;
                const infoBits = d.info_bits !== null ? d.info_bits : 0;
                for (let i = 0; i < xp.length; i++) {{
                    const x = xp[i], y = yp[i];
                    if (y >= 0 && y < imgH && x >= 0 && x < imgW) {{
                        roiMaskMap[y][x] = infoBits;
                        roiIdMap[y][x] = roiId;
                    }}
                }}
            }});

            const showOverlay = fovColorSource !== 'off';
            const roiOverlay = {{
                z: roiMaskMap, type: 'heatmap', colorscale: 'Viridis',
                zmin: vmin, zmax: colorSrcVmax, hoverongaps: false,
                opacity: showOverlay ? 1 : 0,
                showscale: showOverlay,
                hoverinfo: showOverlay ? 'all' : 'none',
                hovertemplate: showOverlay
                    ? `Pixel (%{{y}}, %{{x}})<br>Info [${{effectiveSource}}]: %{{z:.4f}}<extra></extra>`
                    : undefined,
                colorbar: showOverlay ? {{
                    title: {{ text: '' }},
                    x: -0.02, xanchor: 'right',
                    y: 0.5, yanchor: 'middle',
                    len: 0.5, thickness: 8,
                    tickfont: {{ size: 8 }}, ticklen: 3,
                    xpad: 0
                }} : undefined
            }};
            const cbarAnnotation = showOverlay ? [{{
                text: `${{infoLabel}} [${{effectiveSource}}]`,
                xref: 'paper', yref: 'paper',
                x: -0.12, y: 0.5,
                textangle: -90,
                showarrow: false,
                font: {{ size: 9, color: '#333' }}
            }}] : [];
            const layout = {{
                title: 'Shared FOV — Click to select',
                autosize: true,
                annotations: cbarAnnotation,
                xaxis: {{
                    scaleanchor: 'y', constrain: 'domain',
                    range: [-0.5, imgW - 0.5],
                    showticklabels: false, showgrid: false,
                    zeroline: false, ticks: ''
                }},
                yaxis: {{
                    autorange: 'reversed',
                    range: [-0.5, imgH - 0.5],
                    showticklabels: false, showgrid: false,
                    zeroline: false, ticks: ''
                }},
                margin: {{ t: 40, b: 20, l: 60, r: 20 }}
            }};
            window.roiIdMap = roiIdMap;
            plot('response-map', [refImgTrace, roiOverlay], layout);
            document.getElementById('response-map').on('plotly_click', function(data) {{
                const p = data.points[0];
                const px = Math.round(p.x), py = Math.round(p.y);
                if (py >= 0 && py < imgH && px >= 0 && px < imgW) {{
                    const clickedRoiId = window.roiIdMap[py][px];
                    if (clickedRoiId !== null) selectRoi(clickedRoiId);
                }}
            }});
        }}

        function highlightSelectedROI(roiId) {{
            const mapDiv = document.getElementById('response-map');
            const layout = mapDiv.layout;
            const d = fovRoiData[roiId];
            let shapes = layout.shapes ? [...layout.shapes] : [];
            shapes = shapes.filter(s => s.line.color !== 'cyan');
            if (d && d.xpix.length > 0) {{
                const minX = Math.min(...d.xpix), maxX = Math.max(...d.xpix);
                const minY = Math.min(...d.ypix), maxY = Math.max(...d.ypix);
                shapes.push({{
                    type: 'rect',
                    x0: minX - 2, x1: maxX + 2, y0: minY - 2, y1: maxY + 2,
                    line: {{ color: 'cyan', width: 3 }},
                    fillcolor: 'rgba(0,255,255,0.15)'
                }});
            }}
            Plotly.relayout('response-map', {{ shapes: shapes }});
        }}

        function selectRoi(roiId) {{
            selectedRoiId = roiId;
            for (const s of SESSIONS) {{
                if (state[s]) {{
                    updateTrace(roiId, s);
                    highlightSequenceCell(roiId, s);
                }}
            }}
            highlightSelectedROI(roiId);
        }}

        function updateTrace(cellId, session) {{
            const st = state[session];
            if (!st) return;
            const key = cellId.toString();
            const perLap = st.perLapData[key];
            const trace = st.traces[key];
            const rd = st.roiData[key];
            const info = st.cellInfo[key];
            const cellType = isCellSignificant(info) ? cellTypeLabel : nonCellTypeLabel;
            const infoBits = rd ? rd.info_bits : null;

            const plotData = [];
            const ms = computeMeanSem(perLap);
            if (ms) {{
                const nBins = ms.mean.length;
                const x = xaxis.slice(0, nBins);
                const upper = ms.mean.map((m, i) => (m === null || Number.isNaN(m)) ? null : m + ms.sem[i]);
                const lower = ms.mean.map((m, i) => (m === null || Number.isNaN(m)) ? null : m - ms.sem[i]);
                // Upper bound (invisible), then lower bound with fill to previous for SEM band.
                plotData.push({{
                    x: x, y: upper, mode: 'lines',
                    line: {{ color: 'rgba(0,128,0,0)', width: 0 }},
                    hoverinfo: 'skip', showlegend: false
                }});
                plotData.push({{
                    x: x, y: lower, mode: 'lines',
                    line: {{ color: 'rgba(0,128,0,0)', width: 0 }},
                    fill: 'tonexty', fillcolor: 'rgba(0,128,0,0.3)',
                    hoverinfo: 'skip', showlegend: false
                }});
                plotData.push({{
                    x: x, y: ms.mean, mode: 'lines',
                    name: signalLabel, line: {{ color: 'green', width: 1 }}
                }});
            }} else if (trace) {{
                plotData.push({{
                    x: xaxis, y: trace, mode: 'lines',
                    name: signalLabel, line: {{ color: 'green', width: 1 }}
                }});
            }}
            const shapes = [];
            if (info && info.peak_position !== null) {{
                shapes.push({{
                    type: 'line',
                    x0: info.peak_position, x1: info.peak_position,
                    y0: 0, y1: 1, yref: 'paper',
                    line: {{ color: 'red', width: 2, dash: 'dash' }}
                }});
            }}
            const layout = {{
                title: `Cell ${{cellId}} [${{session}}] - ${{cellType}}`,
                autosize: true,
                xaxis: {{ title: xLabel, zeroline: false, range: [0, maxXLimit] }},
                yaxis: {{ title: 'dF/F', zeroline: true }},
                shapes: shapes, showlegend: false,
                margin: {{ t: 40, b: 50, l: 60, r: 30 }}
            }};
            plot(`trace-plot-${{session}}`, plotData.length > 0 ? plotData : [], layout);

            const infoBitsStr = infoBits !== null ? infoBits.toFixed(4) : '-';
            const peakPosStr = info && info.peak_position !== null ? info.peak_position.toFixed(2) : '-';
            const shuffP1Str = info && info.shuff_p1 !== null ? info.shuff_p1.toFixed(4) : '-';
            const shuffP99Str = info && info.shuff_p99 !== null ? info.shuff_p99.toFixed(4) : '-';
            document.getElementById(`roi-info-${{session}}`).innerHTML =
                `<strong>Cell ${{cellId}}</strong> | ` +
                `Type: <span style="color: ${{cellType === cellTypeLabel ? 'green' : 'blue'}}">${{cellType}}</span> | ` +
                `Peak: ${{peakPosStr}}<br>` +
                `Info: ${{infoBitsStr}} bits | Shuff [1%-99%]: [${{shuffP1Str}}, ${{shuffP99Str}}]`;

            updatePerLapHeatmap(cellId, session);
            highlightStabilityValue(cellId, session);
        }}

        function updatePerLapHeatmap(cellId, session) {{
            const st = state[session];
            if (!st) return;
            const key = cellId.toString();
            const perLap = st.perLapData[key];
            if (!perLap || perLap.length === 0) {{
                plot(`per-lap-heatmap-${{session}}`, [], {{
                    title: `Cell ${{cellId}} [${{session}}] - Per-Lap (No data)`,
                    autosize: true,
                    margin: {{ t: 40, b: 60, l: 60, r: 80 }}
                }});
                return;
            }}
            const nTrials = perLap.length;
            const nBins = perLap[0].length;
            const heatmapTrace = {{
                z: perLap,
                x: xaxis.slice(0, nBins),
                y: Array.from({{ length: nTrials }}, (_, i) => i + 1),
                type: 'heatmap', colorscale: 'Greys',
                colorbar: {{
                    title: {{ text: 'Activity', side: 'top' }},
                    orientation: 'h',
                    x: 0.5, xanchor: 'center',
                    y: -0.25, yanchor: 'top',
                    len: 0.8, thickness: 12
                }},
                hovertemplate: 'Trial %{{y}}<br>X: %{{x:.2f}}<br>Activity: %{{z:.3f}}<extra></extra>'
            }};
            const layout = {{
                title: `Cell ${{cellId}} [${{session}}] - Per-Lap (${{nTrials}} trials)`,
                autosize: true,
                xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                yaxis: {{ title: 'Trial', autorange: 'reversed' }},
                margin: {{ t: 40, b: 100, l: 60, r: 30 }}
            }};
            plot(`per-lap-heatmap-${{session}}`, [heatmapTrace], layout);
        }}

        function createSequenceHeatmap(session) {{
            const st = state[session];
            if (!st || !st.sequenceHeatmap || st.sequenceHeatmap.length === 0) {{
                plot(`sequence-heatmap-${{session}}`, [], {{
                    title: `${{cellTypeLabel}} Sequence [${{session}}] (None)`,
                    autosize: true,
                    margin: {{ t: 40, b: 60, l: 60, r: 80 }}
                }});
                return;
            }}
            const nCells = st.sequenceHeatmap.length;
            const heatmapTrace = {{
                z: st.sequenceHeatmap,
                x: xaxis,
                y: Array.from({{ length: nCells }}, (_, i) => i),
                type: 'heatmap', colorscale: 'Greys',
                colorbar: {{
                    title: {{ text: 'Activity', side: 'top' }},
                    orientation: 'h',
                    x: 0.5, xanchor: 'center',
                    y: -0.25, yanchor: 'top',
                    len: 0.8, thickness: 12
                }},
                hovertemplate: 'Cell %{{customdata}}<br>X: %{{x:.2f}}<br>Activity: %{{z:.3f}}<extra></extra>',
                customdata: st.sequenceCellIds.map(id => Array(xaxis.length).fill(id))
            }};
            const layout = {{
                title: `${{cellTypeLabel}}s [${{session}}] (${{nCells}})`,
                autosize: true,
                xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                yaxis: {{ title: 'Cells', tickmode: 'array', tickvals: [], ticktext: [], autorange: 'reversed' }},
                margin: {{ t: 40, b: 100, l: 60, r: 30 }}
            }};
            plot(`sequence-heatmap-${{session}}`, [heatmapTrace], layout);
            document.getElementById(`sequence-heatmap-${{session}}`).on('plotly_click', function(data) {{
                const p = data.points[0];
                const cellIdx = Math.round(p.y);
                if (cellIdx >= 0 && cellIdx < st.sequenceCellIds.length) {{
                    const cellId = st.sequenceCellIds[cellIdx];
                    selectRoi(cellId.toString());
                }}
            }});
        }}

        function highlightSequenceCell(roiId, session) {{
            const st = state[session];
            if (!st) return;
            const div = document.getElementById(`sequence-heatmap-${{session}}`);
            if (!div || !st.sequenceHeatmap || st.sequenceHeatmap.length === 0) return;
            const cellId = parseInt(roiId);
            const cellIdx = st.sequenceCellIds.indexOf(cellId);
            let shapes = [];
            if (cellIdx >= 0) {{
                shapes.push({{
                    type: 'rect',
                    x0: -0.05, x1: maxXLimit + 0.05,
                    y0: cellIdx - 0.5, y1: cellIdx + 0.5,
                    line: {{ color: 'cyan', width: 2 }},
                    fillcolor: 'rgba(0,255,255,0)'
                }});
            }}
            Plotly.relayout(`sequence-heatmap-${{session}}`, {{ shapes: shapes }});
        }}

        function createStabilityHistogram(session) {{
            const st = state[session];
            if (!st || !st.stabilityValues || st.stabilityValues.length === 0) {{
                plot(`stability-hist-${{session}}`, [], {{
                    title: `Stability [${{session}}] (No data)`,
                    autosize: true,
                    margin: {{ t: 40, b: 50, l: 60, r: 30 }}
                }});
                return;
            }}
            const histTrace = {{
                x: st.stabilityValues, type: 'histogram',
                marker: {{ color: 'lightblue', line: {{ width: 0 }} }},
                nbinsx: 20,
                hovertemplate: 'Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            }};
            const shapes = [];
            const annotations = [];
            if (st.stabilityMedian !== null) {{
                shapes.push({{ type: 'line', x0: st.stabilityMedian, x1: st.stabilityMedian, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'teal', width: 2, dash: 'dash' }} }});
                annotations.push({{ x: 0.95, y: 0.95, xref: 'paper', yref: 'paper', text: `Median = ${{st.stabilityMedian.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'teal' }}, xanchor: 'right', yanchor: 'top' }});
            }}
            if (st.stabilityMean !== null) {{
                shapes.push({{ type: 'line', x0: st.stabilityMean, x1: st.stabilityMean, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'coral', width: 2, dash: 'dot' }} }});
                annotations.push({{ x: 0.95, y: 0.85, xref: 'paper', yref: 'paper', text: `Mean = ${{st.stabilityMean.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'coral' }}, xanchor: 'right', yanchor: 'top' }});
            }}
            const layout = {{
                title: `Stability [${{session}}] (odd-even corr)`,
                autosize: true,
                xaxis: {{ title: 'Odd-Even Correlation', range: [-0.5, 1.0] }},
                yaxis: {{ title: 'ROI Count' }},
                shapes: shapes, annotations: annotations, bargap: 0.05,
                margin: {{ t: 40, b: 50, l: 60, r: 30 }}
            }};
            plot(`stability-hist-${{session}}`, [histTrace], layout);
        }}

        function highlightStabilityValue(cellId, session) {{
            const st = state[session];
            if (!st) return;
            const info = st.cellInfo[cellId.toString()];
            if (!info || info.odd_even_corr === null) return;
            const div = document.getElementById(`stability-hist-${{session}}`);
            if (!div || !div.layout) return;
            const ex = div.layout.shapes ? [...div.layout.shapes] : [];
            const filtered = ex.filter(s => s.line.color !== 'red');
            filtered.push({{
                type: 'line',
                x0: info.odd_even_corr, x1: info.odd_even_corr,
                y0: 0, y1: 1, yref: 'paper',
                line: {{ color: 'red', width: 3 }}
            }});
            const exA = div.layout.annotations ? [...div.layout.annotations] : [];
            const filteredA = exA.filter(a => a.font.color !== 'red');
            filteredA.push({{
                x: 0.95, y: 0.75, xref: 'paper', yref: 'paper',
                text: `Cell ${{cellId}} = ${{info.odd_even_corr.toFixed(2)}}`,
                showarrow: false, font: {{ size: 11, color: 'red' }},
                xanchor: 'right', yanchor: 'top'
            }});
            Plotly.relayout(`stability-hist-${{session}}`, {{ shapes: filtered, annotations: filteredA }});
        }}

        function createSigStabilityHist(session) {{
            const st = state[session];
            if (!st || !st.sigStabilityValues || st.sigStabilityValues.length === 0) {{
                plot(`sig-stability-box-${{session}}`, [], {{
                    title: `Sig. Cells Stability [${{session}}] (No data)`,
                    autosize: true,
                    margin: {{ t: 40, b: 50, l: 60, r: 30 }}
                }});
                return;
            }}
            const histTrace = {{
                x: st.sigStabilityValues, type: 'histogram',
                marker: {{ color: 'lightgreen', line: {{ width: 0 }} }},
                nbinsx: 20,
                hovertemplate: 'Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            }};
            const shapes = [];
            const annotations = [];
            if (st.sigStabilityMedian !== null) {{
                shapes.push({{ type: 'line', x0: st.sigStabilityMedian, x1: st.sigStabilityMedian, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'teal', width: 2, dash: 'dash' }} }});
                annotations.push({{ x: 0.95, y: 0.95, xref: 'paper', yref: 'paper', text: `Median = ${{st.sigStabilityMedian.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'teal' }}, xanchor: 'right', yanchor: 'top' }});
            }}
            if (st.sigStabilityMean !== null) {{
                shapes.push({{ type: 'line', x0: st.sigStabilityMean, x1: st.sigStabilityMean, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'coral', width: 2, dash: 'dot' }} }});
                annotations.push({{ x: 0.95, y: 0.85, xref: 'paper', yref: 'paper', text: `Mean = ${{st.sigStabilityMean.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'coral' }}, xanchor: 'right', yanchor: 'top' }});
            }}
            annotations.push({{ x: 0.95, y: 0.75, xref: 'paper', yref: 'paper', text: `n = ${{st.sigStabilityValues.length}}`, showarrow: false, font: {{ size: 11, color: 'gray' }}, xanchor: 'right', yanchor: 'top' }});
            const layout = {{
                title: `Sig. Cells Stability [${{session}}]`,
                autosize: true,
                xaxis: {{ title: 'Odd-Even Correlation', range: [-0.5, 1.0] }},
                yaxis: {{ title: 'Count' }},
                shapes: shapes, annotations: annotations, bargap: 0.05,
                margin: {{ t: 40, b: 50, l: 60, r: 30 }}
            }};
            plot(`sig-stability-box-${{session}}`, [histTrace], layout);
        }}

        function recomputeSignificance(threshold, session) {{
            const st = state[session];
            if (!st) return;
            const sigIds = [];
            const sigPeaks = [];
            const sigStab = [];
            for (const key in st.cellInfo) {{
                const info = st.cellInfo[key];
                if (!isCellSignificant(info)) continue;
                sigIds.push(parseInt(key));
                sigPeaks.push(info.peak_position);
                if (info.odd_even_corr !== null) sigStab.push(info.odd_even_corr);
            }}
            const order = Array.from({{ length: sigIds.length }}, (_, i) => i).sort((a, b) => {{
                const pa = sigPeaks[a], pb = sigPeaks[b];
                if (pa === null && pb === null) return 0;
                if (pa === null) return 1;
                if (pb === null) return -1;
                return pa - pb;
            }});
            const sortedIds = order.map(i => sigIds[i]);
            const sortedPeaks = order.map(i => sigPeaks[i]);
            const nBins = xaxis.length;
            const heatmapRows = sortedIds.map(id => {{
                const tr = st.traces[id.toString()];
                return tr ? tr : new Array(nBins).fill(0);
            }});
            st.sequenceHeatmap = heatmapRows.length > 0 ? heatmapRows : null;
            st.sequenceCellIds = sortedIds;
            st.sequencePeakTimes = sortedPeaks;
            st.sigStabilityValues = sigStab;
            st.sigStabilityMedian = computeMedian(sigStab);
            st.sigStabilityMean = computeMean(sigStab);

            createSequenceHeatmap(session);
            createSigStabilityHist(session);
            if (selectedRoiId !== null) {{
                highlightSequenceCell(selectedRoiId, session);
                updateTrace(selectedRoiId, session);
            }}

            const nSig = sortedIds.length;
            const tot = st.totalCells;
            const pct = tot > 0 ? (100 * nSig / tot) : 0;
            document.getElementById(`stats-counter-${{session}}`).innerHTML =
                `<strong>Session ${{session}}</strong>: ${{nSig}}/${{tot}} significant (${{pct.toFixed(1)}}%)`;
        }}

        function resetTracePanel(session) {{
            const emptyLayout = {{
                title: `Select a cell [${{session}}]`,
                autosize: true,
                xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                yaxis: {{ title: 'dF/F' }},
                margin: {{ t: 40, b: 50, l: 60, r: 30 }}
            }};
            plot(`trace-plot-${{session}}`, [], emptyLayout);
            const emptyPerLapLayout = {{
                title: `Per-Lap [${{session}}] (Select a cell)`,
                autosize: true,
                xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                yaxis: {{ title: 'Trial' }},
                margin: {{ t: 40, b: 60, l: 60, r: 80 }}
            }};
            plot(`per-lap-heatmap-${{session}}`, [], emptyPerLapLayout);
            document.getElementById(`roi-info-${{session}}`).innerHTML = 'Click an ROI on the map to view its field map.';
        }}

        function updateThresholdUI() {{
            document.getElementById('threshold-input').value = currentThreshold;
            document.getElementById('threshold-label').innerText = thresholdLabel;
            document.getElementById('shuff-thresh-input').value = currentShuffThresh;
            document.getElementById('mean-dff-thresh-input').value = currentMeanDffThresh;
            document.getElementById('perc-active-laps-thresh-input').value = currentPercActiveLapsThresh;
            document.getElementById('perc-active-frames-thresh-input').value = currentPercActiveFramesThresh;
            document.getElementById('min-peak-dff-input').value = currentMinPeakDff;
            document.getElementById('min-in-out-ratio-input').value = currentMinInOutRatio;
            document.getElementById('min-width-cm-input').value = currentMinWidthCm;
            document.getElementById('min-transient-frac-input').value = currentMinTransientFrac;
            document.getElementById('shuffle-info').innerText =
                currentShuffThresh > 0
                    ? `Shuffle test: enabled (info > ${{currentShuffThresh}}th pct of shuffled info)`
                    : 'Shuffle test: off (set percentile > 0 to enable)';
        }}

        function renderAll() {{
            updateThresholdUI();
            createResponseMap();
            createBehaviourPlot();
            for (const s of SESSIONS) {{
                if (state[s]) {{
                    createStabilityHistogram(s);
                    recomputeSignificance(currentThreshold, s);
                }} else {{
                    document.getElementById(`stats-counter-${{s}}`).innerHTML = `<strong>Session ${{s}}</strong>: no data`;
                    plot(`sequence-heatmap-${{s}}`, [], {{ title: `Sequence [${{s}}] (No data)`, autosize: true }}, {{ responsive: true }});
                    plot(`stability-hist-${{s}}`, [], {{ title: `Stability [${{s}}] (No data)`, autosize: true }}, {{ responsive: true }});
                    plot(`sig-stability-box-${{s}}`, [], {{ title: `Sig. Cells Stability [${{s}}] (No data)`, autosize: true }}, {{ responsive: true }});
                }}
            }}
            syncSigStabilityYRange();
            if (selectedRoiId !== null && fovRoiData[selectedRoiId]) {{
                for (const s of SESSIONS) {{
                    if (state[s]) updateTrace(selectedRoiId, s);
                }}
                highlightSelectedROI(selectedRoiId);
            }} else {{
                selectedRoiId = null;
                for (const s of SESSIONS) resetTracePanel(s);
            }}
        }}

        function setMode(mode) {{
            if (!modeData[mode]) return;
            applyModePointers(mode);
            renderAll();
        }}

        document.querySelectorAll('input[name="mode"]').forEach(radio => {{
            radio.addEventListener('change', function(e) {{
                if (e.target.checked) setMode(e.target.value);
            }});
        }});

        // FOV color-source switch (ss1 / ss2 / off)
        document.querySelectorAll('input[name="fov-color"]').forEach(radio => {{
            radio.addEventListener('change', function(e) {{
                if (!e.target.checked) return;
                fovColorSource = e.target.value;
                createResponseMap();
                if (selectedRoiId !== null) highlightSelectedROI(selectedRoiId);
            }});
        }});

        document.getElementById('threshold-input').addEventListener('input', function(e) {{
            const val = parseFloat(e.target.value);
            if (isNaN(val)) return;
            currentThreshold = val;
            for (const s of SESSIONS) {{
                if (modeData[currentMode][s]) modeData[currentMode][s].threshold = val;
            }}
            for (const s of SESSIONS) recomputeSignificance(val, s);
            syncSigStabilityYRange();
        }});

        document.getElementById('shuff-thresh-input').addEventListener('input', function(e) {{
            const val = parseFloat(e.target.value);
            if (isNaN(val) || val < 0 || val > 100) return;
            currentShuffThresh = val;
            for (const s of SESSIONS) {{
                if (modeData[currentMode][s]) modeData[currentMode][s].shuffThresh = val;
            }}
            document.getElementById('shuffle-info').innerText =
                val > 0 ? `Shuffle test: enabled (info > ${{val}}th pct of shuffled info)`
                         : 'Shuffle test: off (set percentile > 0 to enable)';
            for (const s of SESSIONS) recomputeSignificance(currentThreshold, s);
            syncSigStabilityYRange();
        }});

        document.getElementById('mean-dff-thresh-input').addEventListener('input', function(e) {{
            const val = parseFloat(e.target.value);
            if (isNaN(val)) return;
            currentMeanDffThresh = val;
            for (const s of SESSIONS) {{
                if (modeData[currentMode][s]) modeData[currentMode][s].meanDffThreshold = val;
            }}
            for (const s of SESSIONS) recomputeSignificance(currentThreshold, s);
            syncSigStabilityYRange();
        }});

        document.getElementById('perc-active-laps-thresh-input').addEventListener('input', function(e) {{
            const val = parseFloat(e.target.value);
            if (isNaN(val)) return;
            currentPercActiveLapsThresh = val;
            for (const s of SESSIONS) {{
                if (modeData[currentMode][s]) modeData[currentMode][s].percActiveLapsThreshold = val;
            }}
            for (const s of SESSIONS) recomputeSignificance(currentThreshold, s);
            syncSigStabilityYRange();
        }});

        document.getElementById('perc-active-frames-thresh-input').addEventListener('input', function(e) {{
            const val = parseFloat(e.target.value);
            if (isNaN(val)) return;
            currentPercActiveFramesThresh = val;
            for (const s of SESSIONS) {{
                if (modeData[currentMode][s]) modeData[currentMode][s].percActiveFramesThreshold = val;
            }}
            for (const s of SESSIONS) recomputeSignificance(currentThreshold, s);
            syncSigStabilityYRange();
        }});

        // Dombeck field criteria inputs
        function recomputeAll() {{
            for (const s of SESSIONS) recomputeSignificance(currentThreshold, s);
            syncSigStabilityYRange();
        }}
        ['min-peak-dff-input', 'min-in-out-ratio-input', 'min-width-cm-input', 'min-transient-frac-input'].forEach(id => {{
            document.getElementById(id).addEventListener('input', function(e) {{
                const val = parseFloat(e.target.value);
                if (isNaN(val)) return;
                if (id === 'min-peak-dff-input') currentMinPeakDff = val;
                else if (id === 'min-in-out-ratio-input') currentMinInOutRatio = val;
                else if (id === 'min-width-cm-input') currentMinWidthCm = val;
                else if (id === 'min-transient-frac-input') currentMinTransientFrac = val;
                recomputeAll();
            }});
        }});

        // Arrow Up/Down: navigate cells in the sequence heatmap
        document.addEventListener('keydown', function(e) {{
            if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
            // Don't intercept when typing in an input field
            if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
            e.preventDefault();
            // Use ss1 sequence as reference; fall back to ss2
            const refSt = state.ss1 || state.ss2;
            if (!refSt || !refSt.sequenceCellIds || refSt.sequenceCellIds.length === 0) return;
            const ids = refSt.sequenceCellIds;
            let curIdx = -1;
            if (selectedRoiId !== null) {{
                curIdx = ids.indexOf(parseInt(selectedRoiId));
            }}
            let newIdx;
            if (e.key === 'ArrowDown') {{
                newIdx = curIdx < 0 ? 0 : Math.min(curIdx + 1, ids.length - 1);
            }} else {{
                newIdx = curIdx < 0 ? ids.length - 1 : Math.max(curIdx - 1, 0);
            }}
            selectRoi(ids[newIdx].toString());
        }});

        applyModePointers(currentMode);
        decodeRefImage().then(() => renderAll());
    </script>
</body>
</html>
'''

    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Saved: {save_path}")

    return html_content