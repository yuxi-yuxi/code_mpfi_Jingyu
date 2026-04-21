# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2026

Interactive response map with clickable ROIs for time cell / place cell analysis.
- Left panel: Response map (clickable)
- Right column: field-map trace + per-lap heatmap of the clicked ROI
- Right-most column: sequence heatmap + stability distributions
- Top controls: mode switch (time / place) + info-threshold input

Generates standalone HTML files with embedded JavaScript.

@author: Jingyu Cao
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

#============================================================================
# CONFIGURATION
#============================================================================
CONFIG_TIME = {
    'out_dir_raw_data': Path(r"Z:\Jingyu\GCaMP_drug_infusion"),
    'df_subdir': 'time_cell_dataframe',
    'df_pattern': '{rec}_time_cell_dataframe.parquet',
    'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\suite2p\plane0",
    'gcamp_stats_pattern': r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{anm}-{date}\gcamp_stats.npy",
    'soma_class_pattern': r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{anm}-{date}\soma_class.npz",
    'mean_img_key': 'meanImg',

    'cell_id_col': 'cell_id',
    'field_map_col': 'time_field_map_norm',
    'info_bits_col': 'temporal_information_bits',
    'shuffled_info_col': 'shuffled_TI',
    'peak_position_col': 'time_field_position_s',
    'peak_amplitude_col': 'time_field_peak_amplitude',

    'signal_label': 'Time Field',
    'x_label': 'Time (s)',
    'info_label': 'Temporal Info (bits)',
    'threshold_label': 'TI Threshold (bits):',
    'cell_type_label': 'Time Cell',
    'non_cell_type_label': 'Non-time Cell',
    'max_x_limit': 4.0,
}

CONFIG_PLACE = {
    **CONFIG_TIME,
    'df_subdir': 'place_cell_dataframe',
    'df_pattern': '{rec}_place_cell_dataframe.parquet',
    'field_map_col': 'place_field_map_norm',
    'info_bits_col': 'spatial_information_bits',
    'shuffled_info_col': 'shuffled_SI',
    'peak_position_col': 'place_field_position_cm',
    'peak_amplitude_col': 'place_field_peak_amplitude',
    'signal_label': 'Place Field',
    'x_label': 'Position (cm)',
    'info_label': 'Spatial Info (bits)',
    'threshold_label': 'SI Threshold (bits):',
    'cell_type_label': 'Place Cell',
    'non_cell_type_label': 'Non-place Cell',
    'max_x_limit': 180.0,
}


def get_config(mode='time'):
    """Get configuration for a given mode ('time' or 'place')."""
    return CONFIG_TIME if mode == 'time' else CONFIG_PLACE


#%% functions
def trace_is_valid(trace):
    """Check if trace is not all NaN."""
    if trace is None:
        return False
    arr = np.asarray(trace)
    if arr.size == 0:
        return False
    return not np.all(np.isnan(arr))


def normalize_traces(traces):
    """Normalize traces to 0-1 range."""
    traces = np.asarray(traces)
    min_val = np.nanmin(traces)
    max_val = np.nanmax(traces)
    if max_val - min_val == 0:
        return traces - min_val
    return (traces - min_val) / (max_val - min_val)


def select_significant_cells(df, config, info_threshold, shuff_thresh):
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
    df['cell_type'] = np.where(df['is_significant'], label, non_label)
    return df


def select_time_cell(df, TI_threshold, shuff_TI_thresh):
    return select_significant_cells(df, CONFIG_TIME, TI_threshold, shuff_TI_thresh)


def select_place_cell(df, SI_threshold, shuff_SI_thresh):
    return select_significant_cells(df, CONFIG_PLACE, SI_threshold, shuff_SI_thresh)


def _build_mode_data(df_valid, xaxis, config, roi_stat, active_soma_indices,
                     threshold, shuff_thresh):
    """Extract all per-mode data from df_valid into a JSON-ready dict."""
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
            traces_dict[key] = np.asarray(row[field_map_col]).tolist()

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
                per_lap_dict[key] = per_lap_arr.tolist()

        shuff_p1 = None
        shuff_p99 = None
        if shuffled_col and shuffled_col in row and row[shuffled_col] is not None:
            shuff_arr = np.asarray(row[shuffled_col])
            if len(shuff_arr) > 0 and not np.all(np.isnan(shuff_arr)):
                p1_val = np.nanpercentile(shuff_arr, 1)
                p99_val = np.nanpercentile(shuff_arr, 99)
                if not np.isnan(p1_val):
                    shuff_p1 = float(p1_val)
                if not np.isnan(p99_val):
                    shuff_p99 = float(p99_val)

        odd_even_corr = None
        if 'odd_even_corr' in row and pd.notna(row['odd_even_corr']):
            odd_even_corr = float(row['odd_even_corr'])

        cell_info[key] = {
            'info_bits': float(row[info_bits_col]) if pd.notna(row.get(info_bits_col)) else None,
            'peak_position': float(row.get(peak_position_col, np.nan)) if pd.notna(row.get(peak_position_col)) else None,
            'peak_amplitude': float(row.get(peak_amplitude_col, np.nan)) if pd.notna(row.get(peak_amplitude_col)) else None,
            'shuff_p1': shuff_p1,
            'shuff_p99': shuff_p99,
            'odd_even_corr': odd_even_corr,
        }

    # ROI pixel coords + per-mode info_bits / cell_type
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

    print(f"  ROIs with pixel coordinates: {n_with_coords}/{len(df_valid)} ({config['cell_type_label']} mode)")

    # Color scale based on info bits
    valid_info = df_valid[info_bits_col].dropna().values
    vmax = float(np.percentile(valid_info, 95)) if len(valid_info) > 0 else 1.0
    vmin = 0.0

    # Stats
    n_sig = int(df_valid['is_significant'].sum()) if 'is_significant' in df_valid.columns else 0
    n_valid = int(len(df_valid))
    pct_sig = (100.0 * n_sig / n_valid) if n_valid > 0 else 0.0

    # Stability distributions
    stability_values = []
    if 'odd_even_corr' in df_valid.columns:
        stability_values = df_valid['odd_even_corr'].dropna().tolist()
    stability_median = float(np.nanmedian(stability_values)) if len(stability_values) > 0 else None
    stability_mean = float(np.nanmean(stability_values)) if len(stability_values) > 0 else None

    sig_stability_values = []
    if 'odd_even_corr' in df_valid.columns and 'is_significant' in df_valid.columns:
        sig_stability_values = df_valid[df_valid['is_significant'] == True]['odd_even_corr'].dropna().tolist()
    sig_stability_median = float(np.nanmedian(sig_stability_values)) if len(sig_stability_values) > 0 else None
    sig_stability_mean = float(np.nanmean(sig_stability_values)) if len(sig_stability_values) > 0 else None

    # Sequence heatmap (sig cells sorted by peak position)
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
                heatmap_rows.append(np.asarray(field_map).tolist())
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

        'threshold': float(threshold) if threshold is not None else 0.0,
        'useShuffle': shuff_thresh is not None,
        'maxXLimit': float(config.get('max_x_limit', xaxis_list[-1] if xaxis_list else 1.0)),

        'vmin': vmin,
        'vmax': vmax,
        'nSig': n_sig,
        'nValid': n_valid,
        'pctSig': pct_sig,
    }


def generate_interactive_html(rec, mean_img,
                              time_data=None, place_data=None,
                              save_path=None):
    """
    Generate interactive HTML with response map and trace viewer.

    Parameters
    ----------
    rec : str
        Recording ID
    mean_img : np.ndarray
        Reference image for FOV
    time_data : dict, optional
        Output of _build_mode_data for time cell mode.
    place_data : dict, optional
        Output of _build_mode_data for place cell mode.
    save_path : str or Path, optional
        Path to save HTML file.

    At least one of time_data / place_data must be provided. If both are given,
    the HTML includes a mode switch.
    """
    if time_data is None and place_data is None:
        raise ValueError("Must provide at least one of time_data or place_data")

    img_h, img_w = mean_img.shape
    img_norm = (mean_img - np.percentile(mean_img, 1)) / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1))
    img_norm = np.clip(img_norm, 0, 1)

    initial_mode = 'time' if time_data is not None else 'place'
    has_time = time_data is not None
    has_place = place_data is not None

    mode_data_js = {}
    if has_time:
        mode_data_js['time'] = time_data
    if has_place:
        mode_data_js['place'] = place_data
    mode_data_json = json.dumps(mode_data_js)
    img_json = json.dumps(img_norm.tolist())

    init = mode_data_js[initial_mode]
    init_threshold = init['threshold']
    init_n_sig = init['nSig']
    init_n_valid = init['nValid']
    init_pct_sig = init['pctSig']

    # Build mode-switch UI HTML (only show if both modes present)
    if has_time and has_place:
        mode_switch_html = (
            '<div class="mode-switch">'
            '<strong>Mode:</strong> '
            f'<input type="radio" name="mode" id="mode-time" value="time"{" checked" if initial_mode == "time" else ""}>'
            '<label for="mode-time">Time Cell</label> '
            f'<input type="radio" name="mode" id="mode-place" value="place"{" checked" if initial_mode == "place" else ""}>'
            '<label for="mode-place">Place Cell</label>'
            '</div>'
        )
    else:
        mode_switch_html = ''

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{rec} - Interactive Time/Place Cell Map</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: flex-start;
        }}
        .panel {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 5px;
        }}
        .stats {{
            color: #666;
            margin-bottom: 15px;
        }}
        .info {{
            margin-top: 10px;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 4px;
            font-size: 14px;
        }}
        #map-panel {{ flex: 0 0 auto; }}
        #right-column {{ display: flex; flex-direction: column; gap: 15px; }}
        #sequence-column {{ display: flex; flex-direction: column; gap: 15px; }}
        #trace-panel {{ flex: 0 0 auto; }}
        .trace-container {{ display: flex; gap: 10px; }}
        .legend {{ display: flex; gap: 15px; margin-top: 10px; font-size: 12px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-box {{ width: 15px; height: 15px; }}
        .controls {{
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 14px;
            flex-wrap: wrap;
        }}
        .controls input[type="number"] {{ width: 80px; padding: 4px 6px; font-size: 14px; }}
        .mode-switch {{ display: inline-flex; align-items: center; gap: 5px; }}
        .mode-switch label {{ margin-right: 8px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>{rec}</h1>
    <div class="controls">
        {mode_switch_html}
        <span>
            <label id="threshold-label" for="threshold-input">TI Threshold (bits):</label>
            <input type="number" id="threshold-input" step="0.05" min="0" value="{init_threshold}">
        </span>
        <span id="shuffle-info" style="color: #888;"></span>
    </div>
    <div class="stats" id="stats-counter">Significant Cells: {init_n_sig}/{init_n_valid} ({init_pct_sig:.1f}%) | Click on an ROI to view field map</div>

    <div class="container">
        <div id="map-panel" class="panel">
            <div id="response-map"></div>
            <div class="legend">
                <div class="legend-item"><div class="legend-box" style="background: #440154;"></div> Low Info</div>
                <div class="legend-item"><div class="legend-box" style="background: #21918c;"></div> Mid Info</div>
                <div class="legend-item"><div class="legend-box" style="background: #fde725;"></div> High Info</div>
            </div>
        </div>
        <div id="right-column">
            <div id="trace-panel" class="panel">
                <div class="trace-container">
                    <div id="trace-plot"></div>
                </div>
                <div id="roi-info" class="info">Click on an ROI in the map to view its field map.</div>
            </div>
            <div id="per-lap-panel" class="panel">
                <div id="per-lap-heatmap"></div>
            </div>
        </div>
        <div id="sequence-column">
            <div id="sequence-panel" class="panel">
                <div id="sequence-heatmap"></div>
            </div>
            <div id="stability-panel" class="panel">
                <div id="stability-hist"></div>
            </div>
            <div id="sig-stability-panel" class="panel">
                <div id="sig-stability-box"></div>
            </div>
        </div>
    </div>

    <script>
        // All mode-specific data
        const modeData = {mode_data_json};

        // Shared (mode-independent)
        const refImg = {img_json};
        const imgH = {img_h};
        const imgW = {img_w};

        // Current-mode pointers — reassigned by applyMode()
        let currentMode = {json.dumps(initial_mode)};
        let cellTypes, traces, cellInfo, roiData, perLapData;
        let xaxis, signalLabel, xLabel, infoLabel, thresholdLabel, cellTypeLabel;
        let vmin, vmax, maxXLimit, useShuffle, currentThreshold;
        let sequenceHeatmap, sequenceCellIds, sequencePeakTimes;
        let stabilityValues, stabilityMedian, stabilityMean;
        let sigStabilityValues, sigStabilityMedian, sigStabilityMean;
        let totalCells;

        function applyModePointers(mode) {{
            const d = modeData[mode];
            cellTypes = d.cellTypes;
            traces = d.traces;
            cellInfo = d.cellInfo;
            roiData = d.roiData;
            perLapData = d.perLapData;
            xaxis = d.xaxis;
            signalLabel = d.signalLabel;
            xLabel = d.xLabel;
            infoLabel = d.infoLabel;
            thresholdLabel = d.thresholdLabel;
            cellTypeLabel = d.cellTypeLabel;
            vmin = d.vmin;
            vmax = d.vmax;
            maxXLimit = d.maxXLimit;
            useShuffle = d.useShuffle;
            currentThreshold = d.threshold;
            sequenceHeatmap = d.sequenceHeatmap;
            sequenceCellIds = d.sequenceCellIds;
            sequencePeakTimes = d.sequencePeakTimes;
            stabilityValues = d.stabilityValues;
            stabilityMedian = d.stabilityMedian;
            stabilityMean = d.stabilityMean;
            sigStabilityValues = d.sigStabilityValues;
            sigStabilityMedian = d.sigStabilityMedian;
            sigStabilityMean = d.sigStabilityMean;
            totalCells = Object.keys(cellInfo).length;
        }}

        // Stats helpers
        function computeMedian(arr) {{
            if (!arr || arr.length === 0) return null;
            const sorted = [...arr].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
        }}
        function computeMean(arr) {{
            if (!arr || arr.length === 0) return null;
            return arr.reduce((a, b) => a + b, 0) / arr.length;
        }}

        // Normalize trace function (unused externally but kept for parity)
        function normalizeTrace(trace) {{
            if (!trace || trace.length === 0) return null;
            const validVals = trace.filter(v => v !== null && !isNaN(v));
            if (validVals.length === 0) return null;
            const minVal = Math.min(...validVals);
            const maxVal = Math.max(...validVals);
            const range = maxVal - minVal;
            if (range === 0) return trace.map(() => 0);
            return trace.map(v => (v - minVal) / range);
        }}

        // Build and render response map (FOV overlay with ROI colored by info_bits)
        function createResponseMap() {{
            const refImgTrace = {{
                z: refImg,
                type: 'heatmap',
                colorscale: 'Greys',
                showscale: false,
                hoverinfo: 'skip',
                zmin: 0,
                zmax: 1
            }};

            const roiIds = Object.keys(roiData);
            const roiMaskMap = [];
            const roiIdMap = [];
            for (let py = 0; py < imgH; py++) {{
                roiMaskMap.push(new Array(imgW).fill(null));
                roiIdMap.push(new Array(imgW).fill(null));
            }}

            roiIds.forEach((roiId) => {{
                const data = roiData[roiId];
                const xpix = data.xpix;
                const ypix = data.ypix;
                const infoBits = data.info_bits !== null ? data.info_bits : 0;
                for (let i = 0; i < xpix.length; i++) {{
                    const x = xpix[i];
                    const y = ypix[i];
                    if (y >= 0 && y < imgH && x >= 0 && x < imgW) {{
                        roiMaskMap[y][x] = infoBits;
                        roiIdMap[y][x] = roiId;
                    }}
                }}
            }});

            const roiOverlay = {{
                z: roiMaskMap,
                type: 'heatmap',
                colorscale: 'Viridis',
                zmin: vmin,
                zmax: vmax,
                hoverongaps: false,
                hovertemplate: 'Pixel (%{{y}}, %{{x}})<br>Info: %{{z:.4f}}<extra></extra>',
                colorbar: {{
                    title: infoLabel,
                    titleside: 'right',
                    len: 0.8
                }}
            }};

            const layout = {{
                title: 'Cell Map - Click to select',
                width: 700,
                height: 700,
                xaxis: {{ title: 'X (pixels)', scaleanchor: 'y', constrain: 'domain', range: [-0.5, imgW - 0.5] }},
                yaxis: {{ title: 'Y (pixels)', autorange: 'reversed', range: [-0.5, imgH - 0.5] }},
                margin: {{ t: 50, b: 60, l: 60, r: 100 }}
            }};

            window.roiIdMap = roiIdMap;
            Plotly.newPlot('response-map', [refImgTrace, roiOverlay], layout);

            document.getElementById('response-map').on('plotly_click', function(data) {{
                const point = data.points[0];
                const px = Math.round(point.x);
                const py = Math.round(point.y);
                if (py >= 0 && py < imgH && px >= 0 && px < imgW) {{
                    const clickedRoiId = window.roiIdMap[py][px];
                    if (clickedRoiId !== null) {{
                        updateTrace(clickedRoiId);
                        highlightSelectedROI(clickedRoiId);
                    }}
                }}
            }});
        }}

        // Update trace panel for a selected ROI
        function updateTrace(cellId) {{
            const key = cellId.toString();
            const trace = traces[key];
            const data = roiData[key];
            const info = cellInfo[key];
            const cellType = data ? data.cell_type : 'Unknown';
            const infoBits = data ? data.info_bits : null;

            const plotData = [];
            if (trace) {{
                plotData.push({{
                    x: xaxis,
                    y: trace,
                    mode: 'lines',
                    name: signalLabel,
                    line: {{ color: 'green', width: 2 }}
                }});
            }}

            const shapes = [];
            if (info && info.peak_position !== null) {{
                shapes.push({{
                    type: 'line',
                    x0: info.peak_position, x1: info.peak_position,
                    y0: 0, y1: 1,
                    yref: 'paper',
                    line: {{ color: 'red', width: 2, dash: 'dash' }}
                }});
            }}

            const layout = {{
                title: `Cell ${{cellId}} - ${{cellType}}`,
                width: 500,
                height: 350,
                xaxis: {{ title: xLabel, zeroline: false, range: [0, maxXLimit] }},
                yaxis: {{ title: 'dF/F', zeroline: true }},
                shapes: shapes,
                showlegend: false,
                margin: {{ t: 50, b: 50, l: 60, r: 30 }}
            }};

            Plotly.newPlot('trace-plot', plotData.length > 0 ? plotData : [], layout);

            const infoBitsStr = infoBits !== null ? infoBits.toFixed(4) : '-';
            const peakPosStr = info && info.peak_position !== null ? info.peak_position.toFixed(2) : '-';
            const shuffP1Str = info && info.shuff_p1 !== null ? info.shuff_p1.toFixed(4) : '-';
            const shuffP99Str = info && info.shuff_p99 !== null ? info.shuff_p99.toFixed(4) : '-';
            document.getElementById('roi-info').innerHTML =
                `<strong>Cell ${{cellId}}</strong> | ` +
                `Type: <span style="color: ${{cellType === cellTypeLabel ? 'green' : 'blue'}}">${{cellType}}</span> | ` +
                `Peak: ${{peakPosStr}}<br>` +
                `Info: ${{infoBitsStr}} bits | ` +
                `Shuff [1%-99%]: [${{shuffP1Str}}, ${{shuffP99Str}}]`;

            updatePerLapHeatmap(cellId);
            highlightStabilityValue(cellId);
        }}

        // Highlight selected ROI in the response map
        let selectedRoiId = null;
        function highlightSelectedROI(roiId) {{
            const mapDiv = document.getElementById('response-map');
            const layout = mapDiv.layout;
            const data = roiData[roiId];

            let shapes = layout.shapes ? [...layout.shapes] : [];
            shapes = shapes.filter(s => s.line.color !== 'cyan');

            if (data && data.xpix.length > 0) {{
                const minX = Math.min(...data.xpix);
                const maxX = Math.max(...data.xpix);
                const minY = Math.min(...data.ypix);
                const maxY = Math.max(...data.ypix);
                shapes.push({{
                    type: 'rect',
                    x0: minX - 2, x1: maxX + 2,
                    y0: minY - 2, y1: maxY + 2,
                    line: {{ color: 'cyan', width: 3 }},
                    fillcolor: 'rgba(0,255,255,0.15)'
                }});
            }}

            Plotly.relayout('response-map', {{ shapes: shapes }});
            selectedRoiId = roiId;
            highlightSequenceCell(roiId);
        }}

        function highlightSequenceCell(roiId) {{
            const heatmapDiv = document.getElementById('sequence-heatmap');
            if (!heatmapDiv || !sequenceHeatmap || sequenceHeatmap.length === 0) return;

            const cellId = parseInt(roiId);
            const cellIdx = sequenceCellIds.indexOf(cellId);

            let shapes = [];
            if (cellIdx >= 0) {{
                shapes.push({{
                    type: 'rect',
                    x0: -0.05,
                    x1: maxXLimit + 0.05,
                    y0: cellIdx - 0.5,
                    y1: cellIdx + 0.5,
                    line: {{ color: 'cyan', width: 2 }},
                    fillcolor: 'rgba(0,255,255,0)'
                }});
            }}
            Plotly.relayout('sequence-heatmap', {{ shapes: shapes }});
        }}

        // Sequence heatmap (sig cells sorted by peak position)
        function createSequenceHeatmap() {{
            if (!sequenceHeatmap || sequenceHeatmap.length === 0) {{
                const emptyLayout = {{
                    title: `${{cellTypeLabel}} Sequence (No significant cells)`,
                    width: 400,
                    height: 350,
                    margin: {{ t: 50, b: 60, l: 60, r: 80 }}
                }};
                Plotly.newPlot('sequence-heatmap', [], emptyLayout);
                return;
            }}

            const nCells = sequenceHeatmap.length;

            const heatmapTrace = {{
                z: sequenceHeatmap,
                x: xaxis,
                y: Array.from({{ length: nCells }}, (_, i) => i),
                type: 'heatmap',
                colorscale: 'Greys',
                colorbar: {{ title: 'Activity', titleside: 'right', len: 0.8 }},
                hovertemplate: 'Cell %{{customdata}}<br>X: %{{x:.2f}}<br>Activity: %{{z:.3f}}<extra></extra>',
                customdata: sequenceCellIds.map(id => Array(xaxis.length).fill(id))
            }};

            const layout = {{
                title: `${{cellTypeLabel}}s (${{nCells}}, sorted by peak)`,
                width: 400,
                height: 350,
                xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                yaxis: {{ title: 'Cells', tickmode: 'array', tickvals: [], ticktext: [], autorange: 'reversed' }},
                margin: {{ t: 50, b: 60, l: 60, r: 80 }}
            }};

            Plotly.newPlot('sequence-heatmap', [heatmapTrace], layout);

            document.getElementById('sequence-heatmap').on('plotly_click', function(data) {{
                const point = data.points[0];
                const cellIdx = Math.round(point.y);
                if (cellIdx >= 0 && cellIdx < sequenceCellIds.length) {{
                    const cellId = sequenceCellIds[cellIdx];
                    updateTrace(cellId.toString());
                    highlightSelectedROI(cellId.toString());
                }}
            }});
        }}

        // Per-lap activity heatmap for a selected cell
        function updatePerLapHeatmap(cellId) {{
            const key = cellId.toString();
            const perLap = perLapData[key];

            if (!perLap || perLap.length === 0) {{
                const emptyLayout = {{
                    title: `Cell ${{cellId}} - Per-Lap Activity (No data)`,
                    width: 500,
                    height: 350,
                    margin: {{ t: 50, b: 60, l: 60, r: 80 }}
                }};
                Plotly.newPlot('per-lap-heatmap', [], emptyLayout);
                return;
            }}

            const nTrials = perLap.length;
            const nBins = perLap[0].length;

            const heatmapTrace = {{
                z: perLap,
                x: xaxis.slice(0, nBins),
                y: Array.from({{ length: nTrials }}, (_, i) => i + 1),
                type: 'heatmap',
                colorscale: 'Greys',
                colorbar: {{ title: 'Activity', titleside: 'right', len: 0.8 }},
                hovertemplate: 'Trial %{{y}}<br>X: %{{x:.2f}}<br>Activity: %{{z:.3f}}<extra></extra>'
            }};

            const layout = {{
                title: `Cell ${{cellId}} - Per-Lap Activity (${{nTrials}} trials)`,
                width: 500,
                height: 350,
                xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                yaxis: {{ title: 'Trial', autorange: 'reversed' }},
                margin: {{ t: 50, b: 60, l: 60, r: 80 }}
            }};

            Plotly.newPlot('per-lap-heatmap', [heatmapTrace], layout);
        }}

        // Stability distribution (all cells — threshold-independent)
        function createStabilityHistogram() {{
            if (!stabilityValues || stabilityValues.length === 0) {{
                const emptyLayout = {{
                    title: 'Stability Distribution (No data)',
                    width: 350, height: 350,
                    margin: {{ t: 50, b: 60, l: 60, r: 30 }}
                }};
                Plotly.newPlot('stability-hist', [], emptyLayout);
                return;
            }}

            const histTrace = {{
                x: stabilityValues,
                type: 'histogram',
                marker: {{ color: 'lightblue', line: {{ width: 0 }} }},
                nbinsx: 20,
                hovertemplate: 'Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            }};

            const shapes = [];
            const annotations = [];

            if (stabilityMedian !== null) {{
                shapes.push({{ type: 'line', x0: stabilityMedian, x1: stabilityMedian, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'teal', width: 2, dash: 'dash' }} }});
                annotations.push({{ x: 0.95, y: 0.95, xref: 'paper', yref: 'paper', text: `Median = ${{stabilityMedian.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'teal' }}, xanchor: 'right', yanchor: 'top' }});
            }}
            if (stabilityMean !== null) {{
                shapes.push({{ type: 'line', x0: stabilityMean, x1: stabilityMean, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'coral', width: 2, dash: 'dot' }} }});
                annotations.push({{ x: 0.95, y: 0.85, xref: 'paper', yref: 'paper', text: `Mean = ${{stabilityMean.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'coral' }}, xanchor: 'right', yanchor: 'top' }});
            }}

            const layout = {{
                title: 'Stability Distribution (odd-even corr)',
                width: 350, height: 350,
                xaxis: {{ title: 'Odd-Even Correlation', range: [-0.5, 1.0] }},
                yaxis: {{ title: 'ROI Count' }},
                shapes: shapes,
                annotations: annotations,
                margin: {{ t: 50, b: 60, l: 60, r: 30 }},
                bargap: 0.05
            }};

            Plotly.newPlot('stability-hist', [histTrace], layout);
        }}

        function highlightStabilityValue(cellId) {{
            const key = cellId.toString();
            const info = cellInfo[key];
            if (!info || info.odd_even_corr === null) return;

            const stabDiv = document.getElementById('stability-hist');
            if (!stabDiv || !stabDiv.layout) return;

            const existingShapes = stabDiv.layout.shapes ? [...stabDiv.layout.shapes] : [];
            const filteredShapes = existingShapes.filter(s => s.line.color !== 'red');

            filteredShapes.push({{
                type: 'line',
                x0: info.odd_even_corr, x1: info.odd_even_corr,
                y0: 0, y1: 1,
                yref: 'paper',
                line: {{ color: 'red', width: 3 }}
            }});

            const existingAnnotations = stabDiv.layout.annotations ? [...stabDiv.layout.annotations] : [];
            const filteredAnnotations = existingAnnotations.filter(a => a.font.color !== 'red');
            filteredAnnotations.push({{
                x: 0.95, y: 0.75, xref: 'paper', yref: 'paper',
                text: `Cell ${{cellId}} = ${{info.odd_even_corr.toFixed(2)}}`,
                showarrow: false, font: {{ size: 11, color: 'red' }},
                xanchor: 'right', yanchor: 'top'
            }});

            Plotly.relayout('stability-hist', {{ shapes: filteredShapes, annotations: filteredAnnotations }});
        }}

        function createSigStabilityHist() {{
            if (!sigStabilityValues || sigStabilityValues.length === 0) {{
                const emptyLayout = {{
                    title: 'Sig. Cells Stability (No data)',
                    width: 350, height: 300,
                    margin: {{ t: 50, b: 60, l: 60, r: 30 }}
                }};
                Plotly.newPlot('sig-stability-box', [], emptyLayout);
                return;
            }}

            const histTrace = {{
                x: sigStabilityValues,
                type: 'histogram',
                marker: {{ color: 'lightgreen', line: {{ width: 0 }} }},
                nbinsx: 20,
                hovertemplate: 'Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            }};

            const shapes = [];
            const annotations = [];
            if (sigStabilityMedian !== null) {{
                shapes.push({{ type: 'line', x0: sigStabilityMedian, x1: sigStabilityMedian, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'teal', width: 2, dash: 'dash' }} }});
                annotations.push({{ x: 0.95, y: 0.95, xref: 'paper', yref: 'paper', text: `Median = ${{sigStabilityMedian.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'teal' }}, xanchor: 'right', yanchor: 'top' }});
            }}
            if (sigStabilityMean !== null) {{
                shapes.push({{ type: 'line', x0: sigStabilityMean, x1: sigStabilityMean, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'coral', width: 2, dash: 'dot' }} }});
                annotations.push({{ x: 0.95, y: 0.85, xref: 'paper', yref: 'paper', text: `Mean = ${{sigStabilityMean.toFixed(2)}}`, showarrow: false, font: {{ size: 11, color: 'coral' }}, xanchor: 'right', yanchor: 'top' }});
            }}
            annotations.push({{ x: 0.95, y: 0.75, xref: 'paper', yref: 'paper', text: `n = ${{sigStabilityValues.length}}`, showarrow: false, font: {{ size: 11, color: 'gray' }}, xanchor: 'right', yanchor: 'top' }});

            const layout = {{
                title: 'Sig. Cells Stability',
                width: 350, height: 300,
                xaxis: {{ title: 'Odd-Even Correlation', range: [-0.5, 1.0] }},
                yaxis: {{ title: 'Count' }},
                shapes: shapes,
                annotations: annotations,
                margin: {{ t: 50, b: 60, l: 60, r: 30 }},
                bargap: 0.05
            }};

            Plotly.newPlot('sig-stability-box', [histTrace], layout);
        }}

        // Recompute significance client-side from the info-threshold.
        // Mirrors the Python logic: info > threshold, and if shuffle is enabled,
        // info > shuff_p99 as well.
        function recomputeSignificance(threshold) {{
            const sigIds = [];
            const sigPeaks = [];
            const sigStabVals = [];

            for (const key in cellInfo) {{
                const info = cellInfo[key];
                if (info.info_bits === null) continue;

                let isSig = info.info_bits > threshold;
                if (useShuffle) {{
                    if (info.shuff_p99 === null) {{
                        isSig = false;
                    }} else {{
                        isSig = isSig && (info.info_bits > info.shuff_p99);
                    }}
                }}
                if (!isSig) continue;

                sigIds.push(parseInt(key));
                sigPeaks.push(info.peak_position);
                if (info.odd_even_corr !== null) sigStabVals.push(info.odd_even_corr);
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
                const tr = traces[id.toString()];
                return tr ? tr : new Array(nBins).fill(0);
            }});

            sequenceHeatmap = heatmapRows.length > 0 ? heatmapRows : null;
            sequenceCellIds = sortedIds;
            sequencePeakTimes = sortedPeaks;

            sigStabilityValues = sigStabVals;
            sigStabilityMedian = computeMedian(sigStabVals);
            sigStabilityMean = computeMean(sigStabVals);

            createSequenceHeatmap();
            createSigStabilityHist();

            if (selectedRoiId !== null) highlightSequenceCell(selectedRoiId);

            const nSig = sortedIds.length;
            const pct = totalCells > 0 ? (100 * nSig / totalCells) : 0;
            document.getElementById('stats-counter').innerHTML =
                `Significant Cells: ${{nSig}}/${{totalCells}} (${{pct.toFixed(1)}}%) | Click on an ROI to view field map`;
        }}

        // Render everything from the current active-mode pointers.
        // Runs recomputeSignificance so the sequence heatmap and sig-stability
        // are rebuilt at the mode's current threshold.
        function renderAll() {{
            // Update threshold input + label + shuffle-info text
            document.getElementById('threshold-input').value = currentThreshold;
            document.getElementById('threshold-label').innerText = thresholdLabel;
            document.getElementById('shuffle-info').innerText =
                useShuffle ? 'Shuffle test: enabled (info > 99th percentile of shuffled info)'
                           : 'Shuffle test: disabled';

            createResponseMap();
            createStabilityHistogram();
            recomputeSignificance(currentThreshold);

            // Keep selection across mode switches if the cell exists
            if (selectedRoiId !== null && roiData[selectedRoiId]) {{
                updateTrace(selectedRoiId);
                highlightSelectedROI(selectedRoiId);
            }} else {{
                selectedRoiId = null;
                const emptyLayout = {{
                    title: 'Select a cell',
                    width: 500, height: 350,
                    xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                    yaxis: {{ title: 'dF/F' }},
                    margin: {{ t: 50, b: 50, l: 60, r: 30 }}
                }};
                Plotly.newPlot('trace-plot', [], emptyLayout);
                const emptyPerLapLayout = {{
                    title: 'Per-Lap Activity (Select a cell)',
                    width: 500, height: 350,
                    xaxis: {{ title: xLabel, range: [0, maxXLimit] }},
                    yaxis: {{ title: 'Trial' }},
                    margin: {{ t: 50, b: 60, l: 60, r: 80 }}
                }};
                Plotly.newPlot('per-lap-heatmap', [], emptyPerLapLayout);
                document.getElementById('roi-info').innerHTML = 'Click on an ROI in the map to view its field map.';
            }}
        }}

        function setMode(mode) {{
            if (!modeData[mode]) return;
            currentMode = mode;
            applyModePointers(mode);
            renderAll();
        }}

        // Wire mode-switch radio buttons (if present)
        document.querySelectorAll('input[name="mode"]').forEach(radio => {{
            radio.addEventListener('change', function(e) {{
                if (e.target.checked) setMode(e.target.value);
            }});
        }});

        // Wire threshold input
        document.getElementById('threshold-input').addEventListener('input', function(e) {{
            const val = parseFloat(e.target.value);
            if (!isNaN(val)) {{
                currentThreshold = val;
                // Persist into the active mode bundle so mode switches remember it
                modeData[currentMode].threshold = val;
                recomputeSignificance(val);
            }}
        }});

        // Initialize
        applyModePointers(currentMode);
        renderAll();
    </script>
</body>
</html>
'''

    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Saved: {save_path}")

    return html_content


def load_mean_image(rec, config):
    """Load mean reference image based on configuration."""
    anm, date, ss = rec.split('-')
    suite2p_path = Path(config['suite2p_pattern'].format(anm=anm, date=date, ss=ss))

    ops_path = suite2p_path / 'ops.npy'
    if ops_path.exists():
        suite2p_ops = np.load(ops_path, allow_pickle=True).item()
        return suite2p_ops[config['mean_img_key']]

    alt_path = suite2p_path.parent / 'ops.npy'
    if alt_path.exists():
        suite2p_ops = np.load(alt_path, allow_pickle=True).item()
        return suite2p_ops.get(config['mean_img_key'], suite2p_ops.get('meanImg'))

    raise FileNotFoundError(f"Could not find mean image for {rec}")


def load_roi_stat(rec, config):
    """Load gcamp_stats.npy and soma_class.npz for active-soma ROI coords."""
    parts = rec.split('-')
    anm, date = parts[0], parts[1]

    gcamp_stats_path = Path(config['gcamp_stats_pattern'].format(anm=anm, date=date))
    if not gcamp_stats_path.exists():
        raise FileNotFoundError(f"Could not find gcamp_stats.npy for {rec}: {gcamp_stats_path}")
    gcamp_stats = np.load(gcamp_stats_path, allow_pickle=True)

    soma_class_path = Path(config['soma_class_pattern'].format(anm=anm, date=date))
    if not soma_class_path.exists():
        raise FileNotFoundError(f"Could not find soma_class.npz for {rec}: {soma_class_path}")
    soma_data = np.load(soma_class_path)
    is_active_soma = soma_data['is_soma']
    active_soma_indices = np.where(is_active_soma)[0]

    return gcamp_stats, active_soma_indices


def load_dataframe(rec, config):
    """Load processed dataframe based on configuration."""
    df_subdir = config['df_subdir']
    df_pattern = config['df_pattern']
    df_name = df_pattern.format(rec=rec)
    df_path = config['out_dir_raw_data'] / df_subdir / df_name

    if df_path.exists():
        return pd.read_parquet(df_path)

    raise FileNotFoundError(f"DataFrame not found: {df_path}")


#%% PARAMETERS
OUT_DIR_RAW_DATA = CONFIG_TIME['out_dir_raw_data']
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'interactive_plots'
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

# Time cell thresholds + axis
TI_threshold = 0.2
shuff_TI_thresh = None
time_bin_size = 0.1
max_lap_duration_s = 6.0
frame_rate = 30
n_time_bins = int(max_lap_duration_s / time_bin_size)
xaxis_time = np.arange(n_time_bins) * time_bin_size

# Place cell thresholds + axis
SI_threshold = 0.2
shuff_SI_thresh = None
track_length = 180  # cm
pos_bin_size = 4    # cm
n_pos_bins = int(track_length / pos_bin_size)
xaxis_place = np.arange(n_pos_bins) * pos_bin_size

rec_lst = ['AC989-20250711-02', ]

#%% Main loop
if __name__ == '__main__':
    error_lst = []

    for rec in rec_lst:
        print(f"Processing {rec}...")

        try:
            # Reference image (same for time + place)
            mean_img = load_mean_image(rec, CONFIG_TIME)

            # Time cell dataframe + significance
            time_data = None
            try:
                df_time = load_dataframe(rec, CONFIG_TIME)
                df_time = select_significant_cells(df_time, CONFIG_TIME, TI_threshold, shuff_TI_thresh)
            except FileNotFoundError as e:
                print(f"  Warning (time cell df missing): {e}")
                df_time = None

            # Place cell dataframe + significance
            try:
                df_place = load_dataframe(rec, CONFIG_PLACE)
                df_place = select_significant_cells(df_place, CONFIG_PLACE, SI_threshold, shuff_SI_thresh)
            except FileNotFoundError as e:
                print(f"  Warning (place cell df missing): {e}")
                df_place = None

            # Shared ROI stats
            roi_stat, active_soma_indices = None, None
            try:
                roi_stat, active_soma_indices = load_roi_stat(rec, CONFIG_TIME)
                print(f"  Loaded {len(roi_stat)} ROIs from gcamp_stats.npy")
                print(f"  Active soma cells: {len(active_soma_indices)}")
            except FileNotFoundError as e:
                print(f"  Warning: {e}")

            # Build per-mode data bundles
            if df_time is not None:
                time_data = _build_mode_data(
                    df_time, xaxis_time, CONFIG_TIME, roi_stat, active_soma_indices,
                    TI_threshold, shuff_TI_thresh
                )

            place_data = None
            if df_place is not None:
                place_data = _build_mode_data(
                    df_place, xaxis_place, CONFIG_PLACE, roi_stat, active_soma_indices,
                    SI_threshold, shuff_SI_thresh
                )

            if time_data is None and place_data is None:
                print(f"  Skipping {rec}: no dataframes loaded.")
                error_lst.append(rec)
                continue

            save_path = OUT_DIR_FIG / f"{rec}_interactive_time_place_cell.html"
            generate_interactive_html(
                rec=rec,
                mean_img=mean_img,
                time_data=time_data,
                place_data=place_data,
                save_path=save_path,
            )

        except Exception as e:
            print(f"  Error processing {rec}: {e}")
            import traceback
            traceback.print_exc()
            error_lst.append(rec)

    if error_lst:
        print(f"\nErrors occurred for: {error_lst}")
    else:
        print("\nAll sessions processed successfully!")
