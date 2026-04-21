# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2026

Interactive response map with clickable grid ROIs.
- Left panel: Response map (clickable)
- Right panel: Mean trace of clicked ROI

Generates standalone HTML files with embedded JavaScript.

Supports three data types:
- 'Rdlight': Rdlight imaging data (grid-based, 16x16)
- 'Dbh_dlight': Dbh axon + dLight imaging (grid-based, 32x32)
- 'geco_dlight': GECO + dLight imaging (ROI-based, not grid)

@author: Jingyu Cao
"""
#%% IMPORTS AND FUNCS

import numpy as np
import pandas as pd
from pathlib import Path
import json

def get_config():
    """Get configuration for current data type."""
    return DATA_CONFIGS[DATA_TYPE]

def get_recording_list():
    """Get recording list for current data type."""
    if DATA_TYPE == 'Rdlight':
        from Rdlight_imaging.rec_lst import all_rec
        return all_rec
    elif DATA_TYPE == 'Dbh_dlight':
        from dlight_imaging.Dbh_dlight.recording_list import all_recs
        return all_recs
    elif DATA_TYPE == 'geco_dlight_learning':
        from dlight_imagaing_learning.geco_dlight.recording_list import all_recs
        return all_recs
    elif DATA_TYPE == 'Dbh_dlight_learning':
        from dlight_imagaing_learning.Dbh_dlight.recording_list import all_recs
        return all_recs
    else:
        raise ValueError(f"Unknown data type: {DATA_TYPE}")

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

def classify_rois(df_profile_ori, 
                  amp_shuff_thresh_up, amp_shuff_thresh_down,
                  effect_size_thresh):
    
    df_profile = df_profile_ori.copy()
    df_profile['shuffle_amps_thresh_up'] = df_profile['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_up))
    df_profile['shuffle_amps_thresh_down'] = df_profile['shuff_response_amplitude'].apply(lambda x: np.nanpercentile(x, amp_shuff_thresh_down))

    df_profile['Up'] = np.where(
                                # ~(df_profile['edge'])&
                                (df_profile['response_amplitude']>df_profile['shuffle_amps_thresh_up'])&
                                (df_profile['effect_size']>effect_size_thresh),
                                True, False)

    df_profile['Down'] = np.where(
                                # ~(df_profile['edge'])&
                                (df_profile['response_amplitude']<df_profile['shuffle_amps_thresh_down'])&
                                (df_profile['effect_size']< -effect_size_thresh),
                                True, False)

    df_profile.loc[df_profile['Up'], 'roi_type'] = 'Up'
    df_profile.loc[df_profile['Down'], 'roi_type'] = 'Down'
    df_profile.loc[(df_profile['Up']==0)&
                       (df_profile['Down']==0)
                       , 'roi_type'] = 'Stable'
    
    return df_profile

def load_mean_image(rec, config):
    """Load mean reference image based on data type configuration."""
    anm, date, ss = rec.split('-')
    suite2p_path = Path(config['suite2p_pattern'].format(anm=anm, date=date, ss=ss))

    if config['mean_img_key'] is not None:
        # Load from suite2p ops.npy
        ops_path = suite2p_path / 'ops.npy'
        if ops_path.exists():
            suite2p_ops = np.load(ops_path, allow_pickle=True).item()
            return suite2p_ops[config['mean_img_key']]
        else:
            # Try alternative paths
            alt_path = suite2p_path.parent / 'ops.npy'
            if alt_path.exists():
                suite2p_ops = np.load(alt_path, allow_pickle=True).item()
                return suite2p_ops.get(config['mean_img_key'], suite2p_ops.get('meanImg'))
    else:
        # For Dbh_dlight, load from RegOnly or alternative source
        reg_path = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\RegOnly\suite2p\plane0")
        ops_path = reg_path / 'ops.npy'
        if ops_path.exists():
            suite2p_ops = np.load(ops_path, allow_pickle=True).item()
            return suite2p_ops.get('meanImg_chan2', suite2p_ops.get('meanImg'))
        # Try nonrigid_reg path
        alt_path = Path(rf"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg\suite2p\plane0\ops.npy")
        if alt_path.exists():
            suite2p_ops = np.load(alt_path, allow_pickle=True).item()
            return suite2p_ops.get('meanImg_chan2', suite2p_ops.get('meanImg'))

    raise FileNotFoundError(f"Could not find mean image for {rec}")


def load_roi_stat(rec, config):
    """Load suite2p stat.npy for ROI coordinates."""
    anm, date, ss = rec.split('-')
    suite2p_path = Path(config['suite2p_pattern'].format(anm=anm, date=date, ss=ss))

    stat_path = suite2p_path / 'stat.npy'
    if stat_path.exists():
        return np.load(stat_path, allow_pickle=True)

    # Try parent directory
    alt_path = suite2p_path.parent / 'stat.npy'
    if alt_path.exists():
        return np.load(alt_path, allow_pickle=True)

    raise FileNotFoundError(f"Could not find stat.npy for {rec}")


def load_dataframe(rec, config, alignment):
    """Load processed dataframe based on data type configuration."""
    df_subdir = config['df_subdir']
    df_pattern = config['df_pattern']
    df_name = df_pattern.format(rec=rec, alignment=alignment)
    df_path = config['out_dir_raw_data'] / df_subdir / df_name

    if df_path.exists():
        return pd.read_parquet(df_path)

    # Try alternative patterns for geco_dlight types (no alignment in pattern)
    if DATA_TYPE in ['geco_dlight', 'geco_dlight_learning']:
        alt_pattern = '{rec}_profile_combined_geco_pre(-1, 0)_geco_post(0.5, 1.5).parquet'
        df_path = config['out_dir_raw_data'] / df_subdir / alt_pattern.format(rec=rec)
        if df_path.exists():
            return pd.read_parquet(df_path)

    raise FileNotFoundError(f"DataFrame not found: {df_path}")


#%% HTML PLOT FUNC
def generate_interactive_html(rec, df_valid, mean_img, xaxis, grid_size=16,
                              behaviour_data=None, save_path=None, config=None,
                              roi_stat=None):
    """
    Generate interactive HTML with response map and trace viewer.

    Parameters
    ----------
    rec : str
        Recording ID
    df_valid : pd.DataFrame
        DataFrame with valid ROIs
    mean_img : np.ndarray
        Reference image for FOV
    xaxis : np.ndarray
        Time axis for traces
    grid_size : int
        Size of each grid cell in pixels (None for ROI-based)
    behaviour_data : dict, optional
        Behaviour data containing speed_times_aligned and lick_times_aligned
    save_path : str or Path, optional
        Path to save HTML file
    config : dict, optional
        Data type configuration. If None, uses global config.
    roi_stat : np.ndarray, optional
        Suite2p stat array with ROI pixel coordinates (for ROI-based data)
    """
    if config is None:
        config = get_config()

    control_col = config['control_col']
    control_label = config['control_label']
    roi_type_col = config['roi_type_col']
    up_col = config['up_col']
    signal_label = config['signal_label']
    signal_color = config['signal_color']
    is_grid_based = grid_size is not None

    img_h, img_w = mean_img.shape

    if is_grid_based:
        n_grids_y = img_h // grid_size
        n_grids_x = img_w // grid_size
    else:
        # For ROI-based data, we'll display ROIs on the image
        n_grids_y = None
        n_grids_x = None

    # Store traces in a dictionary keyed by "gy_gx" (grid) or "roi_idx" (ROI-based)
    traces_dict = {}
    traces_ctrl_dict = {}

    if is_grid_based:
        # Create response map for grid-based data
        response_map = np.full((n_grids_y, n_grids_x), np.nan)
        roi_types = np.full((n_grids_y, n_grids_x), '', dtype=object)

        for idx, row in df_valid.iterrows():
            roi_id = row['roi_id']
            gy, gx = roi_id
            response_map[gy, gx] = row[response_key]
            roi_types[gy, gx] = row.get(roi_type_col, 'Stable')

            # Store trace if valid
            if trace_is_valid(row['mean_profile']):
                traces_dict[f"{gy}_{gx}"] = np.asarray(row['mean_profile']).tolist()
            if control_col in row and trace_is_valid(row[control_col]):
                traces_ctrl_dict[f"{gy}_{gx}"] = np.asarray(row[control_col]).tolist()
    else:
        # For ROI-based data (geco_dlight)
        # response_map will be a sparse representation
        response_map = None
        roi_types = {}
        roi_coords = {}  # Store centroid coordinates for each ROI

        for idx, row in df_valid.iterrows():
            roi_id = row['roi_id']
            key = str(roi_id)
            roi_types[key] = row.get(roi_type_col, 'Stable')

            # Store trace if valid
            if trace_is_valid(row['mean_profile']):
                traces_dict[key] = np.asarray(row['mean_profile']).tolist()
            if control_col in row and trace_is_valid(row[control_col]):
                traces_ctrl_dict[key] = np.asarray(row[control_col]).tolist()

            # For ROI-based, we need centroid coordinates (stored elsewhere or computed)
            # For now, use roi_id as index
            roi_coords[key] = roi_id

    # Calculate color scale
    if is_grid_based:
        valid_responses = response_map[~np.isnan(response_map)]
    else:
        # For ROI-based, get responses from dataframe
        valid_responses = df_valid[response_key].dropna().values
    if len(valid_responses) > 0:
        vmax = np.percentile(np.abs(valid_responses), 95)
    else:
        vmax = 1
    vmin = -vmax

    # Normalize reference image for display
    img_norm = (mean_img - np.percentile(mean_img, 1)) / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1))
    img_norm = np.clip(img_norm, 0, 1)

    # Count stats
    if is_grid_based:
        n_up = np.sum(roi_types == 'Up')
        n_valid = np.sum(~np.isnan(response_map))
    else:
        n_up = df_valid[up_col].sum() if up_col in df_valid.columns else 0
        n_valid = len(df_valid)
    pct_up = 100 * n_up / n_valid if n_valid > 0 else 0

    # Calculate mean Up ROI traces (normalized)
    up_mean_trace = None
    up_mean_trace_ctrl = None
    up_sem_trace = None
    up_sem_trace_ctrl = None
    if up_col in df_valid.columns and df_valid[up_col].sum() > 0:
        up_profiles = df_valid.loc[df_valid[up_col], 'mean_profile']
        up_profiles = np.stack([np.asarray(p) for p in up_profiles if trace_is_valid(p)])
        if len(up_profiles) > 0:
            up_profiles_norm = normalize_traces(up_profiles * 100)
            up_mean_trace = np.nanmean(up_profiles_norm, axis=0).tolist()
            up_sem_trace = (np.nanstd(up_profiles_norm, axis=0) / np.sqrt(up_profiles_norm.shape[0])).tolist()

        if control_col in df_valid.columns:
            up_profiles_ctrl = df_valid.loc[df_valid[up_col], control_col]
            up_profiles_ctrl = np.stack([np.asarray(p) for p in up_profiles_ctrl if trace_is_valid(p)])
            if len(up_profiles_ctrl) > 0:
                up_profiles_ctrl_norm = normalize_traces(up_profiles_ctrl * 100)
                up_mean_trace_ctrl = np.nanmean(up_profiles_ctrl_norm, axis=0).tolist()
                up_sem_trace_ctrl = (np.nanstd(up_profiles_ctrl_norm, axis=0) / np.sqrt(up_profiles_ctrl_norm.shape[0])).tolist()

    # Process behaviour data (speed and licks)
    # Note: speed_times_aligned and lick_times_aligned are already aligned to run onset (t=0)
    speed_mean = None
    speed_sem = None
    lick_rate = None
    beh_xaxis = None
    speed_xaxis = None
    lick_index = None
    if behaviour_data is not None:
        try:
            # Get lick index (selectivity)
            lick_selectivities = behaviour_data.get('lick_selectivities', None)
            if lick_selectivities is not None:
                lick_index = np.nanmean(lick_selectivities)

            # Speed processing - already aligned to run onset at 1000Hz, plot 0-4s
            total_ms = 4000  # 4s after run onset
            speed_trials = []

            for t_idx, tr_speed in enumerate(behaviour_data.get('speed_times_aligned', [])):
                if len(tr_speed) > 0:
                    speed_arr = np.vstack(tr_speed)
                    # Pad or truncate to total_ms
                    if len(speed_arr) >= total_ms:
                        speed_trials.append(speed_arr[:total_ms, 1])
                    else:
                        padded = np.full(total_ms, np.nan)
                        padded[:len(speed_arr)] = speed_arr[:, 1]
                        speed_trials.append(padded)

            if len(speed_trials) > 0:
                speed_trials = np.vstack(speed_trials)
                speed_mean = np.nanmean(speed_trials, axis=0).tolist()
                speed_sem = (np.nanstd(speed_trials, axis=0) / np.sqrt(speed_trials.shape[0])).tolist()
                speed_xaxis = (np.arange(total_ms) / 1000).tolist()  # 0 to 4s

            # Lick processing - compute lick rate histogram (0-4s)
            lick_times_all = []
            for licks in behaviour_data.get('lick_times_aligned', []):
                if isinstance(licks, (list, np.ndarray)) and len(licks) > 0:
                    lick_times_all.extend(np.asarray(licks).flatten() / 1000)  # convert to seconds

            if len(lick_times_all) > 0:
                # Compute histogram with 100ms bins from 0 to 4s
                bin_edges = np.arange(0, 4.0 + 0.1, 0.1)
                lick_hist, _ = np.histogram(lick_times_all, bins=bin_edges)
                n_trials = len(behaviour_data.get('speed_times_aligned', []))
                lick_rate = (lick_hist / n_trials / 0.1).tolist()  # licks per second per trial
                beh_xaxis = (bin_edges[:-1] + 0.05).tolist()  # bin centers

        except Exception as e:
            print(f"  Warning: Could not process behaviour data: {e}")

    # Convert data to JSON for JavaScript
    if is_grid_based:
        response_map_json = json.dumps(np.where(np.isnan(response_map), None, response_map).tolist())
        roi_types_json = json.dumps(roi_types.tolist())
    else:
        response_map_json = json.dumps(None)  # Not used for ROI-based
        roi_types_json = json.dumps(roi_types)  # dict for ROI-based
    traces_json = json.dumps(traces_dict)
    traces_ctrl_json = json.dumps(traces_ctrl_dict)
    xaxis_json = json.dumps(xaxis.tolist())
    img_json = json.dumps(img_norm.tolist())
    img_h_val = img_h
    img_w_val = img_w

    # Convert None to 'null' string for JavaScript
    grid_size_js = 'null' if grid_size is None else grid_size
    n_grids_y_js = 'null' if n_grids_y is None else n_grids_y
    n_grids_x_js = 'null' if n_grids_x is None else n_grids_x

    # For ROI-based data, prepare ROI coordinates from df_valid and roi_stat
    roi_data_for_js = {}
    if not is_grid_based:
        for idx, row in df_valid.iterrows():
            roi_id = row['roi_id']
            key = str(roi_id)

            # Get pixel coordinates from stat.npy
            xpix = []
            ypix = []
            if roi_stat is not None and roi_id < len(roi_stat):
                xpix = roi_stat[roi_id]['xpix'].tolist()
                ypix = roi_stat[roi_id]['ypix'].tolist()

            roi_data_for_js[key] = {
                'response': float(row[response_key]) if pd.notna(row[response_key]) else None,
                'roi_type': row.get(roi_type_col, 'Stable'),
                'is_up': bool(row.get(up_col, False)) if up_col in row else False,
                'xpix': xpix,
                'ypix': ypix
            }
    roi_data_json = json.dumps(roi_data_for_js)

    # Up ROI mean traces
    up_mean_json = json.dumps(up_mean_trace)
    up_mean_ctrl_json = json.dumps(up_mean_trace_ctrl)
    up_sem_json = json.dumps(up_sem_trace)
    up_sem_ctrl_json = json.dumps(up_sem_trace_ctrl)

    # Behaviour data
    speed_mean_json = json.dumps(speed_mean)
    speed_sem_json = json.dumps(speed_sem)
    speed_xaxis_json = json.dumps(speed_xaxis)
    lick_rate_json = json.dumps(lick_rate)
    beh_xaxis_json = json.dumps(beh_xaxis)
    lick_index_json = json.dumps(lick_index)

    # Labels for dynamic display
    signal_label_json = json.dumps(signal_label)
    control_label_json = json.dumps(control_label)
    is_grid_based_json = json.dumps(is_grid_based)

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{rec} - Interactive Response Map</title>
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
        #map-panel {{
            flex: 0 0 auto;
        }}
        #trace-panel {{
            flex: 0 0 auto;
        }}
        .trace-container {{
            display: flex;
            gap: 10px;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-box {{
            width: 15px;
            height: 15px;
        }}
    </style>
</head>
<body>
    <h1>{rec}</h1>
    <div class="stats">Up ROIs: {n_up}/{n_valid} ({pct_up:.1f}%) | Click on a grid cell to view trace</div>

    <div class="container">
        <div id="map-panel" class="panel">
            <div id="response-map"></div>
            <div class="legend">
                <div class="legend-item"><div class="legend-box" style="border: 2px solid orange; background: transparent;"></div> Up ROI</div>
                <div class="legend-item"><div class="legend-box" style="background: rgba(65,105,225,0.6);"></div> Negative response</div>
                <div class="legend-item"><div class="legend-box" style="background: rgba(220,20,60,0.6);"></div> Positive response</div>
            </div>
        </div>
        <div id="trace-panel" class="panel">
            <div class="trace-container">
                <div id="trace-plot"></div>
                <div id="trace-plot-norm"></div>
            </div>
            <div id="roi-info" class="info">Click on a grid cell in the response map to view its trace.</div>
            <div class="trace-container" style="margin-top: 15px;">
                <div id="up-mean-plot"></div>
                <div id="behaviour-plot"></div>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const responseMap = {response_map_json};
        const roiTypes = {roi_types_json};
        const traces = {traces_json};
        const tracesCtrl = {traces_ctrl_json};
        const xaxis = {xaxis_json};
        const refImg = {img_json};
        const gridSize = {grid_size_js};
        const nGridsY = {n_grids_y_js};
        const nGridsX = {n_grids_x_js};
        const roiData = {roi_data_json};
        const imgH = {img_h_val};
        const imgW = {img_w_val};
        const vmin = {vmin};
        const vmax = {vmax};

        // Up ROI mean traces
        const upMeanTrace = {up_mean_json};
        const upMeanTraceCtrl = {up_mean_ctrl_json};
        const upSemTrace = {up_sem_json};
        const upSemTraceCtrl = {up_sem_ctrl_json};

        // Behaviour data
        const speedMean = {speed_mean_json};
        const speedSem = {speed_sem_json};
        const speedXaxis = {speed_xaxis_json};
        const lickRate = {lick_rate_json};
        const behXaxis = {beh_xaxis_json};
        const lickIndex = {lick_index_json};
        const pctUp = {pct_up:.1f};

        // Labels for dynamic display
        const signalLabel = {signal_label_json};
        const controlLabel = {control_label_json};
        const isGridBased = {is_grid_based_json};

        // Normalize trace function
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

        // Create response map with FOV overlay
        function createResponseMap() {{
            // Reference image as background (grayscale)
            const refImgTrace = {{
                z: refImg,
                type: 'heatmap',
                colorscale: 'Greys',
                showscale: false,
                hoverinfo: 'skip',
                zmin: 0,
                zmax: 1
            }};

            if (isGridBased) {{
                // Grid-based visualization (Rdlight, Dbh_dlight)
                // Expand response map to pixel coordinates for overlay
                const responseMapExpanded = [];
                for (let py = 0; py < imgH; py++) {{
                    const row = [];
                    for (let px = 0; px < imgW; px++) {{
                        const gy = Math.floor(py / gridSize);
                        const gx = Math.floor(px / gridSize);
                        if (gy < nGridsY && gx < nGridsX) {{
                            row.push(responseMap[gy][gx]);
                        }} else {{
                            row.push(null);
                        }}
                    }}
                    responseMapExpanded.push(row);
                }}

                // Create custom colorscale (coolwarm with transparency for null)
                const colorscale = [
                    [0, 'rgba(59, 76, 192, 0.6)'],
                    [0.5, 'rgba(255, 255, 255, 0.1)'],
                    [1, 'rgba(180, 4, 38, 0.6)']
                ];

                // Response map overlay
                const heatmapOverlay = {{
                    z: responseMapExpanded,
                    type: 'heatmap',
                    colorscale: colorscale,
                    zmin: vmin,
                    zmax: vmax,
                    hoverongaps: false,
                    hovertemplate: 'Pixel (%{{y}}, %{{x}})<br>Response: %{{z:.4f}}<extra></extra>',
                    colorbar: {{
                        title: 'Effect Size',
                        titleside: 'right',
                        len: 0.8
                    }}
                }};

                // Find Up ROIs for highlighting (in pixel coordinates)
                const upShapes = [];
                for (let gy = 0; gy < nGridsY; gy++) {{
                    for (let gx = 0; gx < nGridsX; gx++) {{
                        if (roiTypes[gy][gx] === 'Up') {{
                            upShapes.push({{
                                type: 'rect',
                                x0: gx * gridSize - 0.5,
                                x1: (gx + 1) * gridSize - 0.5,
                                y0: gy * gridSize - 0.5,
                                y1: (gy + 1) * gridSize - 0.5,
                                line: {{
                                    color: 'orange',
                                    width: 2
                                }},
                                fillcolor: 'rgba(0,0,0,0)'
                            }});
                        }}
                    }}
                }}

                const layout = {{
                    title: 'Response Map on FOV (click to select)',
                    width: 700,
                    height: 700,
                    xaxis: {{
                        title: 'X (pixels)',
                        scaleanchor: 'y',
                        constrain: 'domain',
                        range: [-0.5, imgW - 0.5]
                    }},
                    yaxis: {{
                        title: 'Y (pixels)',
                        autorange: 'reversed',
                        range: [-0.5, imgH - 0.5]
                    }},
                    margin: {{ t: 50, b: 60, l: 60, r: 100 }},
                    shapes: upShapes
                }};

                Plotly.newPlot('response-map', [refImgTrace, heatmapOverlay], layout);

                // Add click handler for grid-based
                document.getElementById('response-map').on('plotly_click', function(data) {{
                    const point = data.points[0];
                    const px = Math.round(point.x);
                    const py = Math.round(point.y);
                    const gx = Math.floor(px / gridSize);
                    const gy = Math.floor(py / gridSize);
                    if (gy >= 0 && gy < nGridsY && gx >= 0 && gx < nGridsX) {{
                        updateTrace(gy, gx);
                        highlightSelectedROI(gy, gx);
                    }}
                }});
            }} else {{
                // ROI-based visualization (geco_dlight, geco_dlight_learning)
                // Create ROI mask overlay using actual pixel coordinates from stat.npy
                const roiIds = Object.keys(roiData);

                // Create response map from ROI pixel coordinates
                const roiMaskMap = [];
                const roiIdMap = [];  // Store which ROI each pixel belongs to
                for (let py = 0; py < imgH; py++) {{
                    roiMaskMap.push(new Array(imgW).fill(null));
                    roiIdMap.push(new Array(imgW).fill(null));
                }}

                // Fill in ROI pixels
                roiIds.forEach((roiId) => {{
                    const data = roiData[roiId];
                    const xpix = data.xpix;
                    const ypix = data.ypix;
                    const response = data.response;

                    for (let i = 0; i < xpix.length; i++) {{
                        const x = xpix[i];
                        const y = ypix[i];
                        if (y >= 0 && y < imgH && x >= 0 && x < imgW) {{
                            roiMaskMap[y][x] = response;
                            roiIdMap[y][x] = roiId;
                        }}
                    }}
                }});

                // Create custom colorscale (coolwarm with transparency for null)
                const colorscale = [
                    [0, 'rgba(59, 76, 192, 0.7)'],
                    [0.5, 'rgba(255, 255, 255, 0.2)'],
                    [1, 'rgba(180, 4, 38, 0.7)']
                ];

                // ROI mask overlay
                const roiOverlay = {{
                    z: roiMaskMap,
                    type: 'heatmap',
                    colorscale: colorscale,
                    zmin: vmin,
                    zmax: vmax,
                    hoverongaps: false,
                    hovertemplate: 'Pixel (%{{y}}, %{{x}})<br>Response: %{{z:.4f}}<extra></extra>',
                    colorbar: {{
                        title: 'Effect Size',
                        titleside: 'right',
                        len: 0.8
                    }}
                }};

                // Find Up ROI boundaries for highlighting
                const upShapes = [];
                roiIds.forEach((roiId) => {{
                    const data = roiData[roiId];
                    if (data.is_up && data.xpix.length > 0) {{
                        // Create a bounding box around the ROI
                        const minX = Math.min(...data.xpix);
                        const maxX = Math.max(...data.xpix);
                        const minY = Math.min(...data.ypix);
                        const maxY = Math.max(...data.ypix);
                        upShapes.push({{
                            type: 'rect',
                            x0: minX - 1,
                            x1: maxX + 1,
                            y0: minY - 1,
                            y1: maxY + 1,
                            line: {{
                                color: 'orange',
                                width: 2
                            }},
                            fillcolor: 'rgba(0,0,0,0)'
                        }});
                    }}
                }});

                const layout = {{
                    title: 'Response Map (ROI-based) - Click to select',
                    width: 700,
                    height: 700,
                    xaxis: {{
                        title: 'X (pixels)',
                        scaleanchor: 'y',
                        constrain: 'domain',
                        range: [-0.5, imgW - 0.5]
                    }},
                    yaxis: {{
                        title: 'Y (pixels)',
                        autorange: 'reversed',
                        range: [-0.5, imgH - 0.5]
                    }},
                    margin: {{ t: 50, b: 60, l: 60, r: 100 }},
                    shapes: upShapes
                }};

                // Store roiIdMap globally for click handling
                window.roiIdMap = roiIdMap;

                Plotly.newPlot('response-map', [refImgTrace, roiOverlay], layout);

                // Add click handler for ROI-based
                document.getElementById('response-map').on('plotly_click', function(data) {{
                    const point = data.points[0];
                    const px = Math.round(point.x);
                    const py = Math.round(point.y);
                    if (py >= 0 && py < imgH && px >= 0 && px < imgW) {{
                        const clickedRoiId = window.roiIdMap[py][px];
                        if (clickedRoiId !== null) {{
                            updateTraceROI(clickedRoiId);
                            highlightSelectedROI_ROI(clickedRoiId);
                        }}
                    }}
                }});
            }}
        }}

        // Highlight selected ROI
        let currentHighlight = null;
        function highlightSelectedROI(gy, gx) {{
            const mapDiv = document.getElementById('response-map');
            const layout = mapDiv.layout;

            // Remove previous highlight if exists
            let shapes = layout.shapes ? [...layout.shapes] : [];
            shapes = shapes.filter(s => s.line.color !== 'cyan');

            // Add new highlight
            shapes.push({{
                type: 'rect',
                x0: gx * gridSize - 0.5,
                x1: (gx + 1) * gridSize - 0.5,
                y0: gy * gridSize - 0.5,
                y1: (gy + 1) * gridSize - 0.5,
                line: {{
                    color: 'cyan',
                    width: 3
                }},
                fillcolor: 'rgba(0,255,255,0.2)'
            }});

            Plotly.relayout('response-map', {{ shapes: shapes }});
        }}

        // Update trace plot when ROI is clicked
        function updateTrace(gy, gx) {{
            const key = gy + '_' + gx;
            const trace = traces[key];
            const traceCtrl = tracesCtrl[key];
            const roiType = roiTypes[gy][gx];
            const response = responseMap[gy][gx];

            // Original trace plot
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
            if (traceCtrl) {{
                plotData.push({{
                    x: xaxis,
                    y: traceCtrl,
                    mode: 'lines',
                    name: controlLabel,
                    line: {{ color: 'gray', width: 2 }}
                }});
            }}

            // Normalized trace plot
            const plotDataNorm = [];
            const traceNorm = normalizeTrace(trace);
            const traceCtrlNorm = normalizeTrace(traceCtrl);
            if (traceNorm) {{
                plotDataNorm.push({{
                    x: xaxis,
                    y: traceNorm,
                    mode: 'lines',
                    name: signalLabel,
                    line: {{ color: 'green', width: 2 }},
                    showlegend: false
                }});
            }}
            if (traceCtrlNorm) {{
                plotDataNorm.push({{
                    x: xaxis,
                    y: traceCtrlNorm,
                    mode: 'lines',
                    name: controlLabel,
                    line: {{ color: 'gray', width: 2 }},
                    showlegend: false
                }});
            }}

            // Vertical line at t=0
            const shapes = [{{
                type: 'line',
                x0: 0, x1: 0,
                y0: 0, y1: 1,
                yref: 'paper',
                line: {{ color: 'black', width: 1, dash: 'dash' }}
            }}];

            const layoutOrig = {{
                title: `ROI (${{gy}}, ${{gx}}) - ${{roiType}} - Original`,
                width: 450,
                height: 350,
                xaxis: {{
                    title: 'Time from run onset (s)',
                    range: [-2, 4],
                    zeroline: false
                }},
                yaxis: {{
                    title: 'dF/F',
                    zeroline: true
                }},
                shapes: shapes,
                showlegend: true,
                legend: {{ x: 1.02, y: 1, xanchor: 'left' }},
                margin: {{ t: 50, b: 50, l: 60, r: 80 }}
            }};

            const layoutNorm = {{
                title: `Normalized`,
                width: 350,
                height: 350,
                xaxis: {{
                    title: 'Time from run onset (s)',
                    range: [-2, 4],
                    zeroline: false
                }},
                yaxis: {{
                    title: 'Normalized',
                    zeroline: true,
                    range: [-0.1, 1.1]
                }},
                shapes: shapes,
                showlegend: false,
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};

            Plotly.newPlot('trace-plot', plotData.length > 0 ? plotData : [], layoutOrig);
            Plotly.newPlot('trace-plot-norm', plotDataNorm.length > 0 ? plotDataNorm : [], layoutNorm);

            // Update info
            const responseStr = response !== null ? response.toFixed(4) : 'N/A';
            const hasTrace = trace ? 'Yes' : 'No';
            document.getElementById('roi-info').innerHTML =
                `<strong>ROI (${{gy}}, ${{gx}})</strong> | ` +
                `Type: <span style="color: ${{roiType === 'Up' ? 'green' : roiType === 'Down' ? 'red' : 'blue'}}">${{roiType}}</span> | ` +
                `Effect Size: ${{responseStr}} | ` +
                `Has Trace: ${{hasTrace}}`;
        }}

        // Update trace plot for ROI-based data
        function updateTraceROI(roiId) {{
            const key = roiId.toString();
            const trace = traces[key];
            const traceCtrl = tracesCtrl[key];
            const data = roiData[key];
            const roiType = data ? data.roi_type : 'Unknown';
            const response = data ? data.response : null;

            // Original trace plot
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
            if (traceCtrl) {{
                plotData.push({{
                    x: xaxis,
                    y: traceCtrl,
                    mode: 'lines',
                    name: controlLabel,
                    line: {{ color: 'gray', width: 2 }}
                }});
            }}

            // Normalized trace plot
            const plotDataNorm = [];
            const traceNorm = normalizeTrace(trace);
            const traceCtrlNorm = normalizeTrace(traceCtrl);
            if (traceNorm) {{
                plotDataNorm.push({{
                    x: xaxis,
                    y: traceNorm,
                    mode: 'lines',
                    name: signalLabel,
                    line: {{ color: 'green', width: 2 }},
                    showlegend: false
                }});
            }}
            if (traceCtrlNorm) {{
                plotDataNorm.push({{
                    x: xaxis,
                    y: traceCtrlNorm,
                    mode: 'lines',
                    name: controlLabel,
                    line: {{ color: 'gray', width: 2 }},
                    showlegend: false
                }});
            }}

            // Vertical line at t=0
            const shapes = [{{
                type: 'line',
                x0: 0, x1: 0,
                y0: 0, y1: 1,
                yref: 'paper',
                line: {{ color: 'black', width: 1, dash: 'dash' }}
            }}];

            const layoutOrig = {{
                title: `ROI ${{roiId}} - ${{roiType}} - Original`,
                width: 450,
                height: 350,
                xaxis: {{
                    title: 'Time from run onset (s)',
                    range: [-2, 4],
                    zeroline: false
                }},
                yaxis: {{
                    title: 'dF/F',
                    zeroline: true
                }},
                shapes: shapes,
                showlegend: true,
                legend: {{ x: 1.02, y: 1, xanchor: 'left' }},
                margin: {{ t: 50, b: 50, l: 60, r: 80 }}
            }};

            const layoutNorm = {{
                title: `Normalized`,
                width: 350,
                height: 350,
                xaxis: {{
                    title: 'Time from run onset (s)',
                    range: [-2, 4],
                    zeroline: false
                }},
                yaxis: {{
                    title: 'Normalized',
                    zeroline: true,
                    range: [-0.1, 1.1]
                }},
                shapes: shapes,
                showlegend: false,
                margin: {{ t: 50, b: 50, l: 50, r: 20 }}
            }};

            Plotly.newPlot('trace-plot', plotData.length > 0 ? plotData : [], layoutOrig);
            Plotly.newPlot('trace-plot-norm', plotDataNorm.length > 0 ? plotDataNorm : [], layoutNorm);

            // Update info
            const responseStr = response !== null ? response.toFixed(4) : 'N/A';
            const hasTrace = trace ? 'Yes' : 'No';
            document.getElementById('roi-info').innerHTML =
                `<strong>ROI ${{roiId}}</strong> | ` +
                `Type: <span style="color: ${{roiType === 'dlightUp' ? 'green' : roiType === 'dlightDown' ? 'red' : 'blue'}}">${{roiType}}</span> | ` +
                `Effect Size: ${{responseStr}} | ` +
                `Has Trace: ${{hasTrace}}`;
        }}

        // Highlight selected ROI for ROI-based data
        let selectedRoiId = null;
        function highlightSelectedROI_ROI(roiId) {{
            const mapDiv = document.getElementById('response-map');
            const layout = mapDiv.layout;
            const data = roiData[roiId];

            // Remove previous highlight if exists
            let shapes = layout.shapes ? [...layout.shapes] : [];
            shapes = shapes.filter(s => s.line.color !== 'cyan');

            // Add new highlight around selected ROI
            if (data && data.xpix.length > 0) {{
                const minX = Math.min(...data.xpix);
                const maxX = Math.max(...data.xpix);
                const minY = Math.min(...data.ypix);
                const maxY = Math.max(...data.ypix);
                shapes.push({{
                    type: 'rect',
                    x0: minX - 2,
                    x1: maxX + 2,
                    y0: minY - 2,
                    y1: maxY + 2,
                    line: {{
                        color: 'cyan',
                        width: 3
                    }},
                    fillcolor: 'rgba(0,255,255,0.15)'
                }});
            }}

            Plotly.relayout('response-map', {{ shapes: shapes }});
            selectedRoiId = roiId;
        }}

        // Initialize
        createResponseMap();

        // Initialize empty trace plots
        const emptyLayout = {{
            title: 'Select an ROI',
            width: 450,
            height: 350,
            xaxis: {{ title: 'Time (s)', range: [-2, 4] }},
            yaxis: {{ title: 'dF/F' }},
            margin: {{ t: 50, b: 50, l: 60, r: 80 }}
        }};
        const emptyLayoutNorm = {{
            title: 'Normalized',
            width: 350,
            height: 350,
            xaxis: {{ title: 'Time (s)', range: [-2, 4] }},
            yaxis: {{ title: 'Normalized', range: [-0.1, 1.1] }},
            margin: {{ t: 50, b: 50, l: 50, r: 20 }}
        }};
        Plotly.newPlot('trace-plot', [], emptyLayout);
        Plotly.newPlot('trace-plot-norm', [], emptyLayoutNorm);

        // Plot Up ROI mean trace
        function plotUpMeanTrace() {{
            const plotData = [];
            const shapes = [{{
                type: 'line',
                x0: 0, x1: 0,
                y0: 0, y1: 1,
                yref: 'paper',
                line: {{ color: 'black', width: 1, dash: 'dash' }}
            }}];

            if (upMeanTrace && upMeanTrace.length > 0) {{
                // Mean trace
                plotData.push({{
                    x: xaxis,
                    y: upMeanTrace,
                    mode: 'lines',
                    name: signalLabel,
                    line: {{ color: 'green', width: 2 }}
                }});
                // SEM band
                if (upSemTrace) {{
                    const upperBound = upMeanTrace.map((v, i) => v + upSemTrace[i]);
                    const lowerBound = upMeanTrace.map((v, i) => v - upSemTrace[i]);
                    plotData.push({{
                        x: xaxis.concat([...xaxis].reverse()),
                        y: upperBound.concat([...lowerBound].reverse()),
                        fill: 'toself',
                        fillcolor: 'rgba(0, 128, 0, 0.2)',
                        line: {{ color: 'transparent' }},
                        showlegend: false,
                        hoverinfo: 'skip'
                    }});
                }}
            }}

            if (upMeanTraceCtrl && upMeanTraceCtrl.length > 0) {{
                plotData.push({{
                    x: xaxis,
                    y: upMeanTraceCtrl,
                    mode: 'lines',
                    name: controlLabel,
                    line: {{ color: 'gray', width: 2 }}
                }});
                if (upSemTraceCtrl) {{
                    const upperBound = upMeanTraceCtrl.map((v, i) => v + upSemTraceCtrl[i]);
                    const lowerBound = upMeanTraceCtrl.map((v, i) => v - upSemTraceCtrl[i]);
                    plotData.push({{
                        x: xaxis.concat([...xaxis].reverse()),
                        y: upperBound.concat([...lowerBound].reverse()),
                        fill: 'toself',
                        fillcolor: 'rgba(128, 128, 128, 0.2)',
                        line: {{ color: 'transparent' }},
                        showlegend: false,
                        hoverinfo: 'skip'
                    }});
                }}
            }}

            const layout = {{
                title: `All Up ROIs Mean (Normalized) - ${{pctUp.toFixed(1)}}% DA-Up`,
                width: 400,
                height: 300,
                xaxis: {{
                    title: 'Time from run onset (s)',
                    range: [-1, 4],
                    zeroline: false
                }},
                yaxis: {{
                    title: 'Norm. dF/F',
                    zeroline: true
                }},
                shapes: shapes,
                showlegend: true,
                legend: {{ x: 1.02, y: 1, xanchor: 'left' }},
                margin: {{ t: 50, b: 50, l: 60, r: 80 }}
            }};

            Plotly.newPlot('up-mean-plot', plotData, layout);
        }}

        // Plot behaviour (speed and licks)
        function plotBehaviour() {{
            const plotData = [];
            const shapes = [{{
                type: 'line',
                x0: 0, x1: 0,
                y0: 0, y1: 1,
                yref: 'paper',
                line: {{ color: 'black', width: 1, dash: 'dash' }}
            }}];

            if (speedMean && speedMean.length > 0 && speedXaxis && speedXaxis.length > 0) {{
                plotData.push({{
                    x: speedXaxis,
                    y: speedMean,
                    mode: 'lines',
                    name: 'Speed',
                    line: {{ color: 'blue', width: 2 }},
                    yaxis: 'y'
                }});

                if (speedSem) {{
                    const upperBound = speedMean.map((v, i) => v + speedSem[i]);
                    const lowerBound = speedMean.map((v, i) => v - speedSem[i]);
                    plotData.push({{
                        x: speedXaxis.concat([...speedXaxis].reverse()),
                        y: upperBound.concat([...lowerBound].reverse()),
                        fill: 'toself',
                        fillcolor: 'rgba(0, 0, 255, 0.15)',
                        line: {{ color: 'transparent' }},
                        showlegend: false,
                        hoverinfo: 'skip',
                        yaxis: 'y'
                    }});
                }}
            }}

            if (lickRate && lickRate.length > 0 && behXaxis && behXaxis.length > 0) {{
                plotData.push({{
                    x: behXaxis,
                    y: lickRate,
                    mode: 'lines',
                    name: 'Lick rate',
                    line: {{ color: 'red', width: 2 }},
                    yaxis: 'y2'
                }});
            }}

            const lickIdxStr = lickIndex !== null ? lickIndex.toFixed(2) : 'N/A';
            const layout = {{
                title: `Behaviour - Lick Index: ${{lickIdxStr}}`,
                width: 400,
                height: 300,
                xaxis: {{
                    title: 'Time from run onset (s)',
                    range: [0, 4],
                    zeroline: false
                }},
                yaxis: {{
                    title: 'Speed (cm/s)',
                    titlefont: {{ color: 'blue' }},
                    tickfont: {{ color: 'blue' }},
                    zeroline: true,
                    side: 'left'
                }},
                yaxis2: {{
                    title: 'Lick rate (Hz)',
                    titlefont: {{ color: 'red' }},
                    tickfont: {{ color: 'red' }},
                    overlaying: 'y',
                    side: 'right',
                    zeroline: false
                }},
                shapes: shapes,
                showlegend: true,
                legend: {{ x: 0.5, y: 1.15, xanchor: 'center', orientation: 'h' }},
                margin: {{ t: 70, b: 50, l: 60, r: 60 }}
            }};

            Plotly.newPlot('behaviour-plot', plotData, layout);
        }}

        // Initialize summary plots
        plotUpMeanTrace();
        plotBehaviour();
    </script>
</body>
</html>
'''

    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Saved: {save_path}")

    return html_content


#%% DATA TYPE CONFIGURATION

DATA_CONFIGS = {
    'Rdlight': {
        'out_dir_raw_data': Path(r"Z:\Jingyu\rdlight_raw_data"),
        'grid_size': 16,
        'n_grids': 32,  # 512/16 = 32 grids per dimension
        'control_col': 'mean_profile_ctrl',
        'control_label': 'Control',
        'roi_type_col': 'roi_type',
        'up_col': 'Up',
        'has_edge_col': True,
        'df_pattern': '{rec}_profile_combined_dilation=0_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95_{alignment}.parquet',
        'df_subdir': 'processed_dataframe',
        'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg\suite2p\plane0",
        'mean_img_key': 'meanImg_chan2',
        'signal_label': 'Rdlight',
        'signal_color': 'green',
    },
    'Dbh_dlight': {
        'out_dir_raw_data': Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight"),
        'grid_size': 16,  # pixel size of each grid tile
        'n_grids': 32,    # 32x32 grid ROIs
        'control_col': 'mean_profile_red',
        'control_label': 'Axon (tdTom)',
        'roi_type_col': 'roi_type',
        'up_col': 'Up',
        'has_edge_col': True,
        'df_pattern': '{rec}_profile_combined_dilation=0_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95_{alignment}.parquet',
        'df_subdir': 'processed_dataframe_grid_free_dilation',
        'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\RegOnly",
        'mean_img_key': None,  # Need to load separately
        'signal_label': 'dLight',
        'signal_color': 'green',
    },
    'geco_dlight': {
        'out_dir_raw_data': Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight"),
        'grid_size': None,  # ROI-based, not grid
        'n_grids': None,
        'control_col': 'mean_profile_geco',
        'control_label': 'GECO',
        'roi_type_col': 'dlight_type',  # different column name!
        'up_col': 'dlightUp',
        'has_edge_col': False,  # soma-based ROIs don't have edge
        'df_pattern': '{rec}_profile_combined_geco_pre(-1, 0)_geco_post(0.5, 1.5).parquet',
        'df_subdir': 'processed_dataframe',
        'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\nonrigid_reg_geco\suite2p_anat_detec\plane0",
        'mean_img_key': 'meanImg',  # GECO channel
        'signal_label': 'dLight',
        'signal_color': 'green',
    },
    
    'Dbh_dlight_learning': {
        'out_dir_raw_data': Path(r"Z:\Jingyu\dlight_learning\Dbh_dlight"),
        'grid_size': 16,  # pixel size of each grid tile
        'n_grids': 32,    # 32x32 grid ROIs
        'control_col': 'mean_profile_red',
        'control_label': 'Axon (tdTom)',
        'roi_type_col': 'roi_type',
        'up_col': 'Up',
        'has_edge_col': True,
        # 'df_pattern': '{rec}_profile_combined_dilation=0_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95_{alignment}.parquet',
        'df_pattern': '{rec}_profile_combined_dilation=0_pre(-1, 0)_post(0, 1)_ES=0.05_shuff95_test.parquet',
        'df_subdir': 'processed_dataframe_grid_free_dilation',
        'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\RegOnly",
        'mean_img_key': None,  # Need to load separately
        'signal_label': 'dLight',
        'signal_color': 'green',
    },
    
    'geco_dlight_learning': {
        'out_dir_raw_data': Path(r"Z:\Jingyu\dlight_learning\geco_dlight"),
        'grid_size': None,  # ROI-based, not grid
        'n_grids': None,
        'control_col': 'mean_profile_geco',
        'control_label': 'GECO',
        'roi_type_col': 'dlight_type',  # different column name!
        'up_col': 'dlightUp',
        'has_edge_col': False,  # soma-based ROIs don't have edge
        'df_pattern': '{rec}_profile_combined_geco_pre(-1, 0)_geco_post(0.5, 1.5).parquet',
        'df_subdir': 'processed_dataframe',
        'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\dLight+GECO\GECO",
        'mean_img_key': 'meanImg',  # GECO channel
        'signal_label': 'dLight',
        'signal_color': 'green',
    },
}

#%% PARAMS
#%% PARAMETERS
# Change this to switch between data types: 'Rdlight', 'Dbh_dlight', 'geco_dlight'
# DATA_TYPE = 'Rdlight'
# DATA_TYPE = 'geco_dlight_learning'
DATA_TYPE = 'Dbh_dlight_learning'

# Get configuration for the selected data type
config = get_config()
OUT_DIR_RAW_DATA = config['out_dir_raw_data']
GRID_SIZE = config['grid_size']

# Output directory based on data type
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'TEST_PLOTS' / 'interactive_maps_learning'
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

alignment = 'run'  # or 'rew' depending on alignment
response_key = 'effect_size'

# Time axis parameters
bef, aft = 2, 4
fps = 30
n_frames = fps * (bef + aft)
xaxis = np.arange(n_frames) / fps - bef

# Get recording list for the selected data type
# rec_lst = get_recording_list()

# Classification thresholds
amp_shuff_thresh_up = 99
amp_shuff_thresh_down = 1
effect_size_thresh = 0.05

if DATA_TYPE == 'Dbh_dlight_learning':
    p_session_info = OUT_DIR_RAW_DATA / 'all_animals_learning_classification.parquet'
    df_all = pd.read_parquet(p_session_info)
    rec_lst = df_all.loc[(df_all['days_from_learned']<=0)&
                         (df_all['animal']=='AC964'), 'rec'].to_list()
elif DATA_TYPE == 'geco_dlight_learning':
    p_session_info = OUT_DIR_RAW_DATA / 'all_animals_learning_classification.parquet'
    df_all = pd.read_parquet(p_session_info)
    rec_lst = df_all.loc[(df_all['days_from_learned']<=2)&
                         (df_all['animal']=='AC953'), 'rec'].to_list()
    
#%% Main loop
if __name__ == '__main__':
    error_lst = []

    for rec in rec_lst:
        print(f"Processing {rec} ({DATA_TYPE})...")

        try:
            # Load reference image
            mean_img = load_mean_image(rec, config)

            # Load dataframe
            df_data = load_dataframe(rec, config, alignment)

            # Filter for valid ROIs
            if config['has_edge_col'] and 'edge' in df_data.columns:
                df_valid = df_data[~df_data['edge']].copy()
            else:
                df_valid = df_data.copy()

            # Classify ROIs (only for grid-based data types that have classify_rois)
            if classify_rois is not None:
                df_valid = classify_rois(df_valid,
                                         amp_shuff_thresh_up, amp_shuff_thresh_down,
                                         effect_size_thresh)

            # Load ROI stat for ROI-based data types
            roi_stat = None
            if GRID_SIZE is None:  # ROI-based
                try:
                    roi_stat = load_roi_stat(rec, config)
                    print(f"  Loaded {len(roi_stat)} ROIs from stat.npy")
                except FileNotFoundError as e:
                    print(f"  Warning: {e}")

            # Load behaviour data
            beh_path = OUT_DIR_RAW_DATA / 'behaviour_profile' / f'{rec}.pkl'
            behaviour_data = None
            if beh_path.exists():
                behaviour_data = pd.read_pickle(beh_path)
            else:
                print(f"  Behaviour data not found: {beh_path}")

            # Generate interactive HTML
            save_path = OUT_DIR_FIG / f"{rec}_interactive_map.html"
            generate_interactive_html(
                rec=rec,
                df_valid=df_valid,
                mean_img=mean_img,
                xaxis=xaxis,
                grid_size=GRID_SIZE,
                behaviour_data=behaviour_data,
                save_path=save_path,
                config=config,
                roi_stat=roi_stat
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
