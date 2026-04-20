# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2026

Interactive response map with clickable ROIs for time cell/place cell analysis.
- Left panel: Response map (clickable)
- Right panel: Time/place field trace of clicked ROI

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
CONFIG = {
    'out_dir_raw_data': Path(r"Z:\Jingyu\GCaMP_drug_infusion"),
    'df_subdir': 'time_cell_dataframe',
    'df_pattern': '{rec}_time_cell_dataframe.parquet',
    'suite2p_pattern': r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\suite2p\plane0",
    'gcamp_stats_pattern': r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{anm}-{date}\gcamp_stats.npy",
    'soma_class_pattern': r"Z:\Jingyu\LC_HPC_manuscript\raw_data\drug_infusion\raw_signals\{anm}-{date}\soma_class.npz",
    'mean_img_key': 'meanImg',

    # Column mappings for time cell / place cell dataframes
    'cell_id_col': 'cell_id',
    'field_map_col': 'time_field_map_norm',  # or 'place_field_map_norm'
    'info_bits_col': 'temporal_information_bits',  # or 'spatial_information_bits'
    'shuffled_info_col': 'shuffled_TI',  # or 'shuffled_SI'
    'peak_position_col': 'time_field_position_s',  # or 'place_field_position_cm'
    'peak_amplitude_col': 'time_field_peak_amplitude',  # or 'place_field_peak_amplitude'

    # Display settings
    'signal_label': 'Time Field',
    'x_label': 'Time (s)',
    'info_label': 'Temporal Info (bits)',
}

def get_config():
    """Get configuration."""
    return CONFIG

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


def select_time_cell(df, TI_threshold, shuff_TI_thresh):
    """
    Select time cells based on temporal information threshold and shuffle significance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cell data
    TI_threshold : float
        Minimum temporal information bits threshold
    shuff_TI_thresh : float
        Percentile threshold for shuffle test (e.g., 99)

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'is_significant' and 'cell_type' columns
    """
    df = df.copy()
    df['shuffle_TI_thresh'] = df['shuffled_TI'].apply(
        lambda x: np.nanpercentile(x, shuff_TI_thresh) if x is not None else np.nan)
    df['is_time_cell'] = ((df['temporal_information_bits'] > TI_threshold) &
                          (df['temporal_information_bits'] > df['shuffle_TI_thresh']))
    # Add columns for compatibility with visualization
    df['is_significant'] = df['is_time_cell']
    df['cell_type'] = np.where(df['is_significant'], 'Time Cell', 'Non-time Cell')
    return df


def generate_interactive_html(rec, df_valid, mean_img, xaxis,
                              save_path=None, config=None, roi_stat=None,
                              active_soma_indices=None):
    """
    Generate interactive HTML with response map and trace viewer.

    Parameters
    ----------
    rec : str
        Recording ID
    df_valid : pd.DataFrame
        DataFrame with valid ROIs (time cells or place cells)
    mean_img : np.ndarray
        Reference image for FOV
    active_soma_indices : np.ndarray
        Mapping from cell_id to original ROI index in roi_stat
    xaxis : np.ndarray
        X-axis for field map (normalized time or position)
    save_path : str or Path, optional
        Path to save HTML file
    config : dict, optional
        Configuration. If None, uses global config.
    roi_stat : np.ndarray, optional
        Suite2p stat array with ROI pixel coordinates
    """
    if config is None:
        config = get_config()

    cell_id_col = config['cell_id_col']
    field_map_col = config['field_map_col']
    info_bits_col = config['info_bits_col']
    signal_label = config['signal_label']
    x_label = config['x_label']
    info_label = config['info_label']

    img_h, img_w = mean_img.shape

    # Store field maps in a dictionary keyed by cell_id
    traces_dict = {}
    cell_types = {}
    cell_info = {}

    for idx, row in df_valid.iterrows():
        cell_id = row[cell_id_col]
        key = str(cell_id)
        cell_types[key] = row.get('cell_type', 'Unknown')

        # Store field map if valid
        if field_map_col in row and trace_is_valid(row[field_map_col]):
            traces_dict[key] = np.asarray(row[field_map_col]).tolist()

        # Store additional info
        cell_info[key] = {
            'info_bits': float(row[info_bits_col]) if pd.notna(row.get(info_bits_col)) else None,
            'peak_position': float(row.get(config['peak_position_col'], np.nan)) if pd.notna(row.get(config['peak_position_col'])) else None,
            'peak_amplitude': float(row.get(config['peak_amplitude_col'], np.nan)) if pd.notna(row.get(config['peak_amplitude_col'])) else None,
        }

    # Calculate color scale based on information bits
    valid_info = df_valid[info_bits_col].dropna().values
    if len(valid_info) > 0:
        vmax = np.percentile(valid_info, 95)
        vmin = 0  # Info bits are always positive
    else:
        vmax = 1
        vmin = 0

    # Normalize reference image for display
    img_norm = (mean_img - np.percentile(mean_img, 1)) / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1))
    img_norm = np.clip(img_norm, 0, 1)

    # Count stats
    n_sig = df_valid['is_significant'].sum() if 'is_significant' in df_valid.columns else 0
    n_valid = len(df_valid)
    pct_sig = 100 * n_sig / n_valid if n_valid > 0 else 0

    # Prepare time cell sequence heatmap data (sorted by peak time)
    df_time_cells = df_valid[df_valid['is_significant'] == True].copy()
    sequence_heatmap = None
    sequence_cell_ids = []
    sequence_peak_times = []

    if len(df_time_cells) > 0:
        # Sort by time field position
        df_time_cells_sorted = df_time_cells.sort_values(config['peak_position_col'])

        # Stack the time field maps
        sequence_cell_ids = df_time_cells_sorted[cell_id_col].tolist()
        sequence_peak_times = df_time_cells_sorted[config['peak_position_col']].tolist()

        # Build the heatmap array
        heatmap_rows = []
        for _, row in df_time_cells_sorted.iterrows():
            field_map = row[field_map_col]
            if trace_is_valid(field_map):
                heatmap_rows.append(np.asarray(field_map).tolist())
            else:
                # Fill with zeros if invalid
                heatmap_rows.append([0] * len(xaxis))

        if len(heatmap_rows) > 0:
            sequence_heatmap = heatmap_rows

    # Convert data to JSON for JavaScript
    cell_types_json = json.dumps(cell_types)
    traces_json = json.dumps(traces_dict)
    cell_info_json = json.dumps(cell_info)
    xaxis_json = json.dumps(xaxis.tolist())
    img_json = json.dumps(img_norm.tolist())
    img_h_val = img_h
    img_w_val = img_w

    # Sequence heatmap data
    sequence_heatmap_json = json.dumps(sequence_heatmap) if sequence_heatmap else 'null'
    sequence_cell_ids_json = json.dumps(sequence_cell_ids)
    sequence_peak_times_json = json.dumps(sequence_peak_times)

    # Prepare ROI coordinates from df_valid and roi_stat
    # cell_id indexes into active_soma_indices to get original ROI index in gcamp_stats
    roi_data_for_js = {}
    n_with_coords = 0

    for _, row in df_valid.iterrows():
        cell_id = int(row[cell_id_col])
        key = str(cell_id)

        # Get pixel coordinates from gcamp_stats.npy
        # Use active_soma_indices to map cell_id to original ROI index
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

        roi_data_for_js[key] = {
            'info_bits': float(row[info_bits_col]) if pd.notna(row[info_bits_col]) else None,
            'cell_type': row.get('cell_type', 'Unknown'),
            'is_significant': bool(row.get('is_significant', False)),
            'xpix': xpix,
            'ypix': ypix
        }

    print(f"  ROIs with pixel coordinates: {n_with_coords}/{len(df_valid)}")
    if roi_stat is not None:
        print(f"  gcamp_stats.npy has {len(roi_stat)} entries")
    if active_soma_indices is not None:
        print(f"  Active soma indices: {len(active_soma_indices)} cells")

    roi_data_json = json.dumps(roi_data_for_js)

    # Labels for dynamic display
    signal_label_json = json.dumps(signal_label)
    x_label_json = json.dumps(x_label)
    info_label_json = json.dumps(info_label)

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
    <div class="stats">Significant Cells: {n_sig}/{n_valid} ({pct_sig:.1f}%) | Click on an ROI to view field map</div>

    <div class="container">
        <div id="map-panel" class="panel">
            <div id="response-map"></div>
            <div class="legend">
                <div class="legend-item"><div class="legend-box" style="background: #440154;"></div> Low Info</div>
                <div class="legend-item"><div class="legend-box" style="background: #21918c;"></div> Mid Info</div>
                <div class="legend-item"><div class="legend-box" style="background: #fde725;"></div> High Info</div>
            </div>
        </div>
        <div id="trace-panel" class="panel">
            <div class="trace-container">
                <div id="trace-plot"></div>
            </div>
            <div id="roi-info" class="info">Click on an ROI in the map to view its field map.</div>
        </div>
        <div id="sequence-panel" class="panel">
            <div id="sequence-heatmap"></div>
        </div>
    </div>

    <script>
        // Data from Python
        const cellTypes = {cell_types_json};
        const traces = {traces_json};
        const cellInfo = {cell_info_json};
        const xaxis = {xaxis_json};
        const refImg = {img_json};
        const roiData = {roi_data_json};
        const imgH = {img_h_val};
        const imgW = {img_w_val};
        const vmin = {vmin};
        const vmax = {vmax};

        // Labels for dynamic display
        const signalLabel = {signal_label_json};
        const xLabel = {x_label_json};
        const infoLabel = {info_label_json};

        // Sequence heatmap data
        const sequenceHeatmap = {sequence_heatmap_json};
        const sequenceCellIds = {sequence_cell_ids_json};
        const sequencePeakTimes = {sequence_peak_times_json};

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

            // ROI-based visualization
            const roiIds = Object.keys(roiData);

            // Create response map from ROI pixel coordinates
            const roiMaskMap = [];
            const roiIdMap = [];
            for (let py = 0; py < imgH; py++) {{
                roiMaskMap.push(new Array(imgW).fill(null));
                roiIdMap.push(new Array(imgW).fill(null));
            }}

            // Fill in ROI pixels
            roiIds.forEach((roiId) => {{
                const data = roiData[roiId];
                const xpix = data.xpix;
                const ypix = data.ypix;
                // Use 0 as fallback if info_bits is null to ensure ROI is still visible
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

            // Use Viridis sequential colorscale for non-negative info bits
            const colorscale = 'Viridis';

            // ROI mask overlay
            const roiOverlay = {{
                z: roiMaskMap,
                type: 'heatmap',
                colorscale: colorscale,
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
                margin: {{ t: 50, b: 60, l: 60, r: 100 }}
            }};

            window.roiIdMap = roiIdMap;

            Plotly.newPlot('response-map', [refImgTrace, roiOverlay], layout);

            // Add click handler
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

        // Update trace plot when ROI is clicked
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

            // Add vertical line at peak position if available
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
                xaxis: {{
                    title: xLabel,
                    zeroline: false
                }},
                yaxis: {{
                    title: 'dF/F',
                    zeroline: true
                }},
                shapes: shapes,
                showlegend: false,
                margin: {{ t: 50, b: 50, l: 60, r: 30 }}
            }};

            Plotly.newPlot('trace-plot', plotData.length > 0 ? plotData : [], layout);

            // Update info
            const infoBitsStr = infoBits !== null ? infoBits.toFixed(4) : 'N/A';
            const peakPosStr = info && info.peak_position !== null ? info.peak_position.toFixed(2) : 'N/A';
            const peakAmpStr = info && info.peak_amplitude !== null ? info.peak_amplitude.toFixed(4) : 'N/A';
            const hasTrace = trace ? 'Yes' : 'No';
            document.getElementById('roi-info').innerHTML =
                `<strong>Cell ${{cellId}}</strong> | ` +
                `Type: <span style="color: ${{cellType === 'Significant' ? 'green' : 'blue'}}">${{cellType}}</span> | ` +
                `Info: ${{infoBitsStr}} bits | ` +
                `Peak Pos: ${{peakPosStr}} | ` +
                `Peak Amp: ${{peakAmpStr}}`;
        }}

        // Highlight selected ROI
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

            // Also highlight in sequence heatmap
            highlightSequenceCell(roiId);
        }}

        // Highlight selected cell in sequence heatmap
        function highlightSequenceCell(roiId) {{
            const heatmapDiv = document.getElementById('sequence-heatmap');
            if (!heatmapDiv || !sequenceHeatmap || sequenceHeatmap.length === 0) return;

            const cellId = parseInt(roiId);
            const cellIdx = sequenceCellIds.indexOf(cellId);

            const layout = heatmapDiv.layout || {{}};
            let shapes = [];

            if (cellIdx >= 0) {{
                // Draw a horizontal line/rectangle to highlight the row
                const maxTime = xaxis[xaxis.length - 1] + (xaxis[1] - xaxis[0]);
                shapes.push({{
                    type: 'rect',
                    x0: -0.05,
                    x1: maxTime + 0.05,
                    y0: cellIdx - 0.5,
                    y1: cellIdx + 0.5,
                    line: {{
                        color: 'cyan',
                        width: 2
                    }},
                    fillcolor: 'rgba(0,255,255,0)'
                }});
            }}

            Plotly.relayout('sequence-heatmap', {{ shapes: shapes }});
        }}

        // Create sequence heatmap for time cells sorted by peak time
        function createSequenceHeatmap() {{
            if (!sequenceHeatmap || sequenceHeatmap.length === 0) {{
                // No time cells to display
                const emptyLayout = {{
                    title: 'Time Cell Sequence (No significant cells)',
                    width: 400,
                    height: 400,
                    margin: {{ t: 50, b: 60, l: 60, r: 80 }}
                }};
                Plotly.newPlot('sequence-heatmap', [], emptyLayout);
                return;
            }}

            const nCells = sequenceHeatmap.length;
            const maxTime = xaxis[xaxis.length - 1] + (xaxis[1] - xaxis[0]);

            // Create y-axis labels (cell IDs)
            const yLabels = sequenceCellIds.map(id => `Cell ${{id}}`);

            const heatmapTrace = {{
                z: sequenceHeatmap,
                x: xaxis,
                y: Array.from({{ length: nCells }}, (_, i) => i),
                type: 'heatmap',
                colorscale: 'Greys',
                colorbar: {{
                    title: 'Activity',
                    titleside: 'right',
                    len: 0.8
                }},
                hovertemplate: 'Cell %{{customdata}}<br>Time: %{{x:.2f}} s<br>Activity: %{{z:.3f}}<extra></extra>',
                customdata: sequenceCellIds.map((id, i) => Array(xaxis.length).fill(id)).flat()
            }};

            // Reshape customdata to match heatmap dimensions
            heatmapTrace.customdata = sequenceCellIds.map(id => Array(xaxis.length).fill(id));

            const layout = {{
                title: `Time Cells (${{nCells}}, sorted by peak)`,
                width: 400,
                height: 400,
                xaxis: {{
                    title: xLabel,
                    range: [0, maxTime]
                }},
                yaxis: {{
                    title: 'Cells',
                    tickmode: 'array',
                    tickvals: [],
                    ticktext: [],
                    autorange: 'reversed'
                }},
                margin: {{ t: 50, b: 60, l: 60, r: 80 }}
            }};

            Plotly.newPlot('sequence-heatmap', [heatmapTrace], layout);

            // Add click handler to highlight cell in both plots
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

        // Initialize
        createResponseMap();
        createSequenceHeatmap();

        // Initialize empty trace plot
        const emptyLayout = {{
            title: 'Select a cell',
            width: 500,
            height: 350,
            xaxis: {{ title: xLabel }},
            yaxis: {{ title: 'dF/F' }},
            margin: {{ t: 50, b: 50, l: 60, r: 30 }}
        }};
        Plotly.newPlot('trace-plot', [], emptyLayout);
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

    # Load from suite2p ops.npy
    ops_path = suite2p_path / 'ops.npy'
    if ops_path.exists():
        suite2p_ops = np.load(ops_path, allow_pickle=True).item()
        return suite2p_ops[config['mean_img_key']]

    # Try alternative paths
    alt_path = suite2p_path.parent / 'ops.npy'
    if alt_path.exists():
        suite2p_ops = np.load(alt_path, allow_pickle=True).item()
        return suite2p_ops.get(config['mean_img_key'], suite2p_ops.get('meanImg'))

    raise FileNotFoundError(f"Could not find mean image for {rec}")


def load_roi_stat(rec, config):
    """
    Load gcamp_stats.npy and soma_class.npz to get ROI coordinates for active somas.

    Returns
    -------
    tuple: (gcamp_stats, active_soma_indices)
        gcamp_stats: array of dicts with 'xpix', 'ypix' for ALL ROIs
        active_soma_indices: array mapping cell_id (0, 1, 2...) to original ROI index
    """
    parts = rec.split('-')
    anm, date = parts[0], parts[1]

    # Load gcamp_stats.npy (contains ALL ROIs)
    gcamp_stats_path = Path(config['gcamp_stats_pattern'].format(anm=anm, date=date))
    if not gcamp_stats_path.exists():
        raise FileNotFoundError(f"Could not find gcamp_stats.npy for {rec}: {gcamp_stats_path}")

    gcamp_stats = np.load(gcamp_stats_path, allow_pickle=True)

    # Load soma_class.npz to get the active soma mask
    soma_class_path = Path(config['soma_class_pattern'].format(anm=anm, date=date))
    if not soma_class_path.exists():
        raise FileNotFoundError(f"Could not find soma_class.npz for {rec}: {soma_class_path}")

    soma_data = np.load(soma_class_path)
    is_active_soma = soma_data['is_soma']  # Boolean mask

    # Get indices of active somas - this maps cell_id to original ROI index
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
config = get_config()
OUT_DIR_RAW_DATA = config['out_dir_raw_data']

# Output directory
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'interactive_plots'
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

# Time cell selection thresholds
TI_threshold = 0.2
shuff_TI_thresh = 99

# Time binning parameters
time_bin_size = 0.1  # seconds (3 frames at 30Hz)
max_lap_duration_s = 6.0  # fixed time window in seconds
frame_rate = 30  # Hz

# Time axis in seconds (0 to max_lap_duration_s)
n_bins = int(max_lap_duration_s / time_bin_size)  # 60 bins
xaxis = np.arange(n_bins) * time_bin_size  # 0, 0.1, 0.2, ... 5.9 seconds

# Example recording list (modify as needed)
rec_lst = ['AC989-20250711-02', ]

#%% Main loop
if __name__ == '__main__':
    error_lst = []

    for rec in rec_lst:
        print(f"Processing {rec}...")

        try:
            # Load reference image
            mean_img = load_mean_image(rec, config)

            # Load dataframe
            df_data = load_dataframe(rec, config)

            # Select time cells
            df_valid = select_time_cell(
                df_data,
                TI_threshold=TI_threshold,
                shuff_TI_thresh=shuff_TI_thresh
            )

            # Load ROI stat for pixel coordinates
            roi_stat = None
            active_soma_indices = None
            try:
                roi_stat, active_soma_indices = load_roi_stat(rec, config)
                print(f"  Loaded {len(roi_stat)} ROIs from gcamp_stats.npy")
                print(f"  Active soma cells: {len(active_soma_indices)}")
            except FileNotFoundError as e:
                print(f"  Warning: {e}")

            # Generate interactive HTML
            save_path = OUT_DIR_FIG / f"{rec}_interactive_time_cell.html"
            generate_interactive_html(
                rec=rec,
                df_valid=df_valid,
                mean_img=mean_img,
                xaxis=xaxis,
                save_path=save_path,
                config=config,
                roi_stat=roi_stat,
                active_soma_indices=active_soma_indices
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
