# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 2026

@author: Jingyu Cao

Comprehensive FOV analysis: similarity quantification, FOV identification, and alignment tracking.

This script combines:
1. FOV Similarity Analysis - Quantify similarity across days using both Ch1 and Ch2
2. FOV Identification - Cluster sessions into distinct FOVs based on structural features
   (blood vessels, stable landmarks) with HIGH TOLERANCE for day-to-day drift
3. Alignment Tracking - Track drift and identify sessions with poor alignment

Key approach for FOV identification:
- Extract structural features (blood vessels) using edge detection and morphological ops
- Use feature-based similarity that is robust to intensity changes and minor drift
- Higher distance threshold for clustering (tolerant of gradual drift within same FOV)
- Sessions from same FOV can have different apparent similarity due to expression changes

Metrics computed:
- Structural Similarity Index (SSIM) for both channels
- Normalized Cross-Correlation (NCC) for both channels
- Feature-based similarity (blood vessel patterns) - PRIMARY for FOV identification
- Phase Correlation for shift estimation
- Combined similarity score with channel weighting

Outputs:
- fov_analysis_combined.csv: Complete per-session results with FOV assignment and alignment metrics
- fov_summary.csv: Summary of FOVs per animal
- alignment_summary.csv: Alignment quality summary per animal
- Per-animal visualization folders with similarity matrices, dendrograms, and alignment plots
"""

from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel, gaussian
from skimage.morphology import binary_dilation, disk
from sklearn.metrics import silhouette_score

from dlight_imagaing_learning.geco_dlight.recording_list import rec_lst
from common import plotting_functions_Jingyu as pf

#%% Configuration
OUT_DIR_RAW_DATA = Path(r"Z:\Jingyu\dlight_learning\geco_dlight")
OUT_DIR_FIG = OUT_DIR_RAW_DATA / 'FOV_analysis'
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

# Image preprocessing parameters
EDGE_CROP = 16  # Pixels to crop from edges (avoid registration artifacts)
GAUSSIAN_SIGMA = 2  # Smoothing for noise reduction

# Alignment warning threshold
ALIGNMENT_WARNING_THRESHOLD = 15  # pixels - flag sessions with shift > this

# FOV identification parameters - VERY HIGH TOLERANCE for same FOV
# Sessions with drift are still the same FOV - only truly different locations should be separate
FOV_DISTANCE_THRESHOLD = 0.7  # Distance threshold for clustering (higher = more tolerant)
MIN_FEATURE_SIMILARITY = 0.15  # Minimum feature similarity to be considered same FOV (very low)
MAX_SHIFT_FOR_SAME_FOV = 50  # pixels - if shift is within this, likely same FOV with drift

# Similarity weights for combined score: (ssim_ch1, ncc_ch1, ssim_ch2, ncc_ch2)
SIMILARITY_WEIGHTS = (0.3, 0.3, 0.2, 0.2)

# Feature extraction weights: prioritize Ch2 (structural) for FOV identification
FEATURE_WEIGHTS = (0.3, 0.7)  # (ch1_weight, ch2_weight) for FOV identification


#%% ============================================================================
#   IMAGE PROCESSING FUNCTIONS
# ==============================================================================

def preprocess_image(img, edge_crop=EDGE_CROP, sigma=GAUSSIAN_SIGMA):
    """Preprocess image for similarity comparison."""
    if img is None:
        return None

    # Crop edges
    if edge_crop > 0:
        img = img[edge_crop:-edge_crop, edge_crop:-edge_crop]

    # Convert to float and normalize to 0-1
    img = img.astype(np.float64)
    img_min, img_max = np.nanpercentile(img, [1, 99])
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-10)

    # Light Gaussian smoothing
    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=sigma)

    return img


def extract_vessel_features(img, sigma_smooth=2, dark_threshold_percentile=15):
    """
    Extract blood vessel / structural features from image.

    Blood vessels appear as DARK branches (shadows) in the image.
    This function detects these dark structures that remain consistent
    across days even with expression level changes.

    Args:
        img: Preprocessed image (normalized 0-1)
        sigma_smooth: Gaussian smoothing before detection
        dark_threshold_percentile: Percentile threshold for dark structures (lower = darker)

    Returns:
        Binary feature map highlighting blood vessels (dark branches)
    """
    if img is None:
        return None

    # Smooth to reduce noise
    smoothed = gaussian(img, sigma=sigma_smooth)

    # Blood vessels are DARK - find pixels below a low percentile threshold
    # This captures the dark branching structures
    threshold = np.percentile(smoothed, dark_threshold_percentile)
    dark_structures = smoothed < threshold

    # Also use edge detection to find vessel boundaries
    edges = sobel(smoothed)
    edges_norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-10)
    strong_edges = edges_norm > np.percentile(edges_norm, 80)

    # Combine: dark regions AND their edges (vessel boundaries)
    # This captures both the vessel interior and boundaries
    vessel_features = dark_structures | strong_edges

    # Dilate to make features more robust to small shifts/drift
    vessel_features = binary_dilation(vessel_features, disk(3))

    return vessel_features.astype(np.float32)


def compute_feature_similarity(feat1, feat2, max_shift=30):
    """
    Compute similarity between two binary feature maps with SHIFT TOLERANCE.

    Uses Dice coefficient computed at the best alignment within max_shift pixels.
    This makes the metric robust to drift between sessions.

    Args:
        feat1, feat2: Binary feature maps
        max_shift: Maximum shift to search for best alignment (pixels)
    """
    if feat1 is None or feat2 is None:
        return np.nan

    if feat1.shape != feat2.shape:
        return np.nan

    # First compute Dice at current alignment
    intersection = np.sum(feat1 * feat2)
    union = np.sum(feat1) + np.sum(feat2)

    if union == 0:
        return 1.0  # Both empty = identical

    dice_original = 2.0 * intersection / union

    # If already good, don't bother searching
    if dice_original > 0.5:
        return dice_original

    # Search for better alignment using phase correlation on feature maps
    try:
        shift, _, _ = phase_cross_correlation(feat1, feat2, upsample_factor=1)
        shift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)

        # If shift is within tolerance, compute Dice at shifted position
        if shift_magnitude <= max_shift:
            # Shift feat2 to align with feat1
            shifted_feat2 = ndimage.shift(feat2, shift, order=0, mode='constant', cval=0)
            intersection_shifted = np.sum(feat1 * shifted_feat2)
            dice_shifted = 2.0 * intersection_shifted / union

            return max(dice_original, dice_shifted)
    except Exception:
        pass

    return dice_original


def compute_feature_correlation(feat1, feat2, max_shift=30):
    """
    Compute correlation between feature maps with SHIFT TOLERANCE.

    Searches for best alignment within max_shift pixels.
    """
    if feat1 is None or feat2 is None:
        return np.nan

    if feat1.shape != feat2.shape:
        return np.nan

    # Flatten and compute correlation at original position
    f1 = feat1.flatten()
    f2 = feat2.flatten()

    # Handle constant arrays
    if np.std(f1) < 1e-10 or np.std(f2) < 1e-10:
        return 1.0 if np.allclose(f1, f2) else 0.0

    corr_original = np.corrcoef(f1, f2)[0, 1]
    if np.isnan(corr_original):
        corr_original = 0.0

    # If already good, return
    if corr_original > 0.5:
        return corr_original

    # Try to find better alignment
    try:
        shift, _, _ = phase_cross_correlation(feat1, feat2, upsample_factor=1)
        shift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)

        if shift_magnitude <= max_shift:
            shifted_feat2 = ndimage.shift(feat2, shift, order=0, mode='constant', cval=0)
            f2_shifted = shifted_feat2.flatten()

            if np.std(f2_shifted) > 1e-10:
                corr_shifted = np.corrcoef(f1, f2_shifted)[0, 1]
                if not np.isnan(corr_shifted):
                    return max(corr_original, corr_shifted)
    except Exception:
        pass

    return corr_original


def compute_vessel_overlap_ratio(feat1, feat2, max_shift=30):
    """
    Compute what fraction of vessels in one image overlap with vessels in another.

    This is asymmetric but useful for determining if images show the same region.
    If most vessels from image 1 appear in image 2 (even if shifted), they're likely
    the same FOV.
    """
    if feat1 is None or feat2 is None:
        return np.nan

    if feat1.shape != feat2.shape:
        return np.nan

    n_vessels_1 = np.sum(feat1)
    n_vessels_2 = np.sum(feat2)

    if n_vessels_1 == 0 or n_vessels_2 == 0:
        return 1.0 if n_vessels_1 == n_vessels_2 else 0.0

    # Dilate feat2 to be more tolerant of small shifts
    feat2_dilated = binary_dilation(feat2, disk(5))

    # What fraction of feat1 vessels are covered by dilated feat2?
    overlap = np.sum(feat1 * feat2_dilated) / n_vessels_1

    return overlap


#%% ============================================================================
#   SIMILARITY METRICS
# ==============================================================================

def compute_ncc(img1, img2):
    """Compute Normalized Cross-Correlation between two images."""
    if img1 is None or img2 is None:
        return np.nan

    if img1.shape != img2.shape:
        return np.nan

    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-10)
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-10)

    ncc = np.mean(img1_norm * img2_norm)
    return ncc


def compute_ssim_score(img1, img2):
    """Compute Structural Similarity Index between two images."""
    if img1 is None or img2 is None:
        return np.nan

    if img1.shape != img2.shape:
        return np.nan

    try:
        score = ssim(img1, img2, data_range=1.0)
        return score
    except Exception:
        return np.nan


def compute_phase_correlation(img1, img2, upsample_factor=20):
    """
    Compute phase correlation to estimate shift between images.

    Returns:
        shift: (y_shift, x_shift) in pixels
        correlation: quality metric
    """
    if img1 is None or img2 is None:
        return (np.nan, np.nan), np.nan

    if img1.shape != img2.shape:
        return (np.nan, np.nan), np.nan

    try:
        shift, error, diffphase = phase_cross_correlation(
            img1, img2,
            upsample_factor=upsample_factor,
            normalization='phase'
        )
        correlation = 1.0 - error if error is not None else np.nan
        return tuple(shift), correlation
    except Exception:
        return (np.nan, np.nan), np.nan


def compute_pairwise_metrics(img1_ch1, img2_ch1, img1_ch2, img2_ch2,
                             feat1_ch1=None, feat2_ch1=None,
                             feat1_ch2=None, feat2_ch2=None):
    """
    Compute all pairwise similarity metrics using both channels.

    Includes feature-based similarity for robust FOV identification.

    Returns dict with all metrics.
    """
    metrics = {}

    # SSIM for both channels
    metrics['ssim_ch1'] = compute_ssim_score(img1_ch1, img2_ch1)
    metrics['ssim_ch2'] = compute_ssim_score(img1_ch2, img2_ch2)

    # NCC for both channels
    metrics['ncc_ch1'] = compute_ncc(img1_ch1, img2_ch1)
    metrics['ncc_ch2'] = compute_ncc(img1_ch2, img2_ch2)

    # Feature-based similarity (blood vessels / structural features)
    # This is MORE ROBUST for FOV identification as it focuses on stable landmarks
    metrics['feature_dice_ch1'] = compute_feature_similarity(feat1_ch1, feat2_ch1)
    metrics['feature_dice_ch2'] = compute_feature_similarity(feat1_ch2, feat2_ch2)
    metrics['feature_corr_ch1'] = compute_feature_correlation(feat1_ch1, feat2_ch1)
    metrics['feature_corr_ch2'] = compute_feature_correlation(feat1_ch2, feat2_ch2)

    # Phase correlation for shift estimation (use Ch2/structural preferentially)
    if img1_ch2 is not None and img2_ch2 is not None:
        shift, corr = compute_phase_correlation(img1_ch2, img2_ch2)
    else:
        shift, corr = compute_phase_correlation(img1_ch1, img2_ch1)

    metrics['shift_y'] = shift[0]
    metrics['shift_x'] = shift[1]
    metrics['phase_correlation'] = corr
    metrics['shift_magnitude'] = np.sqrt(shift[0]**2 + shift[1]**2) if not np.isnan(shift[0]) else np.nan

    # Combined similarity score (for general quality assessment)
    valid_metrics = []
    valid_weights = []
    for metric, weight in zip(['ssim_ch1', 'ncc_ch1', 'ssim_ch2', 'ncc_ch2'], SIMILARITY_WEIGHTS):
        if not np.isnan(metrics[metric]):
            valid_metrics.append(metrics[metric])
            valid_weights.append(weight)

    if len(valid_metrics) > 0:
        metrics['combined'] = np.average(valid_metrics, weights=valid_weights)
    else:
        metrics['combined'] = np.nan

    # Vessel overlap ratio (more tolerant of drift)
    metrics['vessel_overlap_ch1'] = compute_vessel_overlap_ratio(feat1_ch1, feat2_ch1)
    metrics['vessel_overlap_ch2'] = compute_vessel_overlap_ratio(feat1_ch2, feat2_ch2)

    # Feature-based similarity for FOV identification (weighted by channel)
    # Prioritize Ch2 (structural) as it's more stable across days
    # Use MAX of dice, correlation, and overlap - if ANY metric shows similarity, they're likely same FOV
    feat_ch1_metrics = [metrics['feature_dice_ch1'], metrics['feature_corr_ch1'], metrics['vessel_overlap_ch1']]
    feat_ch2_metrics = [metrics['feature_dice_ch2'], metrics['feature_corr_ch2'], metrics['vessel_overlap_ch2']]

    feat_ch1 = np.nanmax(feat_ch1_metrics)  # Use MAX - be tolerant
    feat_ch2 = np.nanmax(feat_ch2_metrics)

    if not np.isnan(feat_ch2):
        metrics['feature_similarity'] = FEATURE_WEIGHTS[0] * feat_ch1 + FEATURE_WEIGHTS[1] * feat_ch2
    elif not np.isnan(feat_ch1):
        metrics['feature_similarity'] = feat_ch1
    else:
        metrics['feature_similarity'] = metrics['combined']  # Fallback

    # Also store if shift is within acceptable range for same FOV
    metrics['shift_within_fov_range'] = metrics['shift_magnitude'] <= MAX_SHIFT_FOR_SAME_FOV if not np.isnan(metrics['shift_magnitude']) else True

    return metrics


#%% ============================================================================
#   FOV CLUSTERING
# ==============================================================================

def cluster_fovs(feature_similarity_matrix, combined_similarity_matrix, session_names,
                 shift_matrix=None, method='complete', distance_threshold=FOV_DISTANCE_THRESHOLD):
    """
    Cluster sessions into distinct FOVs with VERY HIGH TOLERANCE for drift.

    Key principle: Sessions from the same imaging location but with different
    amounts of drift should be classified as the SAME FOV. Only truly different
    anatomical locations should be considered different FOVs.

    Approach:
    1. Default: assume ALL sessions are the same FOV
    2. Only split if there's OVERWHELMING evidence of different location:
       - Very low feature similarity AND
       - Large, discontinuous shift pattern (not gradual drift)

    Args:
        feature_similarity_matrix: Similarity based on structural features
        combined_similarity_matrix: Combined pixel-wise similarity (for reference)
        session_names: List of session names
        shift_matrix: Optional matrix of pairwise shifts
        method: Linkage method
        distance_threshold: Distance threshold for clustering

    Returns:
        fov_labels: Array of FOV assignments (1-indexed)
        n_fovs: Number of distinct FOVs identified
    """
    n_sessions = len(session_names)

    if n_sessions < 2:
        return np.ones(n_sessions, dtype=int), 1

    # DEFAULT: All sessions are the same FOV
    labels = np.ones(n_sessions, dtype=int)

    # Compute statistics to decide if we need to split
    # Get upper triangle (pairwise comparisons)
    triu_indices = np.triu_indices(n_sessions, k=1)
    pairwise_similarities = feature_similarity_matrix[triu_indices]

    # Check if there's evidence for multiple FOVs
    min_similarity = np.nanmin(pairwise_similarities)
    mean_similarity = np.nanmean(pairwise_similarities)
    std_similarity = np.nanstd(pairwise_similarities)

    # Only consider splitting if:
    # 1. Minimum pairwise similarity is VERY low (< MIN_FEATURE_SIMILARITY)
    # 2. There's high variance in similarities (suggesting distinct groups)
    needs_splitting = (min_similarity < MIN_FEATURE_SIMILARITY and
                       std_similarity > 0.15 and
                       mean_similarity < 0.5)

    if not needs_splitting:
        # All same FOV - most common case
        return labels, 1

    # If we get here, there might be multiple FOVs
    # Use very conservative clustering
    distance_matrix = 1 - feature_similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = np.clip(distance_matrix, 0, 1)

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method=method)

    # Use very high distance threshold
    labels = fcluster(Z, distance_threshold, criterion='distance')
    n_clusters = len(np.unique(labels))

    # Post-processing: merge small clusters back
    # If a "different FOV" has only 1-2 sessions, it's probably just noise
    for cluster_id in np.unique(labels):
        cluster_size = np.sum(labels == cluster_id)
        if cluster_size <= 2 and n_clusters > 1:
            # Merge this small cluster into the largest cluster
            largest_cluster = np.argmax(np.bincount(labels))
            labels[labels == cluster_id] = largest_cluster

    # Renumber clusters to be consecutive starting from 1
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels, 1)}
    labels = np.array([label_map[l] for l in labels])

    return labels, len(np.unique(labels))


def cluster_fovs_by_connectivity(feature_similarity_matrix, session_names,
                                  similarity_threshold=0.25):
    """
    Alternative clustering using connectivity-based approach.

    Two sessions are in the same FOV if they can be connected through
    a chain of sessions with similarity above threshold.

    This is more tolerant of gradual drift where adjacent sessions
    are similar but distant sessions may have drifted apart.
    """
    n_sessions = len(session_names)

    if n_sessions < 2:
        return np.ones(n_sessions, dtype=int), 1

    # Build adjacency matrix: sessions are connected if similarity > threshold
    adjacency = feature_similarity_matrix > similarity_threshold
    np.fill_diagonal(adjacency, True)

    # Find connected components using flood fill
    labels = np.zeros(n_sessions, dtype=int)
    current_label = 0

    for i in range(n_sessions):
        if labels[i] == 0:  # Not yet assigned
            current_label += 1
            # BFS to find all connected sessions
            queue = [i]
            while queue:
                node = queue.pop(0)
                if labels[node] == 0:
                    labels[node] = current_label
                    # Add all connected unvisited nodes
                    for j in range(n_sessions):
                        if adjacency[node, j] and labels[j] == 0:
                            queue.append(j)

    return labels, len(np.unique(labels))


#%% ============================================================================
#   ALIGNMENT ANALYSIS
# ==============================================================================

def compute_alignment_metrics(sessions, reference_idx=0):
    """
    Compute alignment metrics for all sessions relative to a reference.

    Args:
        sessions: list of session data dicts with 'proc_ch1' and 'proc_ch2'
        reference_idx: index of reference session (default: first session)

    Returns:
        List of dicts with alignment metrics for each session
    """
    ref_ch1 = sessions[reference_idx]['proc_ch1']
    ref_ch2 = sessions[reference_idx]['proc_ch2']

    alignment_metrics = []

    for i, sess in enumerate(sessions):
        # Shift relative to reference (Ch1)
        shift_ch1, corr_ch1 = compute_phase_correlation(ref_ch1, sess['proc_ch1'])

        # Shift relative to reference (Ch2)
        shift_ch2, corr_ch2 = compute_phase_correlation(ref_ch2, sess['proc_ch2'])

        # SSIM relative to reference
        ssim_ref_ch1 = compute_ssim_score(ref_ch1, sess['proc_ch1'])
        ssim_ref_ch2 = compute_ssim_score(ref_ch2, sess['proc_ch2'])

        # Shift relative to previous session
        if i > 0:
            prev_ch1 = sessions[i-1]['proc_ch1']
            prev_ch2 = sessions[i-1]['proc_ch2']
            shift_prev_ch1, _ = compute_phase_correlation(prev_ch1, sess['proc_ch1'])
            shift_prev_ch2, _ = compute_phase_correlation(prev_ch2, sess['proc_ch2'])
        else:
            shift_prev_ch1 = (0.0, 0.0)
            shift_prev_ch2 = (0.0, 0.0)

        # Shift magnitudes
        shift_mag_ch1 = np.sqrt(shift_ch1[0]**2 + shift_ch1[1]**2) if not np.isnan(shift_ch1[0]) else np.nan
        shift_mag_ch2 = np.sqrt(shift_ch2[0]**2 + shift_ch2[1]**2) if not np.isnan(shift_ch2[0]) else np.nan
        shift_mag_combined = np.nanmean([shift_mag_ch1, shift_mag_ch2])

        alignment_metrics.append({
            'shift_y_ref_ch1': shift_ch1[0],
            'shift_x_ref_ch1': shift_ch1[1],
            'shift_y_ref_ch2': shift_ch2[0],
            'shift_x_ref_ch2': shift_ch2[1],
            'shift_magnitude_ch1': shift_mag_ch1,
            'shift_magnitude_ch2': shift_mag_ch2,
            'shift_magnitude_combined': shift_mag_combined,
            'shift_y_from_prev_ch1': shift_prev_ch1[0],
            'shift_x_from_prev_ch1': shift_prev_ch1[1],
            'shift_y_from_prev_ch2': shift_prev_ch2[0],
            'shift_x_from_prev_ch2': shift_prev_ch2[1],
            'ssim_vs_ref_ch1': ssim_ref_ch1,
            'ssim_vs_ref_ch2': ssim_ref_ch2,
            'alignment_warning': shift_mag_combined > ALIGNMENT_WARNING_THRESHOLD
        })

    return alignment_metrics


#%% ============================================================================
#   PLOTTING FUNCTIONS
# ==============================================================================

def plot_similarity_matrix(similarity_matrix, session_names, fov_labels, animal,
                           metric_name, save_path):
    """Plot similarity matrix with FOV clustering annotations."""
    n_sessions = len(session_names)

    sort_idx = np.argsort(fov_labels)
    sorted_matrix = similarity_matrix[sort_idx][:, sort_idx]
    sorted_names = [session_names[i] for i in sort_idx]
    sorted_labels = fov_labels[sort_idx]

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(sorted_matrix, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect='equal')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{metric_name} Score', fontsize=12)

    short_names = [s.split('-')[1] for s in sorted_names]
    ax.set_xticks(range(n_sessions))
    ax.set_yticks(range(n_sessions))
    ax.set_xticklabels(short_names, rotation=90, fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    # FOV boundary lines
    unique_fovs = np.unique(sorted_labels)
    for fov in unique_fovs[:-1]:
        boundary = np.where(sorted_labels == fov)[0][-1] + 0.5
        ax.axhline(boundary, color='white', linewidth=2)
        ax.axvline(boundary, color='white', linewidth=2)

    # FOV labels
    for fov in unique_fovs:
        fov_indices = np.where(sorted_labels == fov)[0]
        mid_idx = np.mean(fov_indices)
        ax.text(n_sessions + 0.5, mid_idx, f'FOV {fov}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_title(f'{animal} - {metric_name} Similarity Matrix\n({len(unique_fovs)} distinct FOVs)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Session Date', fontsize=12)
    ax.set_ylabel('Session Date', fontsize=12)

    plt.tight_layout()
    pf.save_fig(fig, save_path.parent, save_path.stem, dpi=150, forms=['png'], save=1)
    plt.close(fig)


def plot_dendrogram(similarity_matrix, session_names, fov_labels, animal, save_path):
    """Plot dendrogram showing hierarchical clustering of sessions."""
    n_sessions = len(session_names)

    if n_sessions < 2:
        return

    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')

    fig, ax = plt.subplots(figsize=(max(12, n_sessions * 0.5), 6))

    short_names = [s.split('-')[1] for s in session_names]

    dendrogram(Z, labels=short_names, ax=ax, leaf_rotation=90,
               leaf_font_size=8, color_threshold=0.3)

    ax.set_title(f'{animal} - Session Clustering Dendrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Session Date', fontsize=12)
    ax.set_ylabel('Distance (1 - Similarity)', fontsize=12)

    plt.tight_layout()
    pf.save_fig(fig, save_path.parent, save_path.stem, dpi=150, forms=['png'], save=1)
    plt.close(fig)


def plot_fov_montage(session_data, fov_labels, animal, save_path):
    """Create montage showing representative images from each FOV."""
    unique_fovs = np.unique(fov_labels)
    n_fovs = len(unique_fovs)

    representatives = []
    for fov in unique_fovs:
        fov_indices = np.where(fov_labels == fov)[0]
        mid_idx = fov_indices[len(fov_indices) // 2]
        representatives.append(session_data[mid_idx])

    fig, axs = plt.subplots(2, n_fovs, figsize=(4 * n_fovs, 8), squeeze=False)
    fig.suptitle(f'{animal} - Representative Images from {n_fovs} FOVs',
                 fontsize=14, fontweight='bold')

    for col, (sess, fov) in enumerate(zip(representatives, unique_fovs)):
        # Ch1
        ax = axs[0, col]
        img_ch1 = sess['mean_img_ch1']
        if img_ch1 is not None:
            vmin, vmax = np.nanpercentile(img_ch1, [1, 99])
            ax.imshow(img_ch1, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'FOV {fov}\n{sess["rec"]}', fontsize=10)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Ch1 (dLight)', fontsize=12)

        # Ch2
        ax = axs[1, col]
        img_ch2 = sess['mean_img_ch2']
        if img_ch2 is not None:
            vmin, vmax = np.nanpercentile(img_ch2, [1, 98])
            ax.imshow(img_ch2, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            ax.text(0.5, 0.5, 'No Ch2', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Ch2 (structural)', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pf.save_fig(fig, save_path.parent, save_path.stem, dpi=150, forms=['png'], save=1)
    plt.close(fig)


def plot_alignment_summary(results_df, animal, save_path):
    """Plot comprehensive alignment summary for an animal."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    sessions = range(len(results_df))
    dates = [s.split('-')[1] for s in results_df['session']]
    tick_step = max(1, len(sessions) // 10)

    # Plot 1: X-Y drift trajectory (Ch1)
    ax = fig.add_subplot(gs[0, 0])
    valid = ~results_df['shift_x_ref_ch1'].isna()
    ax.plot(results_df.loc[valid, 'shift_x_ref_ch1'].values,
            results_df.loc[valid, 'shift_y_ref_ch1'].values,
            'b-o', markersize=6, alpha=0.7)
    ax.plot(0, 0, 'k*', markersize=12, label='Reference')
    ax.set_xlabel('X Shift (px)', fontsize=10)
    ax.set_ylabel('Y Shift (px)', fontsize=10)
    ax.set_title('FOV Drift (Ch1)', fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: X-Y drift trajectory (Ch2)
    ax = fig.add_subplot(gs[0, 1])
    valid = ~results_df['shift_x_ref_ch2'].isna()
    ax.plot(results_df.loc[valid, 'shift_x_ref_ch2'].values,
            results_df.loc[valid, 'shift_y_ref_ch2'].values,
            'r-o', markersize=6, alpha=0.7)
    ax.plot(0, 0, 'k*', markersize=12, label='Reference')
    ax.set_xlabel('X Shift (px)', fontsize=10)
    ax.set_ylabel('Y Shift (px)', fontsize=10)
    ax.set_title('FOV Drift (Ch2)', fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 3: Shift magnitude over time
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(sessions, results_df['shift_magnitude_ch1'], 'b-o', label='Ch1', alpha=0.7, markersize=4)
    ax.plot(sessions, results_df['shift_magnitude_ch2'], 'r-o', label='Ch2', alpha=0.7, markersize=4)
    ax.axhline(ALIGNMENT_WARNING_THRESHOLD, color='orange', linestyle='--',
               label=f'Warning ({ALIGNMENT_WARNING_THRESHOLD}px)')
    warnings = results_df[results_df['alignment_warning']]
    if len(warnings) > 0:
        ax.scatter(warnings.index, warnings['shift_magnitude_combined'],
                   c='red', s=80, marker='x', linewidths=2, zorder=5)
    ax.set_xlabel('Session', fontsize=10)
    ax.set_ylabel('Shift (pixels)', fontsize=10)
    ax.set_title('Alignment Drift Over Time', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xticks(list(sessions)[::tick_step])
    ax.set_xticklabels([dates[i] for i in list(sessions)[::tick_step]], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: SSIM vs reference over time
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(sessions, results_df['ssim_vs_ref_ch1'], 'b-o', label='Ch1', alpha=0.7, markersize=4)
    ax.plot(sessions, results_df['ssim_vs_ref_ch2'], 'r-o', label='Ch2', alpha=0.7, markersize=4)
    ax.axhline(0.7, color='orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Session', fontsize=10)
    ax.set_ylabel('SSIM (vs Reference)', fontsize=10)
    ax.set_title('Structural Similarity Over Time', fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(sessions)[::tick_step])
    ax.set_xticklabels([dates[i] for i in list(sessions)[::tick_step]], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Combined similarity over time
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(sessions, results_df['mean_combined_similarity'], 'g-o', alpha=0.7, markersize=4)
    ax.axhline(0.7, color='orange', linestyle='--', alpha=0.7, label='Quality threshold')
    ax.set_xlabel('Session', fontsize=10)
    ax.set_ylabel('Mean Combined Similarity', fontsize=10)
    ax.set_title('Mean Similarity to All Sessions', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(sessions)[::tick_step])
    ax.set_xticklabels([dates[i] for i in list(sessions)[::tick_step]], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: FOV assignment over time
    ax = fig.add_subplot(gs[1, 2])
    colors = plt.cm.Set1(np.linspace(0, 1, results_df['fov_id'].max()))
    for fov in results_df['fov_id'].unique():
        fov_mask = results_df['fov_id'] == fov
        ax.scatter(np.array(list(sessions))[fov_mask], results_df.loc[fov_mask, 'fov_id'],
                   c=[colors[fov-1]], s=60, label=f'FOV {fov}')
    ax.set_xlabel('Session', fontsize=10)
    ax.set_ylabel('FOV ID', fontsize=10)
    ax.set_title('FOV Assignment Over Time', fontsize=11, fontweight='bold')
    ax.set_yticks(range(1, results_df['fov_id'].max() + 1))
    ax.set_xticks(list(sessions)[::tick_step])
    ax.set_xticklabels([dates[i] for i in list(sessions)[::tick_step]], rotation=45, fontsize=8)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 7: Ch1 vs Ch2 shift agreement
    ax = fig.add_subplot(gs[2, 0])
    valid = ~(results_df['shift_x_ref_ch1'].isna() | results_df['shift_x_ref_ch2'].isna())
    ax.scatter(results_df.loc[valid, 'shift_magnitude_ch1'],
               results_df.loc[valid, 'shift_magnitude_ch2'], alpha=0.7, s=40)
    lim = max(results_df['shift_magnitude_ch1'].max(), results_df['shift_magnitude_ch2'].max(), 5)
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('Shift Magnitude Ch1 (px)', fontsize=10)
    ax.set_ylabel('Shift Magnitude Ch2 (px)', fontsize=10)
    ax.set_title('Channel Agreement: Shift', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if valid.sum() > 2:
        corr = np.corrcoef(results_df.loc[valid, 'shift_magnitude_ch1'],
                          results_df.loc[valid, 'shift_magnitude_ch2'])[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')

    # Plot 8: Ch1 vs Ch2 SSIM agreement
    ax = fig.add_subplot(gs[2, 1])
    valid = ~(results_df['ssim_vs_ref_ch1'].isna() | results_df['ssim_vs_ref_ch2'].isna())
    ax.scatter(results_df.loc[valid, 'ssim_vs_ref_ch1'],
               results_df.loc[valid, 'ssim_vs_ref_ch2'], alpha=0.7, s=40)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('SSIM Ch1', fontsize=10)
    ax.set_ylabel('SSIM Ch2', fontsize=10)
    ax.set_title('Channel Agreement: SSIM', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if valid.sum() > 2:
        corr = np.corrcoef(results_df.loc[valid, 'ssim_vs_ref_ch1'],
                          results_df.loc[valid, 'ssim_vs_ref_ch2'])[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')

    # Plot 9: Summary stats text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    n_sessions = len(results_df)
    n_fovs = results_df['fov_id'].nunique()
    n_warnings = results_df['alignment_warning'].sum()
    mean_shift = results_df['shift_magnitude_combined'].mean()
    mean_ssim_ch1 = results_df['ssim_vs_ref_ch1'].mean()
    mean_ssim_ch2 = results_df['ssim_vs_ref_ch2'].mean()

    summary_text = f"""Summary Statistics

Total Sessions: {n_sessions}
Distinct FOVs: {n_fovs}
Sessions with Warnings: {n_warnings}

Mean Shift from Reference: {mean_shift:.2f} px
Mean SSIM Ch1: {mean_ssim_ch1:.3f}
Mean SSIM Ch2: {mean_ssim_ch2:.3f}

FOV Distribution:"""

    for fov in sorted(results_df['fov_id'].unique()):
        count = (results_df['fov_id'] == fov).sum()
        summary_text += f"\n  FOV {fov}: {count} sessions"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    fig.suptitle(f'{animal} - Comprehensive FOV Analysis', fontsize=16, fontweight='bold', y=0.98)

    pf.save_fig(fig, save_path.parent, save_path.stem, dpi=150, forms=['png'], save=1)
    plt.close(fig)


def plot_session_grid(session_data, results_df, animal, save_path, max_sessions=20):
    """Plot grid of session images with FOV labels and alignment info."""
    n_sessions = min(len(session_data), max_sessions)
    n_cols = 5
    n_rows = int(np.ceil(n_sessions / n_cols))

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(3.5 * n_cols, 3 * n_rows * 2), squeeze=False)

    for idx in range(n_sessions):
        sess = session_data[idx]
        row_data = results_df.iloc[idx]

        row_ch1 = (idx // n_cols) * 2
        row_ch2 = row_ch1 + 1
        col = idx % n_cols

        # Ch1
        ax = axes[row_ch1, col]
        img = sess['mean_img_ch1']
        if img is not None:
            vmin, vmax = np.nanpercentile(img, [1, 99])
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)

        fov_id = row_data['fov_id']
        date = sess['rec'].split('-')[1]
        title = f'{date}\nFOV {fov_id}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')

        if row_data['alignment_warning']:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

        # Ch2
        ax = axes[row_ch2, col]
        img = sess['mean_img_ch2']
        if img is not None:
            vmin, vmax = np.nanpercentile(img, [1, 98])
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            shift_info = f'Δ({row_data["shift_x_ref_ch2"]:.1f}, {row_data["shift_y_ref_ch2"]:.1f})'
            ax.set_title(shift_info, fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Ch2', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n_sessions, n_rows * n_cols):
        row_ch1 = (idx // n_cols) * 2
        row_ch2 = row_ch1 + 1
        col = idx % n_cols
        axes[row_ch1, col].axis('off')
        axes[row_ch2, col].axis('off')

    fig.suptitle(f'{animal} - Session Overview (Ch1 top, Ch2 bottom)\nRed border = alignment warning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    pf.save_fig(fig, save_path.parent, save_path.stem, dpi=150, forms=['png'], save=1)
    plt.close(fig)


def plot_feature_extraction(session_data, animal, save_path, n_display=6):
    """
    Visualize the extracted vessel/structural features for quality control.

    Shows original image alongside extracted features to verify
    that blood vessels are being detected correctly.
    """
    n_sessions = len(session_data)
    display_indices = np.linspace(0, n_sessions-1, min(n_display, n_sessions), dtype=int)
    n_cols = len(display_indices)

    fig, axes = plt.subplots(4, n_cols, figsize=(3 * n_cols, 10), squeeze=False)

    for col, idx in enumerate(display_indices):
        sess = session_data[idx]
        date = sess['rec'].split('-')[1]

        # Ch2 original
        ax = axes[0, col]
        img = sess['proc_ch2']
        if img is not None:
            ax.imshow(img, cmap='gray')
        ax.set_title(f'{date}', fontsize=9)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Ch2 Original', fontsize=10)

        # Ch2 features
        ax = axes[1, col]
        feat = sess['feat_ch2']
        if feat is not None:
            ax.imshow(feat, cmap='hot')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Ch2 Features', fontsize=10)

        # Ch1 original
        ax = axes[2, col]
        img = sess['proc_ch1']
        if img is not None:
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Ch1 Original', fontsize=10)

        # Ch1 features
        ax = axes[3, col]
        feat = sess['feat_ch1']
        if feat is not None:
            ax.imshow(feat, cmap='hot')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel('Ch1 Features', fontsize=10)

    fig.suptitle(f'{animal} - Extracted Structural Features (Blood Vessels)\nUsed for FOV identification',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    pf.save_fig(fig, save_path.parent, save_path.stem, dpi=150, forms=['png'], save=1)
    plt.close(fig)


#%% ============================================================================
#   MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_animal(anm_recs, animal_name):
    """
    Complete FOV analysis for a single animal.

    Returns dict with all results.
    """
    # Load all session data
    session_data = []

    for rec in tqdm(anm_recs, desc=f'Loading {animal_name}'):
        anm_name, date, ss = rec.split('-')
        p_suite2p = Path(rf"Z:\Jingyu\2P_Recording\{anm_name}\{anm_name}-{date}\{ss}\RegOnly\suite2p\plane0")

        if not p_suite2p.exists():
            continue

        ops_path = p_suite2p / 'ops.npy'
        if not ops_path.exists():
            continue

        try:
            suite2p_ops = np.load(ops_path, allow_pickle=True).item()

            mean_img_ch1 = suite2p_ops.get('meanImg', None)
            mean_img_ch2 = suite2p_ops.get('meanImg_chan2', None)

            if mean_img_ch1 is None:
                continue

            # Preprocess images
            proc_ch1 = preprocess_image(mean_img_ch1)
            proc_ch2 = preprocess_image(mean_img_ch2)

            # Extract structural features (blood vessels) for robust FOV identification
            feat_ch1 = extract_vessel_features(proc_ch1)
            feat_ch2 = extract_vessel_features(proc_ch2)

            session_data.append({
                'rec': rec,
                'mean_img_ch1': mean_img_ch1,
                'mean_img_ch2': mean_img_ch2,
                'proc_ch1': proc_ch1,
                'proc_ch2': proc_ch2,
                'feat_ch1': feat_ch1,
                'feat_ch2': feat_ch2
            })
        except Exception as e:
            print(f'Error loading {rec}: {e}')
            continue

    if len(session_data) < 2:
        print(f'{animal_name}: Not enough sessions ({len(session_data)})')
        return None

    n_sessions = len(session_data)
    session_names = [s['rec'] for s in session_data]

    print(f'Computing similarity matrices for {animal_name} ({n_sessions} sessions)...')

    # Initialize matrices
    ssim_ch1_matrix = np.eye(n_sessions)
    ssim_ch2_matrix = np.eye(n_sessions)
    ncc_ch1_matrix = np.eye(n_sessions)
    ncc_ch2_matrix = np.eye(n_sessions)
    combined_matrix = np.eye(n_sessions)
    feature_matrix = np.eye(n_sessions)  # Feature-based similarity for FOV ID

    # Compute pairwise similarity
    for i in range(n_sessions):
        for j in range(i + 1, n_sessions):
            metrics = compute_pairwise_metrics(
                session_data[i]['proc_ch1'], session_data[j]['proc_ch1'],
                session_data[i]['proc_ch2'], session_data[j]['proc_ch2'],
                session_data[i]['feat_ch1'], session_data[j]['feat_ch1'],
                session_data[i]['feat_ch2'], session_data[j]['feat_ch2']
            )

            ssim_ch1_matrix[i, j] = ssim_ch1_matrix[j, i] = metrics['ssim_ch1']
            ssim_ch2_matrix[i, j] = ssim_ch2_matrix[j, i] = metrics['ssim_ch2']
            ncc_ch1_matrix[i, j] = ncc_ch1_matrix[j, i] = metrics['ncc_ch1']
            ncc_ch2_matrix[i, j] = ncc_ch2_matrix[j, i] = metrics['ncc_ch2']
            combined_matrix[i, j] = combined_matrix[j, i] = metrics['combined']
            feature_matrix[i, j] = feature_matrix[j, i] = metrics['feature_similarity']

    # Cluster into FOVs using CONNECTIVITY-BASED approach
    # This is tolerant of gradual drift - sessions connected through chain of similar sessions
    # are considered the same FOV
    fov_labels, n_fovs = cluster_fovs_by_connectivity(feature_matrix, session_names,
                                                       similarity_threshold=0.2)

    # If connectivity gives too many clusters, fall back to conservative threshold-based
    if n_fovs > 3:
        fov_labels, n_fovs = cluster_fovs(feature_matrix, combined_matrix, session_names)

    print(f'{animal_name}: Identified {n_fovs} distinct FOV(s)')

    # Compute alignment metrics
    alignment_metrics = compute_alignment_metrics(session_data)

    # Build combined results dataframe
    results_df = pd.DataFrame({
        'animal': animal_name,
        'session': session_names,
        'session_idx': range(n_sessions),
        'fov_id': fov_labels,
        'mean_ssim_ch1': [np.nanmean(ssim_ch1_matrix[i, :]) for i in range(n_sessions)],
        'mean_ssim_ch2': [np.nanmean(ssim_ch2_matrix[i, :]) for i in range(n_sessions)],
        'mean_ncc_ch1': [np.nanmean(ncc_ch1_matrix[i, :]) for i in range(n_sessions)],
        'mean_ncc_ch2': [np.nanmean(ncc_ch2_matrix[i, :]) for i in range(n_sessions)],
        'mean_combined_similarity': [np.nanmean(combined_matrix[i, :]) for i in range(n_sessions)],
        'mean_feature_similarity': [np.nanmean(feature_matrix[i, :]) for i in range(n_sessions)]
    })

    # Add alignment metrics
    for key in alignment_metrics[0].keys():
        results_df[key] = [m[key] for m in alignment_metrics]

    return {
        'animal': animal_name,
        'session_data': session_data,
        'session_names': session_names,
        'ssim_ch1_matrix': ssim_ch1_matrix,
        'ssim_ch2_matrix': ssim_ch2_matrix,
        'ncc_ch1_matrix': ncc_ch1_matrix,
        'ncc_ch2_matrix': ncc_ch2_matrix,
        'combined_matrix': combined_matrix,
        'feature_matrix': feature_matrix,
        'fov_labels': fov_labels,
        'n_fovs': n_fovs,
        'results_df': results_df
    }


#%% ============================================================================
#   RUN ANALYSIS
# ==============================================================================

if __name__ == '__main__':
    # Group recordings by animal
    recs_by_animal = defaultdict(list)
    for rec in rec_lst:
        recs_by_animal[rec[:5]].append(rec)

    all_results = []
    fov_summary_data = []
    alignment_summary_data = []

    for animal, anm_recs in recs_by_animal.items():
        print(f'\n{"="*70}')
        print(f'Processing {animal} ({len(anm_recs)} sessions)')
        print(f'{"="*70}')

        results = analyze_animal(anm_recs, animal)

        if results is None:
            continue

        all_results.append(results)
        results_df = results['results_df']

        # Create output directory
        animal_fig_dir = OUT_DIR_FIG / animal
        animal_fig_dir.mkdir(parents=True, exist_ok=True)

        # Generate all plots
        # Feature similarity matrix (PRIMARY for FOV identification)
        plot_similarity_matrix(
            results['feature_matrix'], results['session_names'],
            results['fov_labels'], animal, 'Feature (Vessels)',
            animal_fig_dir / f'{animal}_similarity_matrix_feature'
        )

        plot_similarity_matrix(
            results['combined_matrix'], results['session_names'],
            results['fov_labels'], animal, 'Combined',
            animal_fig_dir / f'{animal}_similarity_matrix_combined'
        )

        plot_similarity_matrix(
            results['ssim_ch1_matrix'], results['session_names'],
            results['fov_labels'], animal, 'SSIM Ch1',
            animal_fig_dir / f'{animal}_similarity_matrix_ssim_ch1'
        )

        plot_similarity_matrix(
            results['ssim_ch2_matrix'], results['session_names'],
            results['fov_labels'], animal, 'SSIM Ch2',
            animal_fig_dir / f'{animal}_similarity_matrix_ssim_ch2'
        )

        plot_dendrogram(
            results['combined_matrix'], results['session_names'],
            results['fov_labels'], animal,
            animal_fig_dir / f'{animal}_dendrogram'
        )

        plot_fov_montage(
            results['session_data'], results['fov_labels'], animal,
            animal_fig_dir / f'{animal}_fov_montage'
        )

        plot_alignment_summary(
            results_df, animal,
            animal_fig_dir / f'{animal}_alignment_summary'
        )

        plot_session_grid(
            results['session_data'], results_df, animal,
            animal_fig_dir / f'{animal}_session_grid'
        )

        # Plot feature extraction for quality control
        plot_feature_extraction(
            results['session_data'], animal,
            animal_fig_dir / f'{animal}_feature_extraction'
        )

        # Collect FOV summary
        for fov in sorted(np.unique(results['fov_labels'])):
            fov_mask = results['fov_labels'] == fov
            fov_sessions = [s for s, m in zip(results['session_names'], fov_mask) if m]
            fov_summary_data.append({
                'animal': animal,
                'fov_id': fov,
                'n_sessions': len(fov_sessions),
                'first_session': min(fov_sessions),
                'last_session': max(fov_sessions),
                'mean_within_fov_feature_sim': np.nanmean(
                    results['feature_matrix'][fov_mask][:, fov_mask]
                ),
                'mean_within_fov_combined_sim': np.nanmean(
                    results['combined_matrix'][fov_mask][:, fov_mask]
                ),
                'sessions': ', '.join(fov_sessions)
            })

        # Collect alignment summary
        alignment_summary_data.append({
            'animal': animal,
            'n_sessions': len(results_df),
            'n_fovs': results['n_fovs'],
            'n_warnings': results_df['alignment_warning'].sum(),
            'mean_shift_magnitude': results_df['shift_magnitude_combined'].mean(),
            'max_shift_magnitude': results_df['shift_magnitude_combined'].max(),
            'mean_ssim_ch1': results_df['ssim_vs_ref_ch1'].mean(),
            'mean_ssim_ch2': results_df['ssim_vs_ref_ch2'].mean(),
            'min_ssim_ch1': results_df['ssim_vs_ref_ch1'].min(),
            'min_ssim_ch2': results_df['ssim_vs_ref_ch2'].min(),
            'mean_feature_similarity': results_df['mean_feature_similarity'].mean(),
            'min_feature_similarity': results_df['mean_feature_similarity'].min()
        })

        print(f'\n{animal} Summary:')
        print(f'  Sessions: {len(results_df)}')
        print(f'  Distinct FOVs: {results["n_fovs"]}')
        print(f'  Alignment warnings: {results_df["alignment_warning"].sum()}')

    # Save combined results
    if all_results:
        # Per-session combined results
        combined_df = pd.concat([r['results_df'] for r in all_results], ignore_index=True)
        combined_df.to_csv(OUT_DIR_FIG / 'fov_analysis_combined.csv', index=False)

        # FOV summary
        fov_summary_df = pd.DataFrame(fov_summary_data)
        fov_summary_df.to_csv(OUT_DIR_FIG / 'fov_summary.csv', index=False)

        # Alignment summary
        alignment_summary_df = pd.DataFrame(alignment_summary_data)
        alignment_summary_df.to_csv(OUT_DIR_FIG / 'alignment_summary.csv', index=False)

        # Print final summary
        print('\n' + '='*70)
        print('FINAL SUMMARY')
        print('='*70)

        for _, row in alignment_summary_df.iterrows():
            print(f'\n{row["animal"]}:')
            print(f'  Total sessions: {row["n_sessions"]}')
            print(f'  Distinct FOVs: {row["n_fovs"]}')
            print(f'  Alignment warnings: {row["n_warnings"]}')
            print(f'  Mean shift: {row["mean_shift_magnitude"]:.2f} px')
            print(f'  Mean SSIM: Ch1={row["mean_ssim_ch1"]:.3f}, Ch2={row["mean_ssim_ch2"]:.3f}')
            print(f'  Mean feature similarity: {row["mean_feature_similarity"]:.3f}')

        print(f'\n{"="*70}')
        print(f'Results saved to: {OUT_DIR_FIG}')
        print('Output files:')
        print('  - fov_analysis_combined.csv: Complete per-session results')
        print('  - fov_summary.csv: FOV identification summary')
        print('  - alignment_summary.csv: Alignment quality summary per animal')
        print('  - [animal]/: Per-animal visualization folders')
        print('='*70)
