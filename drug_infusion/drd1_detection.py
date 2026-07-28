# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 17:33:03 2025

@author: Jingyu Cao
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.optimize import linear_sum_assignment

def reg2im(reg, height, width):
    im = np.zeros((height, width), dtype=np.uint8)
    im[reg[:, 0], reg[:, 1]] = 1
    return im

def mser_detection(im, **kwargs):
    # im = (norm(np.array(im)) * 255).astype(np.uint8)
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = im.shape
    mser = cv2.MSER_create(**kwargs)
    regions, _ = mser.detectRegions(im)
    nreg = len(regions)
    regs = np.zeros((nreg, h, w), dtype=bool)
    for i in range(nreg):
        # opencv coordinates are flipped
        regs[i, :, :] = reg2im(regions[i][:, ::-1], h, w)
    return regs

# ---------- helpers ----------

def coerce_roi_stack(X, name="X"):
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[None, ...]
    elif X.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D; got shape {X.shape}.")
    if X.dtype.kind != 'b':
        X = X.astype(bool, copy=False)
    if X.shape[0] == 0:
        raise ValueError(f"{name} has zero ROIs (shape {X.shape}).")
    return X

def centroids_and_areas(stack):
    N, H, W = stack.shape
    ys = np.arange(H, dtype=np.float32)[None, :, None]
    xs = np.arange(W, dtype=np.float32)[None, None, :]
    areas = stack.sum(axis=(1, 2))
    areas_safe = np.clip(areas, 1, None).astype(np.float32)
    cy = (stack * ys).sum(axis=(1, 2)) / areas_safe
    cx = (stack * xs).sum(axis=(1, 2)) / areas_safe
    return np.stack([cy, cx], axis=1), areas

def bounding_boxes(stack):
    N, H, W = stack.shape
    bboxes = np.zeros((N, 4), dtype=np.int32)
    any_y = stack.any(axis=2)
    ymins = np.argmax(any_y, axis=1)
    ymaxs = H - np.argmax(any_y[:, ::-1], axis=1)

    any_x = stack.any(axis=1)
    xmins = np.argmax(any_x, axis=1)
    xmaxs = W - np.argmax(any_x[:, ::-1], axis=1)

    empty = (stack.sum(axis=(1, 2)) == 0)
    bboxes[:, 0] = np.where(empty, 0, ymins)
    bboxes[:, 1] = np.where(empty, H, ymaxs)
    bboxes[:, 2] = np.where(empty, 0, xmins)
    bboxes[:, 3] = np.where(empty, W, xmaxs)
    return bboxes

def dice_on_dilated(a2d, b2d, dr: int):
    a, b = a2d, b2d
    if dr and dr > 0:
        s = generate_binary_structure(2, 1)  # 4-connected
        for _ in range(int(dr)):
            a = binary_dilation(a, structure=s)
            b = binary_dilation(b, structure=s)
    inter = np.count_nonzero(a & b)
    denom = np.count_nonzero(a) + np.count_nonzero(b)
    return (2.0 * inter / denom) if denom else 0.0

def crop_union_bbox(bboxA, bboxB, i, j):
    ya0, ya1, xa0, xa1 = bboxA[i]
    yb0, yb1, xb0, xb1 = bboxB[j]
    y0 = min(ya0, yb0); y1 = max(ya1, yb1)
    x0 = min(xa0, xb0); x1 = max(xa1, xb1)
    return y0, y1, x0, x1

def best_shifted_dice(a2d, b2d, dr: int, K: int):
    """Brute-force integer shift in [-K, K] for tiny windows."""
    if K <= 0:
        return dice_on_dilated(a2d, b2d, dr)
    H, W = a2d.shape
    best = 0.0
    for dy in range(-K, K + 1):
        y0a = max(0,  dy); y1a = min(H, H + dy)
        y0b = max(0, -dy); y1b = min(H, H - dy)
        if y1a <= y0a or y1b <= y0b: 
            continue
        for dx in range(-K, K + 1):
            x0a = max(0,  dx); x1a = min(W, W + dx)
            x0b = max(0, -dx); x1b = min(W, W - dx)
            if x1a <= x0a or x1b <= x0b:
                continue
            d = dice_on_dilated(
                a2d[y0a:y1a, x0a:x1a],
                b2d[y0b:y1b, x0b:x1b],
                dr
            )
            if d > best:
                best = d
    return best

# ---------- main matcher ----------

def match_rois(
    A, B,
    *,
    centroid_radius: float = 6.0,
    dilate_radius: int = 1,
    dice_thresh: float = 0.5,
    crop_to_bbox: bool = True,
    max_shift: int = 0
):
    """
    One-to-one matching between two ROI stacks using centroid-pruned Dice overlap.
    """
    # preprocess
    A = coerce_roi_stack(A, "A")
    B = coerce_roi_stack(B, "B")

    nA, H, W = A.shape
    nB, H2, W2 = B.shape
    if (H, W) != (H2, W2):
        raise ValueError(f"A and B must share spatial size; got {A.shape}, {B.shape}")

    # features & pruning
    centA, _ = centroids_and_areas(A)
    centB, _ = centroids_and_areas(B)
    treeB = cKDTree(centB)
    candidate_lists = treeB.query_ball_point(centA, r=centroid_radius)

    bboxA = bounding_boxes(A) if crop_to_bbox else None
    bboxB = bounding_boxes(B) if crop_to_bbox else None

    # score candidates
    scores = np.zeros((nA, nB), dtype=np.float32)
    for i in range(nA):
        cand = candidate_lists[i]
        if not cand:
            continue
        for j in cand:
            if crop_to_bbox:
                y0, y1, x0, x1 = crop_union_bbox(bboxA, bboxB, i, j)
                Ai = A[i, y0:y1, x0:x1]
                Bj = B[j, y0:y1, x0:x1]
            else:
                Ai, Bj = A[i], B[j]
            dice = best_shifted_dice(Ai, Bj, dilate_radius, max_shift)
            if dice >= dice_thresh:
                scores[i, j] = dice

    # assignment (maximize Dice)
    cost = 1.0 - scores
    row_ind, col_ind = linear_sum_assignment(cost)

    matches, usedB = [], set()
    for i, j in zip(row_ind, col_ind):
        s = float(scores[i, j])
        if s > 0.0:
            dist = float(np.linalg.norm(centA[i] - centB[j]))
            matches.append((int(i), int(j), s, dist))
            usedB.add(int(j))

    matchedA = {m[0] for m in matches}
    unmatched_A = [i for i in range(nA) if i not in matchedA]
    unmatched_B = [j for j in range(nB) if j not in usedB]

    matches.sort(key=lambda t: (-t[2], t[3]))  # best Dice first
    return matches, unmatched_A, unmatched_B, scores

def drd1_cell_match(img_ch2, gcamp_rois):
    
    drd1_cells = mser_detection(img_ch2, delta=2, max_variation=1, min_area=100, max_area=1000)
    matches, unmatched_A, unmatched_B, scores = match_rois(
        drd1_cells, gcamp_rois, 
        centroid_radius=6.0,    # px: prune by centroid distance
        dilate_radius=1,        # px: robustness to tiny shifts
        dice_thresh=0.6,        # minimum Dice to consider “same” cell
        crop_to_bbox=True       # speed boost: compute overlaps in local bbox
    )
    return matches, unmatched_A, unmatched_B, scores

import numpy as np
import matplotlib.pyplot as plt

# ---------- drawing helpers ----------

def _draw_bg(ax, bg, vmin=None, vmax=None):
    if bg is None:
        ax.imshow(np.zeros((512, 512)), cmap="gray", vmin=0, vmax=1)
    else:
        # accept float/uint, 2D or 3D; convert to 2D grayscale if needed
        bg = np.asarray(bg)
        if bg.ndim == 3 and bg.shape[2] in (3, 4):
            # convert RGB/RGBA to grayscale for clean ROI contrast
            bg = np.mean(bg[..., :3], axis=2)
        if vmin is None or vmax is None:
            vmin = np.percentile(bg, 1)
            vmax = np.percentile(bg, 99)
        ax.imshow(bg, cmap="gray", vmin=vmin, vmax=vmax)

def _contour(ax, mask, lw=1.2, color="tab:blue", alpha=1.0):
    # Matplotlib can contour a boolean mask directly
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], linewidths=lw, colors=[color], alpha=alpha)

def _title(ax, s, size=9):
    ax.set_title(s, fontsize=size, pad=2)
    ax.set_axis_off()

# ---------- main visualizers ----------

def visualize_matches(
    A, B, matches,
    background=None,    # 2D or RGB image (same HxW); optional
    ncols=5,
    nrows=5,
    max_matches=None,   # if None → show all, else limit
    colorA="tab:red",
    colorB="tab:green",
    lw=.8,
    alphaA=1.0,
    alphaB=1.0,
):
    """
    Show overlays of matched ROI pairs, paginated into multiple figures.
    Each figure is nrows×ncols (default 5×5 = 25 panels per page).

    Parameters
    ----------
    A, B : (n_rois, H, W) bool arrays
    matches : list of (i, j, dice, dist)
        From match_rois().
    background : array-like, optional
        Mean image or background for context (H×W or H×W×3).
    ncols, nrows : int
        Grid layout per figure.
    max_matches : int or None
        Limit how many matches are displayed in total.
        If None (default), show all.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)

    if len(matches) == 0:
        print("No matches to display.")
        return

    # apply user limit
    if max_matches is not None:
        matches = matches[:max_matches]

    per_page = nrows * ncols
    total = len(matches)
    npages = (total + per_page - 1) // per_page

    for page in range(npages):
        start = page * per_page
        end = min(start + per_page, total)
        subset = matches[start:end]

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.2*nrows), dpi=150)
        axes = np.atleast_2d(axes).ravel()

        for k, (i, j, dice, dist) in enumerate(subset):
            ax = axes[k]
            _draw_bg(ax, background)
            _contour(ax, A[i], lw=lw, color=colorA, alpha=alphaA)
            _contour(ax, B[j], lw=lw, color=colorB, alpha=alphaB)
            _title(ax, f"A[{i}] ↔ B[{j}]\nDice={dice:.2f}, d={dist:.1f}px")

        # hide any extra panels on the last page
        for ax in axes[len(subset):]:
            ax.set_visible(False)

        fig.suptitle(f"Matched ROI overlays — page {page+1}/{npages}", fontsize=12)
        fig.tight_layout()
        plt.show()



def visualize_unmatched(A, idxs, background=None, ncols=6, max_show=48, color="tab:orange", lw=1.4):
    """
    Quickly scan ROIs that didn't find a partner.
    """
    A = np.asarray(A, dtype=bool)
    if len(idxs) == 0:
        print("No unmatched ROIs.")
        return
    idxs = idxs[:max_show]
    n = len(idxs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6*ncols, 2.6*nrows), dpi=150)
    axes = np.atleast_2d(axes).ravel()
    for ax, i in zip(axes, idxs):
        _draw_bg(ax, background)
        _contour(ax, A[i], lw=lw, color=color)
        _title(ax, f"Unmatched {i}")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Unmatched ROIs", fontsize=12)
    fig.tight_layout()
    plt.show()

def visualize_labelmap(stack, background=None, title="ROI label map"):
    """
    Color every ROI with a unique label; later ROIs overwrite earlier where overlapping.
    """
    stack = np.asarray(stack, dtype=bool)
    n, H, W = stack.shape
    label_map = np.zeros((H, W), dtype=np.int32)
    for i in range(n):
        label_map[stack[i]] = i + 1  # 0 = background
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    _draw_bg(ax, background)
    im = ax.imshow(np.where(label_map > 0, label_map, np.nan), cmap="nipy_spectral", alpha=0.6)
    ax.set_axis_off()
    fig.colorbar(im, fraction=0.046, pad=0.04, label="ROI label")
    ax.set_title(title, fontsize=12)
    plt.show()

def visualize_score_heatmap(scores, vmax=None):
    """
    Heatmap of Dice scores used for assignment. Bright = strong match.
    rows = A indices, cols = B indices.
    """
    scores = np.asarray(scores)
    if vmax is None:
        vmax = np.percentile(scores[scores > 0], 99) if np.any(scores > 0) else 1.0
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    im = ax.imshow(scores, aspect="auto", origin="lower", vmin=0, vmax=vmax)
    ax.set_xlabel("B index")
    ax.set_ylabel("A index")
    ax.set_title("Dice score matrix (A×B)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Dice")
    plt.tight_layout()
    plt.show()

#%% test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import xarray as xr

    # --- Load background image
    img_ch2 = np.load(
        r"Z:\Jingyu\2P_Recording\AC310\AC310-20250821\02\suite2p\plane0\ops.npy",
        allow_pickle=True
    ).item()['meanImg_chan2_corrected']

    # --- Detect DRD1+ cells with MSER
    drd1_cells = mser_detection(
        img_ch2,
        delta=2,
        max_variation=1,
        min_area=100,
        max_area=1000
    )

    # --- Quick visualization of MSER mask
    plt.figure()
    mask = np.any(drd1_cells, axis=0)
    plt.imshow(mask, cmap='gray')
    plt.title("DRD1 cells (collapsed mask)")
    plt.show()

    plt.figure()
    plt.imshow(img_ch2, cmap='gray')
    plt.title("Mean image (ch2)")
    plt.show()

    # --- Load Suite2p ROIs (GCaMP channel)
    INT_PATH = r"Z:\Jingyu\2P_Recording\AC310\AC310-20250821\concat"
    A_master = xr.open_dataarray(INT_PATH + r"\A_master.nc").rename(
        {"master_uid": "unit_id"}
    )
    gcamp_rois = np.array(A_master).squeeze()

    # --- Match DRD1 ROIs against GCaMP ROIs
    matches, unA, unB, scores = match_rois(
        drd1_cells, gcamp_rois,
        centroid_radius=6.0,
        dilate_radius=1,
        dice_thresh=0.6,
        crop_to_bbox=True
    )

    # --- Visualize matches
    visualize_matches(
        drd1_cells, gcamp_rois, matches,
        background=img_ch2,
        ncols=5, nrows=5
    )
