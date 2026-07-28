# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 11:48:24 2026

@author: Jingyu Cao

plot rastermap
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from rastermap.rastermap import Rastermap


# -----------------------
# Helpers that mimic GUI
# -----------------------
def gui_sp_transform(S_raw: np.ndarray) -> np.ndarray:
    """
    GUI equivalent:
      sp = zscore(S_raw, axis=1)
      sp = clip to [-4,8], then +4, then /12
    """
    S = zscore(S_raw, axis=1, nan_policy="omit")
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.clip(S, -4, 8) + 4
    S = S / 12.0
    return S.astype(np.float32)


def gui_display_spF(sp: np.ndarray, isort: np.ndarray, tsort = None) -> np.ndarray:
    """
    GUI neural_sorting(i) when i < 2:

    self.spF = gaussian_filter1d(self.sp[np.ix_(self.isort, self.tsort)].T,
                                 sigma=min(8, max(1, int(ncells*0.005))),
                                 axis=1).T
    self.spF = zscore(self.spF, axis=1)
    self.spF = clip to [-4,8], +4, /12
    """
    if tsort is None:
        tsort = np.arange(sp.shape[1], dtype=np.int32)

    sp_sorted = sp[np.ix_(isort.astype(np.int32), tsort.astype(np.int32))]  # (ncells, ntime)

    # IMPORTANT: smoothing across neurons (not time) exactly like GUI:
    # take transpose -> (ntime, ncells), filter axis=1 -> across cells, transpose back
    sigma = int(np.minimum(8, np.maximum(1, int(sp.shape[0] * 0.005))))
    spF = gaussian_filter1d(sp_sorted.T, sigma=sigma, axis=1).T

    # GUI then zscores AGAIN and clips/scales again
    spF = zscore(spF, axis=1, nan_policy="omit")
    spF = np.nan_to_num(spF, nan=0.0, posinf=0.0, neginf=0.0)
    spF = np.clip(spF, -4, 8) + 4
    spF = spF / 12.0
    return spF.astype(np.float32)

def sanitize_for_svd(X, eps=1e-12):
    """
    Make sure X won't generate NaNs during centering/scaling inside Rastermap/SVD.
    - enforce float32
    - replace any non-finite just in case
    - fix zero-variance columns (timepoints/features) by setting their std to 1 via tiny noise or leaving centered
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Detect columns with ~zero variance
    col_std = X.std(axis=0, keepdims=False)
    zero_cols = col_std < eps
    if np.any(zero_cols):
        # Option A (recommended): add tiny noise to those columns so std != 0
        # (doesn't affect visualization; prevents 0/0 inside normalization)
        noise = (1e-6) * np.random.randn(X.shape[0], int(zero_cols.sum())).astype(np.float32)
        X[:, zero_cols] = X[:, zero_cols] + noise

        # Option B (more “pure”): just leave them as-is and they won’t dominate;
        # but if Rastermap divides by std internally, you still need Option A.

    # Final guard
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def plot_raster_matplotlib(img01: np.ndarray, out_png: str, title: str,
                           sat=(0.3, 0.7), cmap="gray_r", ax=None,
                           selected_frames=None, ):
    """
    Mimic GUI "saturation" slider by setting vmin/vmax to sat (default [0.3, 0.7]).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
    
    ax.imshow(img01, aspect="auto", interpolation="nearest",
               cmap=cmap, vmin=sat[0], vmax=sat[1])
    ax.set(xlabel="time (frames)", ylabel="neurons (sorted)",
           title=title, 
           # ylim=(0, img01.shape[0])
           )
    
    if selected_frames is not None:
        ax.vlines(selected_frames, 0, img01.shape[0], colors='orange', alpha=.003)
    if ax is None:
        fig.tight_layout()
        if out_png is not None:
            plt.savefig(out_png, dpi=200)
        else:
            plt.show()
        plt.close()
        
        return fig

def plot_suite2p_raster_map(suite2p_dir, ax=None, out_dir=None):
    """
    Offline (no GUI) pipeline that matches Suite2p GUI rastermap plotting style.

    Activity mode: F - 0.7*Fneu  (GUI activityMode == 2)

    It reproduces GUI behavior:
    - sp = zscore(neuron-wise) -> clip [-4,8] -> rescale to [0,1]
    - Rastermap fit on sp
    - spF display for PC/rastermap:
        reorder -> gaussian_filter1d across NEURONS (axis=1 in transposed array) ->
        zscore again -> clip/rescale -> display with levels (sat) default [0.3,0.7]

    Outputs:
    - rastermap_embedding.npy
    - rastermap_isort_selected.npy
    - rastermap_isort_global.npy
    - rastermap_display_rastermap.png
    - rastermap_display_PC1.png
    """
    
    # -----------------------
    # Paths (edit)
    # -----------------------
    # suite2p_dir = p_suite2p  # contains F.npy, Fneu.npy, (optional) iscell.npy
    F_path      = os.path.join(suite2p_dir, "F.npy")
    Fneu_path   = os.path.join(suite2p_dir, "Fneu.npy")
    iscell_path = os.path.join(suite2p_dir, "iscell.npy")
    
    # out_dir = suite2p_dir
    
    # GUI default display levels:
    sat = (0.3, 0.7)   # same as VisWindow self.sat default
    
    
    # -----------------------
    # Load and select cells
    # -----------------------
    F = np.load(F_path)       # (ncells, ntime)
    Fneu = np.load(Fneu_path)
    
    if os.path.exists(iscell_path):
        iscell = np.load(iscell_path)
        cell_inds = np.where(iscell[:, 0].astype(bool))[0]
    else:
        cell_inds = np.arange(F.shape[0])
    
    # activityMode == 2: F - 0.7*Fneu
    S_raw = (F[cell_inds, :] - 0.7 * Fneu[cell_inds, :]).astype(np.float32)
    
    
    # -----------------------
    # Make sp exactly like GUI (this is what Rastermap fits on)
    # -----------------------
    sp = gui_sp_transform(S_raw)  # (nsel, ntime)
    
    
    # -----------------------
    # Rastermap + PCs (same button in GUI)
    # -----------------------
    model = Rastermap()
    model.fit(sp)
    
    embedding = model.embedding
    isort_rmap = np.argsort(embedding[:, 0]).astype(np.int32)
    
    # PC sorting in GUI uses model.Usv (set in activate())
    # PC1 corresponds to index 0
    pc_index = 0
    if not hasattr(model, "Usv") or model.Usv is None:
        raise RuntimeError("Rastermap model did not return Usv; check rastermap version.")
    isort_pc = np.argsort(model.Usv[:, pc_index]).astype(np.int32)
    
    # Global rank map like GUI
    global_sort = -1 * np.ones((F.shape[0],), dtype=np.int64)
    rank = np.zeros(len(cell_inds), dtype=np.int64)
    rank[isort_rmap] = np.arange(len(cell_inds), dtype=np.int64)
    global_sort[cell_inds] = rank
    
    # np.save(os.path.join(out_dir, "rastermap_embedding.npy"), embedding)
    # np.save(os.path.join(out_dir, "rastermap_isort_selected.npy"), isort_rmap)
    # np.save(os.path.join(out_dir, "rastermap_isort_global.npy"), global_sort)
    
    
    # -----------------------
    # Build DISPLAY image spF like GUI (this is why your background differed)
    # -----------------------
    # GUI default: no time sorting unless checkbox enabled
    tsort = np.arange(sp.shape[1], dtype=np.int32)
    
    spF_rmap = gui_display_spF(sp, isort=isort_rmap, tsort=tsort)
    spF_pc1  = gui_display_spF(sp, isort=isort_pc,   tsort=tsort)
    
    
    # -----------------------
    # Plot (matplotlib only)
    # -----------------------
    if out_dir:
        out_png_rmap = os.path.join(out_dir, "rastermap_display_rastermap.png")
    else:
        out_png_rmap = None
    plot_raster_matplotlib(
        spF_rmap,
        out_png_rmap,
        title="GUI-like display: Rastermap sort (F - 0.7*Fneu)",
        sat=sat,
        ax=ax
    )
    if out_dir: 
        out_png_pc1 = os.path.join(out_dir, "rastermap_display_PC1.png")
    else:
        out_png_pc1 = None
    plot_raster_matplotlib(
        spF_pc1,
        out_png_pc1,
        title="GUI-like display: PC1 sort (F - 0.7*Fneu)",
        sat=sat,
        ax=ax
    )
    
    print("Done.")
    print(f"Selected cells: {len(cell_inds)} / total: {F.shape[0]}")
    print("Saved:")
    print(" -", out_png_rmap)
    print(" -", out_png_pc1)

def fill_nan_per_row_timeinterp(X, *, fallback="zero"):
    """
    Fill NaNs/inf in (n_cells, n_time) array per row (neuron) along time.
    - inf -> NaN
    - interpolate NaNs using linear interpolation in time
    - if a row has <2 valid points:
        fallback="zero": fill with 0
        fallback="median": fill with row median (or 0 if all-NaN)
    Returns a float32 array.
    """
    X = np.asarray(X, dtype=np.float32)
    X = X.copy()

    # convert inf -> nan
    X[~np.isfinite(X)] = np.nan

    n_cells, n_time = X.shape
    t = np.arange(n_time, dtype=np.float32)

    for i in range(n_cells):
        row = X[i]
        good = np.isfinite(row)
        n_good = int(good.sum())

        if n_good == n_time:
            continue

        if n_good >= 2:
            # interpolate only where nan
            row_interp = np.interp(t, t[good], row[good]).astype(np.float32)
            X[i] = row_interp
        else:
            if fallback == "median" and n_good == 1:
                # only one point -> fill everything with that value
                X[i] = row[good][0]
            elif fallback == "median" and n_good == 0:
                X[i] = 0.0
            else:
                # fallback="zero" or anything else
                X[i] = 0.0

    # safety: if anything still nan (shouldn't), zero it
    X[~np.isfinite(X)] = 0.0
    return X

def generate_rastermap(S_raw, out_dir=None):
    # filter out invalid values in the original array
    S_raw = fill_nan_per_row_timeinterp(S_raw, fallback="zero")
    
    # GUI default display levels:
    sat = (0.3, 0.7)   # same as VisWindow self.sat default
    
    # -----------------------
    # Make sp exactly like GUI (this is what Rastermap fits on)
    # -----------------------
    sp = gui_sp_transform(S_raw)  # (nsel, ntime)
    
    # IMPORTANT: prevent Rastermap internals from producing NaNs
    sp = sanitize_for_svd(sp)
    
    # -----------------------
    # Rastermap + PCs (same button in GUI)
    # -----------------------
    model = Rastermap()
    model.fit(sp)
    
    embedding = model.embedding
    isort_rmap = np.argsort(embedding[:, 0]).astype(np.int32)
    
    # PC sorting in GUI uses model.Usv (set in activate())
    # PC1 corresponds to index 0
    pc_index = 0
    if not hasattr(model, "Usv") or model.Usv is None:
        raise RuntimeError("Rastermap model did not return Usv; check rastermap version.")
    isort_pc = np.argsort(model.Usv[:, pc_index]).astype(np.int32)
    
    # Global rank map like GUI
    n_cells = S_raw.shape[0]
    # global_sort = -1 * np.ones((n_cells,), dtype=np.int64)
    rank = np.zeros(n_cells, dtype=np.int64)
    rank[isort_rmap] = np.arange(n_cells, dtype=np.int64)
    global_sort = rank
    
    if out_dir is not None:
        np.save(os.path.join(out_dir, "rastermap_embedding.npy"), embedding)
        np.save(os.path.join(out_dir, "rastermap_isort_selected.npy"), isort_rmap)
        np.save(os.path.join(out_dir, "rastermap_isort_global.npy"), global_sort)
    
    # -----------------------
    # Build DISPLAY image spF like GUI (this is why your background differed)
    # -----------------------
    # GUI default: no time sorting unless checkbox enabled
    tsort = np.arange(sp.shape[1], dtype=np.int32)
    
    spF_rmap = gui_display_spF(sp, isort=isort_rmap, tsort=tsort)
    # spF_pc1  = gui_display_spF(sp, isort=isort_pc,   tsort=tsort)
    
    return spF_rmap
    
def plot_rastermap(S_raw=None, spF_rmap=None, out_dir=None, ax=None, prefix=None, 
                   selected_frames=None,
                   # GUI default display levels:
                   sat = (0.3, 0.7)   # same as VisWindow self.sat default
                   ):
    """
    Offline (no GUI) pipeline that matches Suite2p GUI rastermap plotting style.

    Activity mode: F - 0.7*Fneu  (GUI activityMode == 2)

    It reproduces GUI behavior:
    - sp = zscore(neuron-wise) -> clip [-4,8] -> rescale to [0,1]
    - Rastermap fit on sp
    - spF display for PC/rastermap:
        reorder -> gaussian_filter1d across NEURONS (axis=1 in transposed array) ->
        zscore again -> clip/rescale -> display with levels (sat) default [0.3,0.7]

    Outputs:
    - rastermap_embedding.npy
    - rastermap_isort_selected.npy
    - rastermap_isort_global.npy
    - rastermap_display_rastermap.png
    - rastermap_display_PC1.png
    """
    if spF_rmap is None:
        if S_raw is not None:
            generate_rastermap(S_raw, out_dir)
        else:
            print('input raw data or rastermap')
            return 
    #     # filter out invalid values in the original array
    #     S_raw = fill_nan_per_row_timeinterp(S_raw, fallback="zero")
        
    #     # GUI default display levels:
    #     sat = (0.3, 0.7)   # same as VisWindow self.sat default
        
    #     # -----------------------
    #     # Make sp exactly like GUI (this is what Rastermap fits on)
    #     # -----------------------
    #     sp = gui_sp_transform(S_raw)  # (nsel, ntime)
        
    #     # IMPORTANT: prevent Rastermap internals from producing NaNs
    #     sp = sanitize_for_svd(sp)
        
    #     # -----------------------
    #     # Rastermap + PCs (same button in GUI)
    #     # -----------------------
    #     model = Rastermap()
    #     model.fit(sp)
        
    #     embedding = model.embedding
    #     isort_rmap = np.argsort(embedding[:, 0]).astype(np.int32)
        
    #     # PC sorting in GUI uses model.Usv (set in activate())
    #     # PC1 corresponds to index 0
    #     pc_index = 0
    #     if not hasattr(model, "Usv") or model.Usv is None:
    #         raise RuntimeError("Rastermap model did not return Usv; check rastermap version.")
    #     isort_pc = np.argsort(model.Usv[:, pc_index]).astype(np.int32)
        
    #     # Global rank map like GUI
    #     n_cells = S_raw.shape[0]
    #     # global_sort = -1 * np.ones((n_cells,), dtype=np.int64)
    #     rank = np.zeros(n_cells, dtype=np.int64)
    #     rank[isort_rmap] = np.arange(n_cells, dtype=np.int64)
    #     # global_sort = rank
        
    #     # np.save(os.path.join(out_dir, "rastermap_embedding.npy"), embedding)
    #     # np.save(os.path.join(out_dir, "rastermap_isort_selected.npy"), isort_rmap)
    #     # np.save(os.path.join(out_dir, "rastermap_isort_global.npy"), global_sort)
        
        
    #     # -----------------------
    #     # Build DISPLAY image spF like GUI (this is why your background differed)
    #     # -----------------------
    #     # GUI default: no time sorting unless checkbox enabled
    #     tsort = np.arange(sp.shape[1], dtype=np.int32)
        
    #     spF_rmap = gui_display_spF(sp, isort=isort_rmap, tsort=tsort)
    #     # spF_pc1  = gui_display_spF(sp, isort=isort_pc,   tsort=tsort)
        
    #     return spF_rmap
    
    # -----------------------
    # Plot (matplotlib only)
    # -----------------------
    if out_dir:
        out_png_rmap = os.path.join(out_dir, "{}rastermap.png".format(prefix))
    else:
        out_png_rmap = None
    plot_raster_matplotlib(
        spF_rmap,
        out_png_rmap,
        title="{}Rastermap sort".format(prefix),
        sat=sat,
        ax=ax,
        selected_frames=selected_frames
    )
    # if out_dir: 
    #     out_png_pc1 = os.path.join(out_dir, "rastermap_display_PC1.png")
    # else:
    #     out_png_pc1 = None
    # plot_raster_matplotlib(
    #     spF_pc1,
    #     out_png_pc1,
    #     title="GUI-like display: PC1 sort (F - 0.7*Fneu)",
    #     sat=sat,
    #     ax=ax
    # )
    
    print("Done.")
    # print(f"Selected cells: {n_cells} / total: {n_cells}")
    print("Saved:")
    print(" -", out_png_rmap)
    # print(" -", out_png_pc1)
    
#%%
if __name__ == "__main__":
    rec = r'AC989-20250625-02'
    anm_id, date, s = rec.split('-')
    p_suite2p = r"Z:\Jingyu\2P_Recording\{}\{}-{}\{}\suite2p_func_detec\plane0".format(
    anm_id, anm_id, date, s)
    F_raw = np.load(p_suite2p+r'\F.npy')
    
    # plot_suite2p_raster_map(p_suite2p, out_dir=None)
    plot_rastermap(F_raw, out_dir=None, selected_frames=np.arange(500, 700))
