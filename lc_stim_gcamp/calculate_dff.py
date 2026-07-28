# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:28:26 2026

@author: Jingyu Cao
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from common.utils_imaging import percentile_dff
from common.mask.generate_masks import select_gcamp_rois

# def find_frame_cutoff(F_mean):
#     x = np.sort(F_mean)

#     gaps = np.diff(x)
    
#     idx = np.argmax(gaps)
    
#     threshold = (x[idx] + x[idx + 1]) / 2
    
#     return threshold


def find_frame_cutoff(F_mean, search_percentile=25, min_gap=1):
    """
    Find the largest discontinuity in the low-value portion of F_mean.

    Parameters
    ----------
    F_mean : array-like
        Mean fluorescence for each frame.
    search_percentile : float
        Only consider gaps whose lower value is below this percentile.
    min_gap : float
        Minimum acceptable gap size.

    Returns
    -------
    threshold : float
        Midpoint of the selected gap.
    """
    x = np.asarray(F_mean)
    x = x[np.isfinite(x)]

    if x.size < 2:
        raise ValueError("F_mean must contain at least two finite values.")

    x = np.sort(x)
    gaps = np.diff(x)

    search_limit = np.percentile(x, search_percentile)
    candidate_idx = np.where(x[:-1] <= search_limit)[0]

    if candidate_idx.size == 0:
        raise ValueError("No candidate gaps found in the search range.")

    idx = candidate_idx[np.argmax(gaps[candidate_idx])]

    if gaps[idx] < min_gap:
        raise ValueError(
            f"No clear discontinuity found; largest candidate gap is "
            f"{gaps[idx]:.3f}."
        )

    threshold = (x[idx] + x[idx + 1]) / 2

    print(
        f"Shutter cutoff: {threshold:.3f} "
        f"(gap: {x[idx]:.3f} → {x[idx + 1]:.3f})"
    )

    return threshold

DEFAULT_DATA_BASE = (
    r"Z:\Jingyu\2P_Recording\{anm}\{anm}-{date}\{ss}\suite2p_func_detec\plane0"
)

DEFAULT_SOMA_RES = (
    r"Z:\Jingyu\raw_data\lc_stim_gcamp\processed_data\{rec}\soma_roi_selection"
)

DEFAULT_OVERWRITE = {
    "F_corr": False,
    "shutter_mask": False,
    "soma_label": False,
    "dff_ch1": False,
    "dff_ch1_soma": False,
}

def process_F_trace(
    rec,
    neu_factor = 0.7, # factor for neuropil correction
    save_path=None,
    data_base_config=DEFAULT_DATA_BASE,
    soma_res_config=DEFAULT_SOMA_RES,
    overwrite=None,
    active_soma_only=False
):
    print(f"\nProcessing {rec}")
    
    if overwrite is None:
        overwrite = DEFAULT_OVERWRITE.copy()
    else:
        overwrite = {**DEFAULT_OVERWRITE, **overwrite}
        
    anm, date, ss = rec.split("-")

    data_base = Path(
        data_base_config.format(anm=anm, date=date, ss=ss)
    )

    if save_path is None:
        save_path = data_base

    soma_res_path = Path(
        soma_res_config.format(rec=rec)
    )

    # ------------------------------------------------------------------
    # Neuropil correction
    # ------------------------------------------------------------------
    if (not (save_path / "F_corr_ch1.npy").exists()) or overwrite["F_corr"]:
        print("  Correcting neuropil...")
        F_raw = np.load(data_base / "F.npy")
        F_neu = np.load(data_base / "Fneu.npy")
        Fc = F_raw - neu_factor * F_neu
        np.save(save_path / "F_corr_ch1.npy", Fc)
    else:
        print("  Loading corrected fluorescence...")
        Fc = np.load(save_path / "F_corr_ch1.npy")

    # ------------------------------------------------------------------
    # Shutter mask
    # ------------------------------------------------------------------
    if (not (save_path / "shutter_mask.npy").exists()) or overwrite["shutter_mask"]:
        print("  Detecting shutter off frames...")
        F_raw = np.load(data_base/r"F.npy")
        F_roi_mean = np.nanmean(F_raw, axis=0)
        shutter_masks = F_roi_mean < 1
        # shutter_masks = F_roi_mean < find_frame_cutoff(F_roi_mean)
        kept_frames = ~shutter_masks
        np.save(save_path / "shutter_mask.npy", shutter_masks)
    else:
        print("  Loading shutter mask...")
        shutter_masks = np.load(save_path / "shutter_mask.npy")
        kept_frames = ~shutter_masks
    print(f'{np.sum(shutter_masks)} shutter off frames detected, kept_frames={np.sum(kept_frames)}')

    # ------------------------------------------------------------------
    # Soma selection
    # ------------------------------------------------------------------
    if (not (soma_res_path / "soma_label.npz").exists()) or overwrite["soma_label"]:
        print("  Selecting soma ROIs...")
        soma_res_path.mkdir(parents=True, exist_ok=True)

        suite2p_ops = np.load(
            data_base / "ops.npy", allow_pickle=True
        ).item()
        mean_img_ch1 = suite2p_ops["meanImg"]

        F_corr = np.load(save_path / "F_corr_ch1.npy")
        gcamp_stats = np.load(
            data_base / "stat.npy", allow_pickle=True
        )

        is_soma, is_active, is_active_soma = select_gcamp_rois(
            mean_img_ch1,
            F_corr,
            gcamp_stats,
            path_result=soma_res_path,
            thresholds={"hollowness_threshold_min": 0.1},
        )
    else:
        is_active_soma = np.load(soma_res_path / "soma_label.npz")['is_active_soma']
        print("  Soma labels found.")

    # ------------------------------------------------------------------
    # dF/F calculation
    # ------------------------------------------------------------------
    if active_soma_only:
        if (not (save_path / "dff_ch1_soma.npy").exists()) or overwrite["dff_ch1_soma"]:
            print("  Computing dF/F...")
            Fc = Fc[is_active_soma, :]
            
            dff_kept, baseline_kept = percentile_dff(
                Fc[:, kept_frames],
                return_baseline=True,
            )

            dff = np.full(Fc.shape, np.nan)
            dff[:, kept_frames] = dff_kept

            baseline = np.full(Fc.shape, np.nan)
            baseline[:, kept_frames] = baseline_kept

            np.save(save_path / "dff_ch1_soma.npy", dff)
            np.save(save_path / "baseline_ch1_soma.npy", baseline)
        else:
            print("  Loading dF/F for active soma rois...")
            dff = np.load(save_path / "dff_ch1_soma.npy")
    
    else:
        if (not (save_path / "dff_ch1.npy").exists()) or overwrite["dff_ch1"]:
            print("  Computing dF/F...")
    
            dff_kept, baseline_kept = percentile_dff(
                Fc[:, kept_frames],
                return_baseline=True,
            )
    
            dff = np.full(Fc.shape, np.nan)
            dff[:, kept_frames] = dff_kept
    
            baseline = np.full(Fc.shape, np.nan)
            baseline[:, kept_frames] = baseline_kept
    
            np.save(save_path / "dff_ch1.npy", dff)
            np.save(save_path / "baseline_ch1.npy", baseline)
        else:
            print("  Loading dF/F...")
            dff = np.load(save_path / "dff_ch1.npy")

    print("Done.")
    
    return dff, is_active_soma, shutter_masks


if __name__ == "__main__":
    
    rec_lst = [
    'AC333-20260726-02',
    'AC333-20260726-04',
    'AC334-20260726-02'
        ]
    
    for rec in rec_lst:
        dff, is_active_soma, shutter_masks = process_F_trace(rec,
                                                             active_soma_only=True,
                                                             # overwrite=True
                                                             )
