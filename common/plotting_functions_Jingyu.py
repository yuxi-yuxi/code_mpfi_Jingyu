# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 19:20:24 2025

@author: Jingyu Cao
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
if ("Z:\Jingyu\Code\Python" in sys.path) == False:
    sys.path.append("Z:\Jingyu\Code\Python")
import utils_Jingyu as utl
# import plotting_functions_Jingyu as pf
from utils_Jingyu import align_trials, seperate_bad_trial, seperate_eraly_late_lick_trial
from scipy.stats import wilcoxon, ranksums, ttest_rel, ttest_ind, sem
# from common import mpl_formatting

def mpl_formatting(): # from Dinghao, for plotting editable pdf with matplot
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'Arial',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
        })
mpl_formatting() 

def save_fig(fig, OUT_DIR_FIG, fig_name='', save=True,
             forms=None,
             dpi=300):
    if forms is None:
        forms = ['png', 'pdf',]
    fig.tight_layout()
    if save:
        OUT_DIR_FIG = Path(OUT_DIR_FIG)
        for form in forms:
            plt.savefig(OUT_DIR_FIG/f'{fig_name}.{form}',
                        dpi=dpi)
    else:
        plt.show()
    plt.close()

def plot_cdf_profiles(profile_a, profile_b,
                      label_a='profile_a', label_b='profile_b',
                      colors = ['steelblue', 'orange'],
                      xlabel = 'value', ylabel='CDF',
                      ax=None, title='CDF',
                      lw=1):
    """
    Plot empirical CDFs of two 1D profiles using matplotlib.

    Parameters
    ----------
    profile_a : array-like
        First profile values.
    profile_b : array-like
        Second profile values.
    label_a : str
        Label for profile_a.
    label_b : str
        Label for profile_b.
    ax : matplotlib.axes.Axes or None
        Existing axis to plot on. If None, create a new figure.
    title : str
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    profile_a = np.asarray(profile_a).ravel()
    profile_b = np.asarray(profile_b).ravel()

    # remove NaNs
    profile_a = profile_a[~np.isnan(profile_a)]
    profile_b = profile_b[~np.isnan(profile_b)]

    # sort values for empirical CDF
    x_a = np.sort(profile_a)
    y_a = np.arange(1, len(x_a) + 1) / len(x_a)

    x_b = np.sort(profile_b)
    y_b = np.arange(1, len(x_b) + 1) / len(x_b)

    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2), dpi=150)
    else:
        fig = ax.figure

    ax.plot(x_a, y_a, label=label_a, linewidth=lw, color=colors[0])
    ax.plot(x_b, y_b, label=label_b, linewidth=lw, color=colors[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    return fig, ax    

def plot_two_traces_with_binned_stats(profile_a, profile_b, 
                                      bef, aft,
                                      ax=None,
                                      test='ranksum',
                                      time_windows=[(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                                      baseline_window=(-0.5, 0),
                                      labels = ['a', 'b'],
                                      colors = ['steelblue', 'orange'],
                                      sample_freq=30,
                                      figsize=(3, 3),
                                      # --- new options for scalebar ---
                                      show_scalebar=False,
                                      scalebar_time=1.0,       # seconds
                                      scalebar_dff=0.1,        # ΔF/F units
                                      scalebar_loc='lower right',  # 'lower/upper left/right'
                                      scalebar_color='k',
                                      scalebar_fs=8,
                                      scalebar_pad_frac=0.05,
                                      hide_axes_when_scalebar=True,
                                      print_median_iqr=False,
                                      ):
    """
    Plot two mean traces with SEM, run stats in time windows, and optionally
    show a separate time / dF/F scalebar instead of the original axes.
    """
    results = []

    # perfom statistic test for each bin
    if baseline_window is not None:
        win_slice = slice(int((bef + baseline_window[0]) * sample_freq),
                          int((bef + baseline_window[1]) * sample_freq))
        profile_a_baseline = np.nanmean(np.vstack(profile_a)[:, win_slice], axis=-1)
        profile_b_baseline = np.nanmean(np.vstack(profile_b)[:, win_slice], axis=-1)
    else:
        profile_a_baseline = 0
        profile_b_baseline = 0
        
    for window in time_windows:
        # Convert time window to frame indices
        win_slice = slice(int((bef + window[0]) * sample_freq),
                          int((bef + window[1]) * sample_freq))
        
        # Segment and average each condition
        profile_a_seg = np.vstack(profile_a)[:, win_slice]
        profile_b_seg = np.vstack(profile_b)[:, win_slice]
    
        profile_a_mean = np.nanmean(profile_a_seg, axis=-1)
        profile_b_mean = np.nanmean(profile_b_seg, axis=-1)
        
        profile_a_mean = profile_a_mean - profile_a_baseline
        profile_b_mean = profile_b_mean - profile_b_baseline

        # Perform tests
        tests = {
            "ind_ttest": ttest_ind(profile_a_mean, profile_b_mean, nan_policy='omit'),
            "ranksum": ranksums(profile_a_mean, profile_b_mean),
        }

        # Median + IQR per group
        a_med = np.nanmedian(profile_a_mean)
        a_q1, a_q3 = np.nanpercentile(profile_a_mean, [25, 75])
        b_med = np.nanmedian(profile_b_mean)
        b_q1, b_q3 = np.nanpercentile(profile_b_mean, [25, 75])

        if print_median_iqr:
            print(f"[{window[0]}–{window[1]} s] "
                  f"{labels[0]}: median={a_med:.4f}, IQR=[{a_q1:.4f}, {a_q3:.4f}] (n={np.sum(~np.isnan(profile_a_mean))}) | "
                  f"{labels[1]}: median={b_med:.4f}, IQR=[{b_q1:.4f}, {b_q3:.4f}] (n={np.sum(~np.isnan(profile_b_mean))})")

        # Store results
        results.append({
            "window": f"{window[0]}–{window[1]} s",
            f"{labels[0]}_median": a_med,
            f"{labels[0]}_q1": a_q1,
            f"{labels[0]}_q3": a_q3,
            f"{labels[1]}_median": b_med,
            f"{labels[1]}_q1": b_q1,
            f"{labels[1]}_q3": b_q3,
            **{k + "_stat": v.statistic for k, v in tests.items()},
            **{k + "_pval": v.pvalue for k, v in tests.items()},
        })
        
        # fig, ax = plot_unpaired_violin(profile_a_mean, profile_b_mean,
        #                        colors=('firebrick', 'lightcoral'),
        #                        m_point = 'median',
        #                        colname=['DA-UP\nPyrUp', 'non-DA-Up\nPyrUp'], 
        #                        ylabel='mean_dFF', ylim=None,
        #                        title_prefix=f"{window[0]}–{window[1]} s\n mean_diff={np.mean(profile_a_mean)-np.mean(profile_b_mean)}\n",
        #                        )

    # Convert to DataFrame
    df_stats = pd.DataFrame(results)
    
    pad_frac = 0.02
    fs = 8
    color = 'k'
    lw = 1
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
    else:
        fig = ax.figure
    
    xaxis = np.arange(sample_freq * (bef + aft)) / sample_freq - bef

    # Plot traces
    if baseline_window is not None:
        plot_mean_trace(np.vstack(profile_a) - profile_a_baseline[:, None],
                           ax, xaxis, color=colors[0], label=f'{labels[0]}')
        plot_mean_trace(np.vstack(profile_b) - profile_b_baseline[:, None],
                           ax, xaxis, color=colors[1], label=f'{labels[1]}')
    else:
        plot_mean_trace(np.vstack(profile_a), ax, xaxis, color=colors[0], label=f'{labels[0]}')
        plot_mean_trace(np.vstack(profile_b), ax, xaxis, color=colors[1], label=f'{labels[1]}')

    # p-value annotation
    col = f"{test}_pval"
    if col not in df_stats.columns:
        raise ValueError(f"Column '{col}' not found in df_stats. Available columns: {list(df_stats.columns)}")

    pvals = df_stats[col].to_numpy()

    # Adjust ylim to make space (extra rows below bracket for median/IQR if requested)
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    n_extra_rows = 2 if print_median_iqr else 0
    extra = (len(time_windows) + 1 + n_extra_rows) * pad_frac * yr
    ax.set_ylim(y0, y1 + extra)
    y0, y1 = ax.get_ylim()   # refresh after change
    yr = y1 - y0
    base_top = y1 - extra * 0.2  # slightly inside

    for i, (w, p) in enumerate(zip(time_windows, pvals)):
        # y = base_top - i * pad_frac * yr
        y = base_top
        x0_w, x1_w = w[0] + 0.02, w[1] - 0.02
        ax.plot([x0_w, x1_w], [y, y], color=color, lw=lw)
        xm = 0.5 * (x0_w + x1_w)
        txt = f"p={p:.4f}" if np.isfinite(p) else "n/a"
        ax.text(xm, y, txt, ha='center', va='bottom', fontsize=fs, color=color)

        # Optional median/IQR annotation directly under the bracket
        if print_median_iqr:
            row = df_stats.iloc[i]
            a_txt = (f"{row[f'{labels[0]}_median']:.3f}\n"
                     f"[{row[f'{labels[0]}_q1']:.3f}, {row[f'{labels[0]}_q3']:.3f}]")
            b_txt = (f"{row[f'{labels[1]}_median']:.3f}\n"
                     f"[{row[f'{labels[1]}_q1']:.3f}, {row[f'{labels[1]}_q3']:.3f}]")
            y_a = y - 1.2 * pad_frac * yr
            y_b = y_a - 2.4 * pad_frac * yr
            ax.text(xm, y_a, a_txt, ha='center', va='top',
                    fontsize=fs - 1, color=colors[0])
            ax.text(xm, y_b, b_txt, ha='center', va='top',
                    fontsize=fs - 1, color=colors[1])
        
    ax.text(0.5, y + 0.1 * yr, test, ha='center', va='bottom', fontsize=fs, color=color)
    ax.text(0.5, y + 0.2 * yr, f'baseline={baseline_window}', ha='center', va='bottom', fontsize=fs, color=color)

    # Clean spines
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower left')
    
    ax.axvline(0, 0, c='grey', ls='--')
    # ------------------------------------------------------------------
    # Optional scalebar (time + dF/F) and hiding original axes
    # ------------------------------------------------------------------
    if show_scalebar:
        # Optionally hide axes info
        if hide_axes_when_scalebar:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            # Hide all spines so only traces + scalebar remain
            for spine in ax.spines.values():
                spine.set_visible(False)
            
        # Get current limits after all annotations
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        dx = x_max - x_min
        dy = y_max - y_min

        # Pad in data units
        pad_x = scalebar_pad_frac * dx
        pad_y = scalebar_pad_frac * dy

        # Determine origin of scalebar based on requested location
        loc = scalebar_loc.lower().replace(' ', '')
        if loc == 'lowerleft':
            x0_sb = x_min + pad_x
            y0_sb = y_min + pad_y
        elif loc == 'lowerright':
            x0_sb = x_max - pad_x - scalebar_time
            y0_sb = y_min + pad_y
        elif loc == 'upperleft':
            x0_sb = x_min + pad_x
            y0_sb = y_max - pad_y - scalebar_dff
        elif loc == 'upperright':
            x0_sb = x_max - pad_x - scalebar_time
            y0_sb = y_max - pad_y - scalebar_dff
        else:
            # Fallback: lower right
            x0_sb = x_max - pad_x - scalebar_time
            y0_sb = y_min + pad_y

        # Draw time (horizontal) and dF/F (vertical) bars
        ax.plot([x0_sb, x0_sb + scalebar_time],
                [y0_sb, y0_sb],
                color=scalebar_color, lw=1.5)
        ax.plot([x0_sb, x0_sb],
                [y0_sb, y0_sb + scalebar_dff],
                color=scalebar_color, lw=1.5)

        # Labels
        ax.text(x0_sb + scalebar_time / 2.0,
                y0_sb - 0.03 * dy,
                f'{scalebar_time:g} s',
                ha='center', va='top',
                fontsize=scalebar_fs, color=scalebar_color)

        ax.text(x0_sb - 0.02 * dx,
                y0_sb + scalebar_dff / 2.0,
                f'{scalebar_dff:g} ΔF/F',
                ha='right', va='center',
                fontsize=scalebar_fs, color=scalebar_color, rotation=90)

    return fig, ax


def plot_bar_with_paired_scatter(
        ax, ctrl_vals, stim_vals, colors=('grey', 'firebrick'),
        title='', ylabel='% cells', xticklabels=('ctrl.', 'stim.'),
        ylim=None,
        print_median_iqr=False,
    ):
    """Paired bars with lines, auto-ylim, top-anchored stats, and mean±SEM labels."""

    def sigstars(p):
        return ('ns' if p >= 0.05 else
                ('*' if p >= 0.01 else
                 ('**' if p >= 0.001 else
                  ('***' if p >= 1e-4 else '****'))))

    def annotate_sig(ax, x1, x2, y, text, bump):
        ax.plot([x1, x1, x2, x2], [y-0.5*bump, y, y, y-0.5*bump],
                lw=0.8, color='k')
        ax.text((x1+x2)/2, y + 0.5*bump, text, ha='center',
                va='bottom', fontsize=6)

    def annotate_bar(ax, x, y, text, bump):
        ax.text(x, y + bump, text, ha='center', va='bottom', fontsize=6)

    # sanitize inputs (paired, drop any row with NaN in either)
    ctrl_vals = np.asarray(ctrl_vals, dtype=float)
    stim_vals = np.asarray(stim_vals, dtype=float)
    mask = np.isfinite(ctrl_vals) & np.isfinite(stim_vals)
    ctrl = ctrl_vals[mask]
    stim = stim_vals[mask]
    assert len(ctrl) == len(stim) and len(ctrl) > 0, 'need ≥1 paired finite values'

    # bars: mean ± sem
    means = np.array([np.nanmean(ctrl), np.nanmean(stim)], dtype=float)
    errs  = np.array([sem(ctrl, nan_policy='omit'),
                      sem(stim, nan_policy='omit')], dtype=float)
    errs  = np.nan_to_num(errs, nan=0.0, posinf=0.0, neginf=0.0)

    # draw bars
    barc = ax.bar([0, 1], means, yerr=errs, capsize=2, width=0.6, edgecolor='none',
                  alpha=.6, zorder=2,
                  error_kw={'elinewidth': 0.6, 'capthick': 0.6, 'ecolor': 'k'})
    barc.patches[0].set_facecolor(colors[0])
    barc.patches[1].set_facecolor(colors[1])

    # paired points + connecting lines (no jitter)
    for y0, y1 in zip(ctrl, stim):
        ax.plot([0, 1], [y0, y1], lw=0.6, color='k', alpha=.3, zorder=3)
    ax.scatter(np.zeros(len(ctrl)), ctrl, s=8, color=colors[0],
               edgecolor='none', alpha=.5, zorder=4)
    ax.scatter(np.ones(len(stim)),  stim, s=8, color=colors[1],
               edgecolor='none', alpha=.5, zorder=4)
    # ax.scatter(np.zeros(len(ctrl)), ctrl, s=8, color='grey',
    #            edgecolor='none', alpha=.5, zorder=4)
    # ax.scatter(np.ones(len(stim)),  stim, s=8, color='grey',
    #            edgecolor='none', alpha=.5, zorder=4)

    # stats (paired + a couple of unpaired references)
    try:
        w_stat, w_p = wilcoxon(ctrl, stim, alternative='two-sided',
                               zero_method='wilcox', mode='auto')
    except ValueError:
        w_stat, w_p = np.nan, 1.0
    t_stat, t_p       = ttest_rel(ctrl, stim)
    ranksum_stat, ranksum_p = ranksums(ctrl, stim)
    t_ind_stat, t_ind_p     = ttest_ind(ctrl, stim)

    # compute auto y-lims if none specified
    top_data_val = np.nanmax([np.nanmax(ctrl), np.nanmax(stim)])
    top_bar_val  = np.nanmax(means + errs)
    top_y        = float(np.nanmax([top_data_val, top_bar_val]))

    if ylim is None:
        data_min = float(np.nanmin([np.nanmin(ctrl), np.nanmin(stim),
                                    means.min() - errs.max()]))
        y_min0 = min(0.0, data_min)         # keep 0 as floor for % plots unless negatives exist
        y_max0 = top_y
        base_range = max(1e-6, y_max0 - y_min0)
        pad_top = 0.28 * base_range          # generous headroom for multiple lines
        pad_bot = 0.05 * base_range
        ylims = (0, y_max0 + pad_top)
    else:
        ylims = ylim

    # apply axes settings
    ax.set(xticks=[0, 1], xticklabels=xticklabels,
           ylabel=ylabel, title=title, ylim=ylims)
    for s in ['top','right']:
        ax.spines[s].set_visible(False)

    # bump sizes (for offsets)
    yrange = ylims[1] - ylims[0]
    bump = 0.02 * yrange

    # mean ± SEM text just above each bar
    for i, (m, e) in enumerate(zip(means, errs)):
        annotate_bar(ax, i, m + e, f"{m:.3f} ± {e:.3f}", bump * 0.8)

    # optional median + IQR annotation below the x-axis (matches bar color)
    if print_median_iqr:
        med_c = np.nanmedian(ctrl); q1_c, q3_c = np.nanpercentile(ctrl, [25, 75])
        med_s = np.nanmedian(stim); q1_s, q3_s = np.nanpercentile(stim, [25, 75])
        print(f"{xticklabels[0]}: median={med_c:.4f}, IQR=[{q1_c:.4f}, {q3_c:.4f}] (n={len(ctrl)}) | "
              f"{xticklabels[1]}: median={med_s:.4f}, IQR=[{q1_s:.4f}, {q3_s:.4f}] (n={len(stim)})")
        trans = ax.get_xaxis_transform()
        for xi, (med, q1_v, q3_v, c) in enumerate([
            (med_c, q1_c, q3_c, colors[0]),
            (med_s, q1_s, q3_s, colors[1]),
        ]):
            ax.text(xi, -0.08, f"med={med:.3f}\n[{q1_v:.3f}, {q3_v:.3f}]",
                    transform=trans, ha='center', va='top',
                    fontsize=6, color=c, clip_on=False)

    # significance labels at the top (fixed offsets from axis top)
    y1 = ylims[1] - 1.0*bump
    y2 = y1       - 1.2*bump
    y3 = y2       - 1.2*bump
    y4 = y3       - 1.2*bump
    annotate_sig(ax, 0, 1, y1, f"Wilcoxon p={w_p:.3g} ({sigstars(w_p)})", bump)
    annotate_sig(ax, 0, 1, y2, f"paired t-test p={t_p:.3g} ({sigstars(t_p)})", bump)
    annotate_sig(ax, 0, 1, y3, f"t-test ind p={t_ind_p:.3g} ({sigstars(t_ind_p)})", bump)
    annotate_sig(ax, 0, 1, y4, f"ranksum p={ranksum_p:.3g} ({sigstars(ranksum_p)})", bump)

    # Bring axes, labels, spines, error bars, and annotations to front for Illustrator editing
    front_zorder = 10
    for spine in ax.spines.values():
        spine.set_zorder(front_zorder)
    ax.xaxis.set_zorder(front_zorder)
    ax.yaxis.set_zorder(front_zorder)
    for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        label.set_zorder(front_zorder)
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_zorder(front_zorder)
    ax.xaxis.label.set_zorder(front_zorder)
    ax.yaxis.label.set_zorder(front_zorder)
    ax.title.set_zorder(front_zorder)
    for text in ax.texts:
        text.set_zorder(front_zorder)
    for line in ax.lines:
        line.set_zorder(front_zorder)
    for container in ax.containers:
        if hasattr(container, 'errorbar') and container.errorbar is not None:
            for item in container.errorbar:
                if item is not None:
                    if hasattr(item, '__iter__'):
                        for line in item:
                            if line is not None and hasattr(line, 'set_zorder'):
                                line.set_zorder(front_zorder)
                    elif hasattr(item, 'set_zorder'):
                        item.set_zorder(front_zorder)

    return {
        'wilcoxon':   {'stat': float(w_stat),    'p': float(w_p),       'n': int(len(ctrl))},
        'ttest_rel':  {'stat': float(t_stat),    'p': float(t_p),       'n': int(len(ctrl))},
        'ranksums':   {'stat': float(ranksum_stat), 'p': float(ranksum_p),
                       'n1': int(len(ctrl)), 'n2': int(len(stim))},
        'ttest_ind':  {'stat': float(t_ind_stat),   'p': float(t_ind_p),
                       'n1': int(len(ctrl)), 'n2': int(len(stim))},
        'means': means.tolist(),
        'sems':  errs.tolist(),
        'ylim':  list(ylims)
    }

def plot_bar_with_unpaired_scatter(
        ax, ctrl_vals, stim_vals,
        colors=('grey', 'firebrick'),
        title='', ylabel='% cells',
        xticklabels=('ctrl.', 'stim.'),
        ylim=None,
        jitter=0.08, point_size=8,
        use_welch=True,
        seed=0,
        print_median_iqr=False,
    ):
    """
    Unpaired version with automatic ylim, robust annotation placement,
    and mean±SEM text above each bar.
    """
    def sigstars(p):
        return ('ns' if p >= 0.05 else
                ('*' if p >= 0.01 else
                 ('**' if p >= 0.001 else
                  ('***' if p >= 1e-4 else '****'))))

    def annotate_bar(ax, x, y, text, bump):
        ax.text(x, y + bump, text, ha='center', va='bottom', fontsize=6)

    def annotate_sig(ax, x1, x2, y, text, bump):
        ax.plot([x1, x1, x2, x2], [y-0.5*bump, y, y, y-0.5*bump],
                lw=0.8, color='k')
        ax.text((x1+x2)/2, y + 0.5*bump, text, ha='center',
                va='bottom', fontsize=6)

    # sanitize inputs
    ctrl_vals = np.asarray(ctrl_vals, float)
    stim_vals = np.asarray(stim_vals, float)
    ctrl = ctrl_vals[np.isfinite(ctrl_vals)]
    stim = stim_vals[np.isfinite(stim_vals)]
    assert (len(ctrl) > 0) and (len(stim) > 0), 'need ≥1 finite value in each group'

    # mean ± sem
    means = np.array([np.nanmean(ctrl), np.nanmean(stim)], float)
    errs  = np.array([sem(ctrl, nan_policy='omit'), sem(stim, nan_policy='omit')], float)
    errs = np.nan_to_num(errs, nan=0.0)

    # bars
    barc = ax.bar([0, 1], means, yerr=errs, capsize=2, width=0.6, edgecolor='none',
                  alpha=.6, zorder=2,
                  error_kw={'elinewidth': 0.6, 'capthick': 0.6, 'ecolor': 'k'})
    barc.patches[0].set_facecolor(colors[0])
    barc.patches[1].set_facecolor(colors[1])

    # scatter
    rng = np.random.default_rng(seed)
    x_ctrl = -jitter + 2*jitter*rng.random(len(ctrl)) + 0
    x_stim = -jitter + 2*jitter*rng.random(len(stim)) + 1
    ax.scatter(x_ctrl, ctrl, s=point_size, color=colors[0],
               edgecolor='none', alpha=.6, zorder=4)
    ax.scatter(x_stim, stim, s=point_size, color=colors[1],
               edgecolor='none', alpha=.6, zorder=4)
    # ax.scatter(x_ctrl, ctrl, s=point_size, color='grey',
    #            edgecolor='none', alpha=.6, zorder=4)
    # ax.scatter(x_stim, stim, s=point_size, color='grey',
    #            edgecolor='none', alpha=.6, zorder=4)

    # stats
    t_stat, t_p = ttest_ind(ctrl, stim, equal_var=not use_welch)
    rs_stat, rs_p = ranksums(ctrl, stim)

    # ylim (auto)
    top_data_val = np.nanmax([np.nanmax(ctrl), np.nanmax(stim)])
    top_bar_val = np.nanmax(means + errs)
    top_y = np.nanmax([top_data_val, top_bar_val])

    if ylim is None:
        data_min = np.nanmin([np.nanmin(ctrl), np.nanmin(stim), means.min() - errs.max()])
        y_min0 = float(data_min)
        # y_min0 = min(0.0, float(data_min))
        y_max0 = float(top_y)
        base_range = max(1e-6, y_max0 - y_min0)
        pad_top = 0.25 * base_range
        pad_bot = 0.05 * base_range
        ylims = (y_min0-pad_bot, y_max0 + pad_top)
    else:
        ylims = ylim

    ax.set(xticks=[0, 1], xticklabels=xticklabels,
           ylabel=ylabel, title=title, ylim=ylims)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    # bump sizes
    yrange = ylims[1] - ylims[0]
    bump = 0.02 * yrange

    # --- mean ± sem annotation above each bar ---
    for i, (m, e) in enumerate(zip(means, errs)):
        text = f"{m:.3f} ± {e:.3f}"
        annotate_bar(ax, i, m + e, text, bump * 0.8)

    # --- optional median + IQR annotation below the x-axis ---
    if print_median_iqr:
        med_c = np.nanmedian(ctrl); q1_c, q3_c = np.nanpercentile(ctrl, [25, 75])
        med_s = np.nanmedian(stim); q1_s, q3_s = np.nanpercentile(stim, [25, 75])
        print(f"{xticklabels[0]}: median={med_c:.4f}, IQR=[{q1_c:.4f}, {q3_c:.4f}] (n={len(ctrl)}) | "
              f"{xticklabels[1]}: median={med_s:.4f}, IQR=[{q1_s:.4f}, {q3_s:.4f}] (n={len(stim)})")
        trans = ax.get_xaxis_transform()
        for xi, (med, q1_v, q3_v, c) in enumerate([
            (med_c, q1_c, q3_c, colors[0]),
            (med_s, q1_s, q3_s, colors[1]),
        ]):
            ax.text(xi, -0.08, f"med={med:.3f}\n[{q1_v:.3f}, {q3_v:.3f}]",
                    transform=trans, ha='center', va='top',
                    fontsize=6, color=c, clip_on=False)

    # --- statistical annotations at top ---
    y_rank = ylims[1] - 1.0*bump
    y_t    = ylims[1] - 2.2*bump
    annotate_sig(ax, 0, 1, y_t,   f"Welch t-test p={t_p:.3g} ({sigstars(t_p)})", bump)
    annotate_sig(ax, 0, 1, y_rank, f"ranksum p={rs_p:.3g} ({sigstars(rs_p)})", bump)

    # Bring axes, labels, spines, error bars, and annotations to front for Illustrator editing
    front_zorder = 10
    for spine in ax.spines.values():
        spine.set_zorder(front_zorder)
    ax.xaxis.set_zorder(front_zorder)
    ax.yaxis.set_zorder(front_zorder)
    for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        label.set_zorder(front_zorder)
    for tick in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        tick.set_zorder(front_zorder)
    ax.xaxis.label.set_zorder(front_zorder)
    ax.yaxis.label.set_zorder(front_zorder)
    ax.title.set_zorder(front_zorder)
    for text in ax.texts:
        text.set_zorder(front_zorder)
    for line in ax.lines:
        line.set_zorder(front_zorder)
    for container in ax.containers:
        if hasattr(container, 'errorbar') and container.errorbar is not None:
            for item in container.errorbar:
                if item is not None:
                    if hasattr(item, '__iter__'):
                        for line in item:
                            if line is not None and hasattr(line, 'set_zorder'):
                                line.set_zorder(front_zorder)
                    elif hasattr(item, 'set_zorder'):
                        item.set_zorder(front_zorder)

    return {
        'ttest_ind': {'stat': float(t_stat), 'p': float(t_p),
                      'n1': len(ctrl), 'n2': len(stim), 'welch': use_welch},
        'ranksums':  {'stat': float(rs_stat), 'p': float(rs_p),
                      'n1': len(ctrl), 'n2': len(stim)},
        'means': means.tolist(),
        'sems': errs.tolist()
    }

def _norm_rgb(color):
    if isinstance(color, str):
        return to_rgb(color)
    r, g, b = color
    return (r/255.0, g/255.0, b/255.0) if max(r, g, b) > 1 else (r, g, b)

def single_color_cmap(color, kind="gradient", anchor="white",
                      low_strong=False, power=None, N=256, name=None):
    """
    Make a colormap from a single color.
    low_strong=True  -> lower values look stronger (color→white).
    power: None or >0.  <1 boosts low values more; >1 compresses them.
    """
    c = _norm_rgb(color)
    if kind == "flat":
        return ListedColormap([c], name or "flat")

    start = (1, 1, 1) if anchor == "white" else (0, 0, 0)
    ends = [start, c]
    if low_strong:
        ends = ends[::-1]  # color → white/black

    if power is None:
        return LinearSegmentedColormap.from_list(name or "onecolor", ends, N=N)

    # Nonlinear (power-law) bias for stronger low values
    t = np.linspace(0, 1, N) ** power
    c0, c1 = np.array(ends[0]), np.array(ends[1])
    colors = (1 - t)[:, None] * c0 + t[:, None] * c1
    return ListedColormap(colors, name or f"onecolor_p{power}")

# --- examples ---
# Lower values stronger (blue), fading to white:
cmap_low_strong = single_color_cmap("#1f77b4", low_strong=True)

# Even stronger emphasis on the low end:
cmap_low_strong_p = single_color_cmap("#1f77b4", low_strong=True, power=0.6)

def plot_mean_trace(data, ax, xaxis=None, color='green', sem_off=False, **kwargs):
    data = np.array(data)
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
    if xaxis is None:
        xaxis = np.arange(mean.shape[0])
    ax.plot(xaxis, mean, color=color, **kwargs)
    if sem_off==False:
        ax.fill_between(xaxis, mean+sem, mean-sem, 
                       facecolor=color, edgecolor='none', alpha=.3)


def plot_two_traces_with_scalebars(
    data1, data2, xaxis, ax=None,
    colors=("tab:green", "tab:red"),
    labels=("trace A", "trace B"),
    baseline_window=None,     # e.g., (-1.0, 0.0)
    baseline_correct=True,    # <--- NEW: disable to keep raw baselines
    match_centers=True,       # <--- NEW: disable to keep original relative offsets
    sem_alpha=0.25, lw=1.2,
    timebar=1.0, dffbar=0.1,
    time_label="s", dff_label="%dF/F",
    bar_side="right",
    bar_pad_frac=(0.05, 0.08),
    bar_y_bump_frac=0.06,     # extra vertical lift for the scalebar (fraction of y-range) when x-axis is shown
    center_mode="midrange",   # 'midrange' | 'median' | 'mean' | None
    # --- vertical event line ---
    vline=True, vline_t=0.0, vline_kwargs=None,
    # --- keep x-axis while still drawing small scalebar ---
    show_xaxis=False,
    x_tick_step=None,         # e.g., 0.5 (seconds). None = Matplotlib default
    xlabel=None,              # e.g., "Time (s)"; None = no label
    x_tick_params=None,        # dict passed to ax.tick_params(axis='x', ...)
    title=''
):
    """
    One or two mean±SEM traces without full axes, with small time/ΔF/F scale bars.

    NEW:
      - baseline_correct (bool): if False, skip baseline subtraction even if a baseline_window is given.
      - match_centers (bool): if False, do not offset data2 to match data1; preserves original y positions.
      - show_xaxis, bar_y_bump_frac, x_tick_step, xlabel, x_tick_params as before.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def _as_2d(arr):
        arr = np.asarray(arr)
        return arr[None, :] if arr.ndim == 1 else arr

    def _maybe_baseline_correct(arr2d, x, win, do_correct: bool):
        if (not do_correct) or (win is None):
            return arr2d
        t0, t1 = win
        mask = (x >= t0) & (x <= t1)
        if not np.any(mask):
            # fallback: first 10% of samples
            k = max(1, int(0.1 * arr2d.shape[1]))
            mask = np.zeros(arr2d.shape[1], dtype=bool); mask[:k] = True
        base = np.nanmean(arr2d[:, mask], axis=1, keepdims=True)
        return arr2d - base

    def _mean_sem(arr2d):
        mean = np.nanmean(arr2d, axis=0)
        n_eff = np.sum(np.isfinite(arr2d), axis=0)
        std = np.nanstd(arr2d, axis=0, ddof=1)
        sem = np.divide(std, np.sqrt(np.maximum(n_eff, 1)), where=n_eff > 0)
        sem[~np.isfinite(sem)] = 0.0
        return mean, sem

    def _center_value(vec, mode):
        if mode is None: return 0.0
        if mode == "midrange": return 0.5*(np.nanmin(vec) + np.nanmax(vec))
        if mode == "median":   return float(np.nanmedian(vec))
        if mode == "mean":     return float(np.nanmean(vec))
        raise ValueError("center_mode must be {'midrange','median','mean',None}")

    xaxis = np.asarray(xaxis)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=300)
    else:
        fig = ax.figure

    # --- plot data1 ---
    if data1 is not None:
        data1 = _as_2d(data1)
        data1 = _maybe_baseline_correct(data1, xaxis, baseline_window, baseline_correct)
        m1, s1 = _mean_sem(data1)
        ax.fill_between(xaxis, m1 - s1, m1 + s1,
                        facecolor=colors[0], alpha=sem_alpha, edgecolor='none', zorder=1)
        ax.plot(xaxis, m1, color=colors[0], lw=lw, zorder=3, label=labels[0])
    else:
        m1 = None

    # --- plot data2 ---
    if data2 is not None:
        data2 = _as_2d(data2)
        data2 = _maybe_baseline_correct(data2, xaxis, baseline_window, baseline_correct)
        m2, s2 = _mean_sem(data2)

        # optional vertical alignment of centers
        if match_centers and (data1 is not None) and (center_mode is not None):
            c1 = _center_value(m1, center_mode)
            c2 = _center_value(m2, center_mode)
            offset2 = c1 - c2
        else:
            offset2 = 0.0

        ax.fill_between(xaxis, (m2 + offset2) - s2, (m2 + offset2) + s2,
                        facecolor=colors[1], alpha=sem_alpha, edgecolor='none', zorder=2)
        ax.plot(xaxis, m2 + offset2, color=colors[1], lw=lw, zorder=4, label=labels[1])

    # --- vertical event line ---
    if vline and (np.nanmin(xaxis) <= vline_t <= np.nanmax(xaxis)):
        vk = dict(color='grey', ls='--', lw=1.0, zorder=4.7)
        if isinstance(vline_kwargs, dict):
            vk.update(vline_kwargs)
        ax.axvline(vline_t, **vk)

    # --- axes styling ---
    ax.set_yticks([])
    ax.tick_params(axis='y', left=False, labelleft=False)

    if show_xaxis:
        for name, sp in ax.spines.items():
            sp.set_visible(name == 'bottom')
        if x_tick_params is None:
            x_tick_params = dict(length=3, pad=2, labelsize=8, direction='out')
        ax.tick_params(axis='x', bottom=True, labelbottom=True, **x_tick_params)

        if x_tick_step is not None:
            lo, hi = np.min(xaxis), np.max(xaxis)
            start = np.floor(lo / x_tick_step) * x_tick_step
            stop  = np.ceil(hi / x_tick_step) * x_tick_step
            ticks = np.arange(start, stop + 0.5 * x_tick_step, x_tick_step)
            ax.set_xticks(ticks)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("")
    else:
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("")

    ax.margins(x=0.02)

    # --- small scale bars (always drawn) ---
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]; yspan = ylim[1] - ylim[0]
    xpad = bar_pad_frac[0] * xspan
    ypad = bar_pad_frac[1] * yspan
    bump = (bar_y_bump_frac if show_xaxis else 0.0) * yspan

    x0 = (xlim[1] - xpad - timebar) if bar_side.lower().startswith("r") else (xlim[0] + xpad)
    y0 = ylim[0] + ypad + bump

    ax.plot([x0, x0 + timebar], [y0, y0], color="black", lw=1.2, zorder=5, clip_on=False)
    ax.plot([x0 + timebar, x0 + timebar], [y0, y0 + dffbar], color="black", lw=1.2, zorder=5, clip_on=False)
    ax.text(x0 + timebar/2, y0 - 0.02*yspan, f"{timebar:g} {time_label}",
            ha="center", va="top", fontsize=8, clip_on=False)
    ax.text(x0 + timebar + 0.01*xspan, y0 + dffbar/2, f"{dffbar:g} {dff_label}",
            ha="left", va="center", fontsize=8, rotation=90, clip_on=False)

    # title
    ax.set(title=title)

    fig.tight_layout()
    return fig, ax

def plot_two_traces_with_scalebars_fixbar(
    data1, data2, xaxis, ax=None,
    colors=("tab:green", "tab:red"),
    labels=("trace A", "trace B"),
    baseline_window=None,     # e.g., (-1.0, 0.0)
    sem_alpha=0.25, lw=1.2,
    timebar=1.0, dffbar=0.1,  # now can be float or (d1, d2)
    ylims=None,
    time_label="s", dff_label="dF/F",
    bar_side="right",
    bar_pad_frac=(0.05, 0.08),
    bar_y_bump_frac=0.06,     # extra vertical lift for scalebar when x-axis is shown
    center_mode="midrange",   # 'midrange' | 'median' | 'mean' | None
    # --- vertical event line ---
    vline=True, vline_t=0.0, vline_kwargs=None,
    # --- keep x-axis while still drawing small scalebar ---
    show_xaxis=False,
    x_tick_step=None,         # e.g., 0.5 (seconds). None = Matplotlib default
    xlabel=None,              # e.g., "Time (s)"; None = no label
    x_tick_params=None,       # dict passed to ax.tick_params(axis='x', ...)
    # --- NEW: per-trace dF/F scaling and dual bars ---
    scale_by_dffbar=False,    # if True and dffbar=(d1,d2), scale traces by their own bars
    dffbar_plot_height=None,  # on-figure height each bar represents when scaling
    draw_two_dff_bars=False,  # draw separate dF/F bars for the two traces
    dffbar_gap_frac=0.03,     # horizontal gap (fraction of x-span) between the two dF/F bars
    dffbar_colors=None        # None=black; "match"=use colors[i]; or tuple of 2 colors
):
    """
    One or two mean±SEM traces without full axes, with small time/ΔF/F scale bars.
    If both data1 and data2 are provided, trace2 is vertically offset to sit near
    the 'middle' of trace1. Can also draw a vertical line at vline_t.

    NEW:
      - dffbar can be a float or a tuple (d1, d2).
      - scale_by_dffbar: when True and dffbar=(d1,d2), multiply each trace so that
        d1 and d2 map to the same on-figure height (dffbar_plot_height).
      - draw_two_dff_bars: draw separate vertical dF/F bars for the two traces.
      - dffbar_colors: color the bars (use "match" to match the trace colors).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def _as_2d(arr):
        arr = np.asarray(arr)
        return arr[None, :] if arr.ndim == 1 else arr

    def _baseline_correct(arr2d, x, win):
        if win is None:
            return arr2d
        t0, t1 = win
        mask = (x >= t0) & (x <= t1)
        if not np.any(mask):
            # fallback: first 10% of samples
            k = max(1, int(0.1 * arr2d.shape[1]))
            mask = np.zeros(arr2d.shape[1], dtype=bool); mask[:k] = True
        base = np.nanmean(arr2d[:, mask], axis=1, keepdims=True)
        return arr2d - base

    def _mean_sem(arr2d):
        mean = np.nanmean(arr2d, axis=0)
        n_eff = np.sum(np.isfinite(arr2d), axis=0)
        std = np.nanstd(arr2d, axis=0, ddof=1)
        sem = np.divide(std, np.sqrt(np.maximum(n_eff, 1)), where=n_eff > 0)
        sem[~np.isfinite(sem)] = 0.0
        return mean, sem

    def _center_value(vec, mode):
        if mode is None: return 0.0
        if mode == "midrange": return 0.5*(np.nanmin(vec) + np.nanmax(vec))
        if mode == "median":   return float(np.nanmedian(vec))
        if mode == "mean":     return float(np.nanmean(vec))
        raise ValueError("center_mode must be {'midrange','median','mean',None}")

    # --- normalize dffbar inputs ---
    if isinstance(dffbar, (list, tuple, np.ndarray)):
        d1_bar = float(dffbar[0]) if len(dffbar) >= 1 else None
        d2_bar = float(dffbar[1]) if len(dffbar) >= 2 else d1_bar
    else:
        d1_bar = float(dffbar) if dffbar is not None else None
        d2_bar = float(dffbar) if dffbar is not None else None

    xaxis = np.asarray(xaxis)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=300)
    else:
        fig = ax.figure

    # --- prepare data, baseline, and optional per-trace scaling by dF/F bar ---
    if data1 is not None:
        data1 = _as_2d(data1)
        data1 = _baseline_correct(data1, xaxis, baseline_window)
    if data2 is not None:
        data2 = _as_2d(data2)
        data2 = _baseline_correct(data2, xaxis, baseline_window)

    # decide plot height used when scaling
    if scale_by_dffbar:
        if dffbar_plot_height is not None:
            bar_h_plot = float(dffbar_plot_height)
        else:
            # default to the first non-None bar or 0.1 as a fallback
            bar_h_plot = (d1_bar if (d1_bar is not None and d1_bar > 0) else
                          d2_bar if (d2_bar is not None and d2_bar > 0) else 0.1)
        # compute scales
        s1 = (bar_h_plot / d1_bar) if (data1 is not None and d1_bar and d1_bar > 0) else 1.0
        s2 = (bar_h_plot / d2_bar) if (data2 is not None and d2_bar and d2_bar > 0) else 1.0
        if data1 is not None: data1 = data1 * s1
        if data2 is not None: data2 = data2 * s2
    else:
        bar_h_plot = None  # not used in this mode

    # --- plot data1 ---
    if data1 is not None:
        m1, s1_sem = _mean_sem(data1)
        ax.fill_between(xaxis, m1 - s1_sem, m1 + s1_sem,
                        facecolor=colors[0], alpha=sem_alpha, edgecolor='none', zorder=1)
        ax.plot(xaxis, m1, color=colors[0], lw=lw, zorder=3, label=labels[0])
    else:
        m1 = None

    # --- plot data2 (optionally offset relative to data1) ---
    if data2 is not None:
        m2, s2_sem = _mean_sem(data2)
        if (data1 is not None) and (center_mode is not None):
            c1 = _center_value(m1, center_mode)
            c2 = _center_value(m2, center_mode)
            offset2 = c1 - c2
        else:
            offset2 = 0.0
        ax.fill_between(xaxis, (m2 + offset2) - s2_sem, (m2 + offset2) + s2_sem,
                        facecolor=colors[1], alpha=sem_alpha, edgecolor='none', zorder=2)
        ax.plot(xaxis, m2 + offset2, color=colors[1], lw=lw, zorder=4, label=labels[1])

    # --- vertical event line ---
    if vline and (np.nanmin(xaxis) <= vline_t <= np.nanmax(xaxis)):
        vk = dict(color='grey', ls='--', lw=1.0, zorder=4.7)
        if isinstance(vline_kwargs, dict):
            vk.update(vline_kwargs)
        ax.axvline(vline_t, **vk)

    # --- axes styling ---
    ax.set_yticks([])
    ax.tick_params(axis='y', left=False, labelleft=False)

    if show_xaxis:
        for name, sp in ax.spines.items():
            sp.set_visible(name == 'bottom')
        if x_tick_params is None:
            x_tick_params = dict(length=3, pad=2, labelsize=8, direction='out')
        ax.tick_params(axis='x', bottom=True, labelbottom=True, **x_tick_params)

        if x_tick_step is not None:
            lo, hi = np.min(xaxis), np.max(xaxis)
            start = np.floor(lo / x_tick_step) * x_tick_step
            stop  = np.ceil(hi / x_tick_step) * x_tick_step
            ticks = np.arange(start, stop + 0.5 * x_tick_step, x_tick_step)
            ax.set_xticks(ticks)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("")
    else:
        ax.set_xticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("")

    ax.margins(x=0.02)
    
    # calculate ylim_min based on ch2 signal limits
    if ylims is None:
        ch1_low = np.nanmin(m1 - s1_sem) - 0.15
        ch2_low = np.nanmin(m2 + offset2 - s2_sem) - 0.15
        
        # per-trace upper envelopes (mean + sem)
        ch1_high = np.nanmax(m1 + s1_sem) + 0.15
        ch2_high = np.nanmax(m2 + offset2 + s2_sem) + 0.15
        
        ylim_min = np.nanmin([ch1_low, ch2_low])
        ylim_max = np.nanmax([ch1_high, ch2_high])
        
        # optional extra bottom room when x-axis is shown
        if show_xaxis:
            yspan0 = ylim_max - ylim_min
            ylim_min = ylim_min - 0.08 * yspan0   # tune 0.05–0.15
        
        ylims = (ylim_min, ylim_min+5)

    ax.set_ylim(ylims)
    # --- small scale bars (time + one or two dF/F bars) ---
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]; yspan = ylim[1] - ylim[0]
    xpad = bar_pad_frac[0] * xspan
    ypad = bar_pad_frac[1] * yspan
    bump = (bar_y_bump_frac if show_xaxis else 0.0) * yspan

    # base position
    if bar_side.lower().startswith("r"):
        x0 = xlim[1] - xpad - timebar
        t_left = x0; t_right = x0 + timebar
        bar_x1 = t_right
        bar_x2 = bar_x1 + dffbar_gap_frac * xspan
    else:
        x0 = xlim[0] + xpad
        t_left = x0; t_right = x0 + timebar
        bar_x1 = t_left
        bar_x2 = bar_x1 - dffbar_gap_frac * xspan

    y0 = ylim[0] + ypad + bump

    # draw time bar
    ax.plot([t_left, t_right], [y0, y0], color="black", lw=1.2, zorder=5, clip_on=False)
    ax.text((t_left + t_right)/2, y0 - 0.02*yspan, f"{timebar:g} {time_label}",
            ha="center", va="top", fontsize=8, clip_on=False)

    # decide colors for dF/F bars
    if dffbar_colors is None:
        bar_col1 = bar_col2 = "black"
    elif dffbar_colors == "match":
        bar_col1, bar_col2 = colors[0], colors[1]
    elif isinstance(dffbar_colors, (list, tuple)) and len(dffbar_colors) >= 2:
        bar_col1, bar_col2 = dffbar_colors[0], dffbar_colors[1]
    else:
        bar_col1 = bar_col2 = "black"

    # determine bar heights and labels
    def _fmt_bar(v):
        try:
            return f"{v:g} {dff_label}"
        except Exception:
            return f"{v} {dff_label}"

    if draw_two_dff_bars:
        # heights depend on scaling mode
        if scale_by_dffbar:
            h1 = h2 = bar_h_plot
        else:
            h1 = d1_bar if (d1_bar is not None) else 0.1
            h2 = d2_bar if (d2_bar is not None) else h1

        # bar 1
        ax.plot([bar_x1, bar_x1], [y0, y0 + h1], color=bar_col1, lw=1.2, zorder=5, clip_on=False)
        if d1_bar is not None:
            ax.text(bar_x1 + 0.01*xspan, y0 + h1/2, _fmt_bar(d1_bar),
                    ha="left", va="center", fontsize=8, rotation=90, clip_on=False)
        # bar 2
        ax.plot([bar_x2, bar_x2], [y0, y0 + h2], color=bar_col2, lw=1.2, zorder=5, clip_on=False)
        if d2_bar is not None:
            ax.text(bar_x2 + 0.01*xspan, y0 + h2/2, _fmt_bar(d2_bar),
                    ha="left", va="center", fontsize=8, rotation=90, clip_on=False)
    else:
        # single bar behavior (back-compatible)
        if scale_by_dffbar and (bar_h_plot is not None):
            h = bar_h_plot
            label_val = d1_bar if d1_bar is not None else d2_bar
        else:
            h = d1_bar if (d1_bar is not None) else 0.1
            label_val = d1_bar
        ax.plot([t_right, t_right], [y0, y0 + h], color="black", lw=1.2, zorder=5, clip_on=False)
        if label_val is not None:
            ax.text(t_right + 0.01*xspan, y0 + h/2, _fmt_bar(label_val),
                    ha="left", va="center", fontsize=8, rotation=90, clip_on=False)

    fig.tight_layout()
    return fig, ax


# def plot_two_traces_with_scalebars(
#     data1, data2, xaxis, ax=None,
#     colors=("tab:green", "tab:red"),
#     labels=("trace A", "trace B"),
#     baseline_window=None,     # e.g., (-1.0, 0.0)
#     sem_alpha=0.25, lw=1.2,
#     timebar=1.0, dffbar=0.1,
#     time_label="s", dff_label="dF/F",
#     bar_side="right",
#     bar_pad_frac=(0.05, 0.08),
#     center_mode="midrange",   # 'midrange' | 'median' | 'mean' | None
#     # --- NEW: vertical event line ---
#     vline=True, vline_t=0.0, vline_kwargs=None
# ):
#     """
#     Two mean±SEM traces without full axes, with small time/ΔF/F scale bars.
#     Trace 2 is vertically offset to sit near the 'middle' of trace 1.
#     Optionally draws a vertical line at vline_t (default 0).
#     """

#     def _as_2d(arr):
#         arr = np.asarray(arr)
#         return arr[None, :] if arr.ndim == 1 else arr

#     def _baseline_correct(arr2d, x, win):
#         if win is None:
#             return arr2d
#         t0, t1 = win
#         mask = (x >= t0) & (x <= t1)
#         if not np.any(mask):
#             k = max(1, int(0.1 * arr2d.shape[1]))
#             mask = np.zeros(arr2d.shape[1], dtype=bool); mask[:k] = True
#         base = np.nanmean(arr2d[:, mask], axis=1, keepdims=True)
#         return arr2d - base

#     def _mean_sem(arr2d):
#         mean = np.nanmean(arr2d, axis=0)
#         n_eff = np.sum(np.isfinite(arr2d), axis=0)
#         std = np.nanstd(arr2d, axis=0, ddof=1)
#         sem = np.divide(std, np.sqrt(np.maximum(n_eff, 1)), where=n_eff>0)
#         sem[~np.isfinite(sem)] = 0.0
#         return mean, sem

#     def _center_value(vec, mode):
#         if mode is None: return 0.0
#         if mode == "midrange": return 0.5*(np.nanmin(vec) + np.nanmax(vec))
#         if mode == "median":   return float(np.nanmedian(vec))
#         if mode == "mean":     return float(np.nanmean(vec))
#         raise ValueError("center_mode must be {'midrange','median','mean',None}")

#     data1 = _as_2d(data1)
#     data2 = _as_2d(data2)
#     xaxis = np.asarray(xaxis)

#     # optional per-trial baseline correction
#     data1 = _baseline_correct(data1, xaxis, baseline_window)
#     data2 = _baseline_correct(data2, xaxis, baseline_window)

#     # stats
#     m1, s1 = _mean_sem(data1)
#     m2, s2 = _mean_sem(data2)

#     # vertical offset so trace2 sits around the middle of trace1
#     if center_mode is not None:
#         c1 = _center_value(m1, center_mode)
#         c2 = _center_value(m2, center_mode)
#         offset2 = c1 - c2
#     else:
#         offset2 = 0.0

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=300)
#     else:
#         fig = ax.figure

#     # draw traces
#     ax.fill_between(xaxis, m1 - s1, m1 + s1, facecolor=colors[0], alpha=sem_alpha, edgecolor='none', zorder=1)
#     ax.plot(xaxis, m1, color=colors[0], lw=lw, zorder=3, label=labels[0])

#     ax.fill_between(xaxis, (m2 + offset2) - s2, (m2 + offset2) + s2,
#                     facecolor=colors[1], alpha=sem_alpha, edgecolor='none', zorder=2)
#     ax.plot(xaxis, m2 + offset2, color=colors[1], lw=lw, zorder=4, label=labels[1])

#     # --- NEW: vertical line at vline_t (default 0) ---
#     if vline and (np.nanmin(xaxis) <= vline_t <= np.nanmax(xaxis)):
#         vk = dict(color='grey', ls='--', lw=1.0, zorder=4.7)
#         if isinstance(vline_kwargs, dict):
#             vk.update(vline_kwargs)
#         ax.axvline(vline_t, **vk)

#     # minimalist axes
#     ax.set_xticks([]); ax.set_yticks([])
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.set_xlabel(""); ax.set_ylabel("")
#     ax.margins(x=0.02)

#     # scale bars
#     xlim = ax.get_xlim(); ylim = ax.get_ylim()
#     xspan = xlim[1] - xlim[0]; yspan = ylim[1] - ylim[0]
#     xpad = bar_pad_frac[0] * xspan; ypad = bar_pad_frac[1] * yspan
#     x0 = (xlim[1] - xpad - timebar) if bar_side.lower().startswith("r") else (xlim[0] + xpad)
#     y0 = ylim[0] + ypad
#     ax.plot([x0, x0 + timebar], [y0, y0], color="black", lw=1.2, zorder=5)
#     ax.plot([x0 + timebar, x0 + timebar], [y0, y0 + dffbar], color="black", lw=1.2, zorder=5)
#     ax.text(x0 + timebar/2, y0 - 0.02*yspan, f"{timebar:g} {time_label}", ha="center", va="top", fontsize=8)
#     ax.text(x0 + timebar + 0.01*xspan, y0 + dffbar/2, f"{dffbar:g}% {dff_label}", ha="left", va="center", fontsize=8, rotation=90)

#     fig.tight_layout()
#     return fig, ax

def plot_paired_violin(list1, list2,
                       colors=('steelblue', 'orange'),
                       m_point = 'mean',
                       colname=None, ylabel=None, ylim=None,
                       title_prefix=None,
                       show_pair_lines=True,
                       ax=None,
                       markersize=3,
                       fs=10,
                       pos1=0.5, pos2=0.8,
                       width=0.2,
                       dx=0.02,
                       print_median_iqr=False):
    """
    Two separated half violins:
      - list1: left half at x=pos0
      - list2: right half at x=pos1
    Plus paired points and optional connecting lines.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, wilcoxon

    list1 = np.asarray(list1, dtype=float)
    list2 = np.asarray(list2, dtype=float)
    if list1.shape != list2.shape:
        raise ValueError("For paired data, list1 and list2 must have the same shape.")

    mask = np.isfinite(list1) & np.isfinite(list2)
    list1 = list1[mask]
    list2 = list2[mask]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=200)
        created_fig = True

    def _clip_violin_body(body, center, side):
        path = body.get_paths()[0]
        verts = path.vertices
        if side == "right":
            verts[:, 0] = np.maximum(verts[:, 0], center)
        elif side == "left":
            verts[:, 0] = np.minimum(verts[:, 0], center)
        else:
            raise ValueError("side must be 'right' or 'left'")
        path.vertices = verts

    # --- draw separated half violins ---
    vp1 = ax.violinplot([list1], positions=[pos1], widths=width, showextrema=False)
    _clip_violin_body(vp1["bodies"][0], pos1, "left")
    vp1["bodies"][0].set_facecolor(colors[0])
    vp1["bodies"][0].set_edgecolor("none")
    vp1["bodies"][0].set_alpha(0.7)

    vp2 = ax.violinplot([list2], positions=[pos2], widths=width, showextrema=False)
    _clip_violin_body(vp2["bodies"][0], pos2, "right")
    vp2["bodies"][0].set_facecolor(colors[1])
    vp2["bodies"][0].set_edgecolor("none")
    vp2["bodies"][0].set_alpha(0.7)

    # --- points + paired lines ---
    # Place points slightly toward the open side of each half violin
    x1 = pos1 + dx   # list1 is left-half -> open side is to the right
    x2 = pos2 - dx   # list2 is right-half -> open side is to the left

    # ax.plot(np.full(len(list1), x1), list1, 'o',
    #         color=colors[0], alpha=0.4, markersize=markersize,
    #         markeredgecolor='none', markeredgewidth=0, zorder=3
    #         )
    # ax.plot(np.full(len(list2), x2), list2, 'o',
    #         color=colors[1], alpha=0.4, markersize=markersize,
    #         markeredgecolor='none', markeredgewidth=0, zorder=3)

    if show_pair_lines:
        for y1, y2 in zip(list1, list2):
            ax.plot([x1, x2], [y1, y2],
                    color='0.6', alpha=0.4, linewidth=0.8, zorder=2)

    # mean markers (swap to median if desired)
    if m_point == 'mean':
        m1 = np.mean(list1)
        m2 = np.mean(list2)
    elif m_point == 'median':
        m1 = np.median(list1)
        m2 = np.median(list2)
        
    ax.scatter([x1], [m1], s=30, c=[colors[0]], edgecolor=None,
               alpha=1, zorder=4)
    ax.scatter([x2], [m2], s=30, c=[colors[1]], edgecolor=None,
               alpha=1, zorder=4)
    ax.plot([x1, x2], [m1, m2],
            color='k', alpha=.8, linewidth=2, zorder=2)
    ax.plot([pos1-dx, x1], [m1, m1],
            color='k', alpha=.8, linewidth=2, zorder=2)
    ax.plot([pos2+dx, x2], [m2, m2],
            color='k', alpha=.8, linewidth=2, zorder=2)

    # --- axes cosmetics ---
    pad = 0.3
    ax.set_xlim(min(pos1, pos2) - pad, max(pos1, pos2) + pad)

    # x ticks / labels
    if isinstance(colname, (list, tuple)) and len(colname) == 2:
        ax.set_xticks([pos1, pos2])
        ax.set_xticklabels([colname[0], colname[1]])
        plot_title = f"{colname[0]} vs {colname[1]}"
    elif colname:
        ax.set_xticks([pos1, pos2])
        ax.set_xticklabels(['Cond 1', 'Cond 2'])
        plot_title = str(colname)
    else:
        ax.set_xticks([pos1, pos2])
        ax.set_xticklabels(['Cond 1', 'Cond 2'])
        plot_title = "Paired Comparison"

    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)

    # --- stats ---
    t_stat, t_pval = ttest_rel(list1, list2)
    try:
        w_stat, w_pval = wilcoxon(list1, list2)
    except ValueError:
        w_pval = np.nan

    stat_str = f"t-rel p={t_pval:.4g}, Wilcoxon p={'nan' if np.isnan(w_pval) else f'{w_pval:.4g}'}"
    title = f"{plot_title}\n{stat_str}" if not title_prefix else f"{title_prefix}: {plot_title}\n{stat_str}"
    ax.set_title(title, fontsize=fs)

    # sample size
    n_pairs = len(list1)
    y0 = ax.get_ylim()[0]
    ax.text(pos1, y0, f"n={n_pairs}", ha="center", va="top", fontsize=fs-2)
    ax.text(pos2, y0, f"n={n_pairs}", ha="center", va="top", fontsize=fs-2)

    # optional median + IQR (print + annotate below each violin)
    if print_median_iqr:
        med1 = np.nanmedian(list1); q1_1, q3_1 = np.nanpercentile(list1, [25, 75])
        med2 = np.nanmedian(list2); q1_2, q3_2 = np.nanpercentile(list2, [25, 75])
        lab1 = colname[0] if isinstance(colname, (list, tuple)) and len(colname) == 2 else 'Cond 1'
        lab2 = colname[1] if isinstance(colname, (list, tuple)) and len(colname) == 2 else 'Cond 2'
        print(f"{lab1}: median={med1:.4f}, IQR=[{q1_1:.4f}, {q3_1:.4f}] (n={n_pairs}) | "
              f"{lab2}: median={med2:.4f}, IQR=[{q1_2:.4f}, {q3_2:.4f}] (n={n_pairs})")
        trans = ax.get_xaxis_transform()
        ax.text(pos1, -0.10, f"med={med1:.3f}\n[{q1_1:.3f}, {q3_1:.3f}]",
                transform=trans, ha='center', va='top',
                fontsize=fs-3, color=colors[0], clip_on=False)
        ax.text(pos2, -0.10, f"med={med2:.3f}\n[{q1_2:.3f}, {q3_2:.3f}]",
                transform=trans, ha='center', va='top',
                fontsize=fs-3, color=colors[1], clip_on=False)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    if created_fig:
        return fig, ax
    return ax



# def plot_paired_violin(list1, list2, 
#                        colors=['steelblue', 'orange'], 
#                        colname=None, ylabel=None, ylim=None, 
#                        title_prefix=None,
#                        show_pair_lines=True,
#                        ax=None,
#                        markersize=3):
#     """
#     Plot paired violin plots comparing two matched 1D lists.

#     Parameters:
#         list1 (array-like): shape (n,), first condition
#         list2 (array-like): shape (n,), second condition  
#         colors (list): Colors for the two violins.
#         colname (str or list of str): Used for x-axis labels and plot title.
#         ylabel (str): Label for y-axis.
#         ylim (tuple): Y-axis limits.
#         title_prefix (str): Optional prefix for the plot title.
#         show_pair_lines (bool): If True, draw lines connecting paired points.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import ttest_rel, wilcoxon

#     list1 = np.asarray(list1, dtype=float)
#     list2 = np.asarray(list2, dtype=float)

#     if list1.shape != list2.shape:
#         raise ValueError("For paired data, list1 and list2 must have the same shape.")

#     # Drop NaN pairs (keep only pairs where both are finite)
#     mask = np.isfinite(list1) & np.isfinite(list2)
#     list1 = list1[mask]
#     list2 = list2[mask]
    
#     # Drop NaN pairs (keep only pairs where both are finite)
#     mask = np.isfinite(list1) & np.isfinite(list2)
#     list1 = list1[mask]
#     list2 = list2[mask]
    
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    
#     pos = 1.0  # shared center position for split violin
#     width = 0.8
    
#     def _clip_violin_body(body, pos, side):
#         path = body.get_paths()[0]
#         verts = path.vertices
#         if side == "right":
#             verts[:, 0] = np.maximum(verts[:, 0], pos)
#         elif side == "left":
#             verts[:, 0] = np.minimum(verts[:, 0], pos)
#         else:
#             raise ValueError("side must be 'right' or 'left'")
#         path.vertices = verts
    
#     # left half: list1
#     vp1 = ax.violinplot([list1], positions=[pos], widths=width, showextrema=False)
#     _clip_violin_body(vp1["bodies"][0], pos, "left")
    
#     # right half: list2
#     vp2 = ax.violinplot([list2], positions=[pos], widths=width, showextrema=False)
#     _clip_violin_body(vp2["bodies"][0], pos, "right")


#     # Color the violins
#     for i, body in enumerate(vp['bodies']):
#         body.set_color(colors[i])
#         body.set_edgecolor('none')
#         body.set_alpha(0.7)

#     x1, x2 = positions

#     # Add individual data points
#     ax.plot([x1] * len(list1), list1, 'o', color=colors[0], alpha=0.4, markersize=markersize, zorder=3)
#     ax.plot([x2] * len(list2), list2, 'o', color=colors[1], alpha=0.4, markersize=markersize, zorder=3)

#     # Lines connecting paired points
#     if show_pair_lines:
#         for y1, y2 in zip(list1, list2):
#             ax.plot([x1, x2], [y1, y2], color='0.6', alpha=0.4, linewidth=0.8, zorder=2)

#     # Median markers
#     ax.scatter(x1, np.mean(list1), s=30, c=colors[0], alpha=.8, zorder=4)
#     ax.scatter(x2, np.mean(list2), s=30, c=colors[1], alpha=.8, zorder=4)

#     ax.set_xlim(0.5, 2.5)

#     # X labels and title text base
#     if isinstance(colname, list) and len(colname) == 2:
#         ax.set_xticks(positions)
#         ax.set_xticklabels(colname)
#         plot_title = f"{colname[0]} vs {colname[1]}"
#     elif colname:
#         ax.set_xticks(positions)
#         ax.set_xticklabels(['Cond 1', 'Cond 2'])
#         plot_title = str(colname)
#     else:
#         ax.set_xticks(positions)
#         ax.set_xticklabels(['Cond 1', 'Cond 2'])
#         plot_title = "Paired Comparison"

#     if ylabel:
#         ax.set_ylabel(ylabel)
#     if ylim:
#         ax.set_ylim(ylim)

#     # Paired statistical tests
#     t_stat, t_pval = ttest_rel(list1, list2)

#     try:
#         w_stat, w_pval = wilcoxon(list1, list2)
#     except ValueError:
#         # e.g. all differences are zero -> Wilcoxon not defined
#         w_stat, w_pval = np.nan, np.nan

#     # Title with stats
#     if np.isnan(w_pval):
#         stat_str = f"t-rel p={t_pval:.4g}, Wilcoxon p=nan"
#     else:
#         stat_str = f"t-rel p={t_pval:.4g}, Wilcoxon p={w_pval:.4g}"

#     if title_prefix:
#         title = f"{title_prefix}: {plot_title}\n{stat_str}"
#     else:
#         title = f"{plot_title}\n{stat_str}"

#     ax.set_title(title, fontsize=10)

#     # Sample size (number of pairs)
#     n_pairs = len(list1)
#     ax.text(x1, ax.get_ylim()[0], f'n={n_pairs}', ha='center', va='top', fontsize=8)
#     ax.text(x2, ax.get_ylim()[0], f'n={n_pairs}', ha='center', va='top', fontsize=8)

#     # Clean up spines
#     for spine in ['top', 'right', 'bottom']:
#         ax.spines[spine].set_visible(False)
#     # plt.show()
#     if ax is None:
#         return fig, ax

def plot_unpaired_violin(list1, list2,
                         m_point='mean',
                         colors=['steelblue', 'orange'],
                         colname=None, ylabel=None, ylim=None,
                         title_prefix=None,
                         ax=None,
                         markersize=3,
                         print_median_iqr=False):
    """
    Plot unpaired violin plots comparing two independent 1D lists.

    Parameters:
        list1 (array-like): shape (n1,), first group
        list2 (array-like): shape (n2,), second group  
        colname (str or list of str): Used for x-axis labels and plot title.
        ylabel (str): Label for y-axis.
        ylim (tuple): Y-axis limits.
        title_prefix (str): Optional prefix for the plot title.
    """
    # from scipy.stats import ttest_ind, ranksums
    
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)
    
    return_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
        return_ax = True
    
    data = [list1, list2]
    positions = [1, 2]
    
    # Full violin plots (not half)
    vp = ax.violinplot(data, positions=positions, showextrema=False)
    
    # Color the violins (matching paired style)
    for i, body in enumerate(vp['bodies']):
        body.set_color(colors[i])
        body.set_edgecolor('none')
    
    # Add individual data points (matching paired style)
    x1, x2 = positions
    ax.plot([x1]*len(list1), list1, 'o', color=colors[0], alpha=0.3, markersize=markersize)
    ax.plot([x2]*len(list2), list2, 'o', color=colors[1], alpha=0.3, markersize=markersize)
    
    # Add m markers (matching paired style)
    if m_point == 'mean':
        m1 = np.mean(list1)
        m2 = np.mean(list2)
    elif m_point == 'median':
        m1 = np.median(list1)
        m2 = np.median(list2)
    
    ax.scatter(x1, m1, s=30, c=colors[0], alpha=.6, zorder=3)
    ax.scatter(x2, m2, s=30, c=colors[1], alpha=.6, zorder=3)
    
    ax.set_xlim(0.5, 2.5)
    if isinstance(colname, list) and len(colname) == 2:
        ax.set_xticks(positions)
        ax.set_xticklabels(colname)
        plot_title = f"{colname[0]} vs {colname[1]}"
    elif colname:
        ax.set_xticks(positions)
        ax.set_xticklabels(['Group 1', 'Group 2'])
        plot_title = str(colname)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(['Group 1', 'Group 2'])
        plot_title = "Unpaired Comparison"
    
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    
    # Perform statistical tests
    t_stat, t_pval = ttest_ind(list1, list2)
    rs_stat, rs_pval = ranksums(list1, list2)
    
    # Add title with statistics
    if title_prefix:
        title = f"{title_prefix}: {plot_title}\nt-test p={t_pval:.4f}, rank-sum p={rs_pval:.4f}"
    else:
        title = f"{plot_title}\nt-test p={t_pval:.4f}, rank-sum p={rs_pval:.4f}"
    ax.set_title(title, fontsize=10)
    
    # Add sample sizes
    ax.text(x1, ax.get_ylim()[0], f'n={len(list1)}', ha='center', va='top', fontsize=8)
    ax.text(x2, ax.get_ylim()[0], f'n={len(list2)}', ha='center', va='top', fontsize=8)

    # optional median + IQR (print + annotate below each violin)
    if print_median_iqr:
        med1 = np.nanmedian(list1); q1_1, q3_1 = np.nanpercentile(list1, [25, 75])
        med2 = np.nanmedian(list2); q1_2, q3_2 = np.nanpercentile(list2, [25, 75])
        lab1 = colname[0] if isinstance(colname, (list, tuple)) and len(colname) == 2 else 'Group 1'
        lab2 = colname[1] if isinstance(colname, (list, tuple)) and len(colname) == 2 else 'Group 2'
        print(f"{lab1}: median={med1:.4f}, IQR=[{q1_1:.4f}, {q3_1:.4f}] (n={len(list1)}) | "
              f"{lab2}: median={med2:.4f}, IQR=[{q1_2:.4f}, {q3_2:.4f}] (n={len(list2)})")
        trans = ax.get_xaxis_transform()
        ax.text(x1, -0.10, f"med={med1:.3f}\n[{q1_1:.3f}, {q3_1:.3f}]",
                transform=trans, ha='center', va='top',
                fontsize=7, color=colors[0], clip_on=False)
        ax.text(x2, -0.10, f"med={med2:.3f}\n[{q1_2:.3f}, {q3_2:.3f}]",
                transform=trans, ha='center', va='top',
                fontsize=7, color=colors[1], clip_on=False)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    # plt.show()
    if return_ax:
        return fig, ax

def plot_unpaired_boxplot(list1, list2, colors=['steelblue', 'orange'],
                          colname=None, ylabel=None, ylim=None,
                          title_prefix=None, show_points=True, jitter=0.08):
    """
    Box-plot version of your unpaired violin plot.

    Parameters:
        list1, list2: 1D array-like, independent groups
        colors (list[str]): [color_group1, color_group2]
        colname (str or list[str]): x labels / title text (same behavior as yours)
        ylabel (str): y-axis label
        ylim (tuple): (ymin, ymax)
        title_prefix (str): optional prefix for the plot title
        show_points (bool): overlay individual data points
        jitter (float): horizontal jitter for points
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind, ranksums

    list1 = np.asarray(list1).ravel()
    list2 = np.asarray(list2).ravel()

    # Drop NaNs to keep stats/plot stable
    l1 = list1[np.isfinite(list1)]
    l2 = list2[np.isfinite(list2)]

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)

    data = [l1, l2]
    positions = [1, 2]

    # Core boxplot (colored per group)
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,   # allows facecolor on boxes
        showfliers=False     # hide outliers to keep plot clean (points are shown separately)
    )

    # Color and style each element to match group colors
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[i], edgecolor=colors[i], alpha=0.35, linewidth=1.5)

    # Whiskers/caps/medians are returned in order: 2 per box
    for i in range(2):
        # whiskers
        bp['whiskers'][2*i + 0].set(color=colors[i], linewidth=1.5)
        bp['whiskers'][2*i + 1].set(color=colors[i], linewidth=1.5)
        # caps
        bp['caps'][2*i + 0].set(color=colors[i], linewidth=1.5)
        bp['caps'][2*i + 1].set(color=colors[i], linewidth=1.5)

    for i, med in enumerate(bp['medians']):
        med.set(color=colors[i], linewidth=2)

    # Overlay individual data points
    if show_points:
        rng = np.random.default_rng()
        x1 = rng.normal(positions[0], jitter, size=len(l1))
        x2 = rng.normal(positions[1], jitter, size=len(l2))
        ax.scatter(x1, l1, s=9, alpha=0.5, facecolor=colors[0], edgecolor='none', zorder=2)
        ax.scatter(x2, l2, s=9, alpha=0.5, facecolor=colors[1], edgecolor='none', zorder=2)

    # Optional median dot for emphasis (keeps your original vibe)
    ax.scatter(positions[0], np.median(l1), s=30, c=colors[0], alpha=.8, zorder=3)
    ax.scatter(positions[1], np.median(l2), s=30, c=colors[1], alpha=.8, zorder=3)

    # Axes and labels
    ax.set_xlim(0.5, 2.5)
    if isinstance(colname, list) and len(colname) == 2:
        ax.set_xticks(positions)
        ax.set_xticklabels(colname)
        plot_title = f"{colname[0]} vs {colname[1]}"
    elif colname:
        ax.set_xticks(positions)
        ax.set_xticklabels(['Group 1', 'Group 2'])
        plot_title = str(colname)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(['Group 1', 'Group 2'])
        plot_title = "Unpaired Comparison"

    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)

    # Stats (Welch's t-test for robustness) + rank-sum
    t_stat, t_pval = ttest_ind(l1, l2, equal_var=False)
    rs_stat, rs_pval = ranksums(l1, l2)

    if title_prefix:
        title = f"{title_prefix}: {plot_title}\nt-test p={t_pval:.4f}, rank-sum p={rs_pval:.4f}"
    else:
        title = f"{plot_title}\nt-test p={t_pval:.4f}, rank-sum p={rs_pval:.4f}"
    ax.set_title(title, fontsize=10)

    # Sample sizes at the bottom edge
    ymin, _ = ax.get_ylim()
    ax.text(positions[0], ymin, f'n={len(l1)}', ha='center', va='top', fontsize=8)
    ax.text(positions[1], ymin, f'n={len(l2)}', ha='center', va='top', fontsize=8)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    # plt.show()
    return fig, ax

# def plot_overlay_1bar(list1, list2,
#                      colname=['list1', 'list2'],
#                      colors=('steelblue', 'orange'),
#                      ylabel=None, ylim=None, title=None,
#                      jitter=0.06, bar_width=0.5, point_size=10,
#                      err='sem',
#                      annotation=True):
#     """
#     Single-column bar (list2) with error bar, overlaid with:
#       - list2 points (same column)
#       - list1 points (same column)
#       - list1 median line

#     Parameters
#     ----------
#     list1, list2 : 1D array-like
#         list1 supplies overlaid points + median; list2 drives the bar + error bar (+ points).
#     colors : (str, str)
#         (color_list1_points_and_median, color_list2_bar_and_points)
#     ylabel : str
#         Y-axis label.
#     ylim : (float, float)
#         Y-axis limits.
#     title : str
#         Plot title.
#     jitter : float
#         Horizontal jitter for scatter points (both groups share x=1).
#     bar_width : float
#         Width of the bar (for list2).
#     point_size : float
#         Scatter marker size.
#     err : {'sem', 'std', None}
#         Error bar type for list2.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # sanitize
#     l1 = np.asarray(list1).ravel()
#     l2 = np.asarray(list2).ravel()
#     l1 = l1[np.isfinite(l1)]
#     l2 = l2[np.isfinite(l2)]

#     if l2.size == 0:
#         raise ValueError("list2 has no finite values; cannot draw bar.")
#     if l1.size == 0:
#         # We'll allow no list1 points; just draw bar
#         pass

#     # summary stats for list2
#     mean2 = float(np.mean(l2))
#     if err == 'sem':
#         se2 = float(np.std(l2, ddof=1) / np.sqrt(len(l2))) if len(l2) > 1 else 0.0
#     elif err == 'std':
#         se2 = float(np.std(l2, ddof=1)) if len(l2) > 1 else 0.0
#     elif err is None:
#         se2 = None
#     else:
#         raise ValueError("err must be one of {'sem', 'std', None}")

#     # median for list1
#     median1 = float(np.median(l1)) if l1.size else None

#     fig, ax = plt.subplots(figsize=(2.8, 3.2), dpi=300)
#     x = .3

#     # --- draw bar for list2 ---
#     bar = ax.bar([x], [mean2], width=bar_width,
#                  facecolor=colors[1], alpha=0.35, edgecolor='none', linewidth=1.2,
#                  zorder=1)
#     if se2 is not None:
#         ax.errorbar([x], [mean2], yerr=[se2], fmt='none',
#                     elinewidth=2, capsize=3, capthick=1.2,
#                     ecolor=colors[1], zorder=2)

#     # --- overlay points (both lists share x=1 with small jitter) ---
#     rng = np.random.default_rng()
#     # list2 points
#     x2 = rng.normal(x, jitter, size=len(l2))
#     ax.scatter(x2, l2, s=point_size, alpha=0.6, edgecolor='none', facecolor=colors[1], zorder=3, label=f'{colname[1]}_n={len(l2)}')

#     # list1 points
#     if l1.size:
#         x1 = rng.normal(x, jitter, size=len(l1))
#         ax.scatter(x1, l1, s=point_size, alpha=0.7, edgecolor='none', facecolor=colors[0], zorder=4, label=f'{colname[0]}_n={len(l1)}')

#         # median line for list1 across the bar width
#         ax.hlines(median1, x - bar_width*0.65, x + bar_width*0.65,
#                   color=colors[0], linewidth=2, zorder=5)

#     # cosmetics
#     ax.set_xlim(0, 1)
#     ax.set_xticks([x])
#     ax.set_xticklabels([''])
#     if ylabel:
#         ax.set_ylabel(ylabel)
#     if ylim:
#         ax.set_ylim(ylim)

#     # n annotations
#     # ymin, _ = ax.get_ylim()
#     # ax.text(x - bar_width*0.2, ymin, f"n2={len(l2)}", ha='center', va='top', fontsize=8, color=colors[1])
#     # if l1.size:
#     #     ax.text(x + bar_width*0.2, ymin, f"n1={len(l1)}", ha='center', va='top', fontsize=8, color=colors[0])

#     # light cleanup
#     for spine in ['top', 'right', 'bottom']:
#         ax.spines[spine].set_visible(False)

#     if title:
#         ax.set_title(title, fontsize=10)

#     # optional legend (comment out if not needed)
#     ax.legend(frameon=False, fontsize=8, loc='best')

#     fig.tight_layout()
#     return fig, ax

def plot_overlay_1bar(list1, list2,
                     colname=('list1', 'list2'),
                     colors=('grey', 'orange'),
                     ylabel=None, ylim=None, title=None,
                     jitter=0.06, bar_width=0.5, point_size=10,
                     err='sem',
                     annotation=True):
    """
    Single-column bar (list2) with error bar, overlaid with:
      - list2 points (same column)
      - list1 points (same column)
      - list1 median line

    Statistics (list1 vs list2):
      - Independent: Welch's t (ttest_ind with equal_var=False), Rank-sum (scipy.stats.ranksums)
      - Paired (only if len(list1) == len(list2) > 1): ttest_rel, Wilcoxon signed-rank

    Parameters
    ----------
    list1, list2 : 1D array-like
        list1 supplies overlaid points + median; list2 drives the bar + error bar (+ points).
    colors : (str, str)
        (color_list1_points_and_median, color_list2_bar_and_points)
    ylabel : str
        Y-axis label.
    ylim : (float, float)
        Y-axis limits.
    title : str
        Plot title.
    jitter : float
        Horizontal jitter for scatter points (both groups share x=1).
    bar_width : float
        Width of the bar (for list2).
    point_size : float
        Scatter marker size.
    err : {'sem', 'std', None}
        Error bar type for list2.
    annotation : bool
        If True, annotates mean±SEM and test statistics on the plot.

    Returns
    -------
    fig, ax, stats_dict
    """
    def _sem(x):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        n = x.size
        return (np.nan if n <= 1 else float(np.std(x, ddof=1) / np.sqrt(n)))

    def _fmt_p(p):
        if p is None or np.isnan(p):
            return "n/a"
        if p < 1e-4:
            return f"{p:.1e}"
        return f"{p:.4f}"

    # sanitize
    l1 = np.asarray(list1).ravel()
    l2 = np.asarray(list2).ravel()
    l1 = l1[np.isfinite(l1)]
    l2 = l2[np.isfinite(l2)]

    if l2.size == 0:
        raise ValueError("list2 has no finite values; cannot draw bar.")

    # summary stats
    mean1 = float(np.mean(l1)) if l1.size else np.nan
    sem1  = _sem(l1) if l1.size else np.nan
    mean2 = float(np.mean(l2))
    if err == 'sem':
        se2 = _sem(l2)
    elif err == 'std':
        se2 = (np.nan if len(l2) <= 1 else float(np.std(l2, ddof=1)))
    elif err is None:
        se2 = None
    else:
        raise ValueError("err must be one of {'sem', 'std', None}")

    # run stats
    stats = {
        'n1': int(l1.size),
        'n2': int(l2.size),
        'mean1': mean1, 'sem1': sem1,
        'mean2': mean2, 'sem2_for_bar': se2 if err == 'sem' else np.nan,
    }

    # independent tests (need at least one per group; meaningful with n>=2)
    if l1.size >= 1 and l2.size >= 1:
        # try:
        #     ti = ttest_ind(l1, l2, equal_var=False, nan_policy='omit')
        #     stats['ttest_ind'] = {'stat': float(ti.statistic), 'p': float(ti.pvalue)}
        # except Exception:
        #     stats['ttest_ind'] = None
        try:
            # Rank-sum test (asymptotic normal approx), two-sided by default
            rs = ranksums(l1, l2)
            stats['ranksums'] = {'z': float(rs.statistic), 'p': float(rs.pvalue)}
        except Exception:
            stats['ranksums'] = None
    else:
        # stats['ttest_ind'] = None
        stats['ranksums'] = None

    # paired tests only if same length and >1
    # if l1.size == l2.size and l1.size > 1:
    #     try:
    #         tr = ttest_rel(l1, l2, nan_policy='omit')
    #         stats['ttest_rel'] = {'stat': float(tr.statistic), 'p': float(tr.pvalue)}
    #     except Exception:
    #         stats['ttest_rel'] = None
    #     try:
    #         wz = wilcoxon(l1, l2, alternative='two-sided', zero_method='wilcox')
    #         stats['wilcoxon'] = {'W': float(wz.statistic), 'p': float(wz.pvalue)}
    #     except Exception:
    #         stats['wilcoxon'] = None
    # else:
    #     stats['ttest_rel'] = None
    #     stats['wilcoxon'] = None

    # ---- plotting ----
    fig, ax = plt.subplots(figsize=(2.8, 3.2), dpi=300)
    x = 0.3

    # bar for list2
    ax.bar([x], [mean2], width=bar_width,
           facecolor=colors[1], alpha=0.35, edgecolor='none', linewidth=1.2,
           zorder=1)
    if se2 is not None and np.isfinite(se2):
        ax.errorbar([x], [mean2], yerr=[se2], fmt='none',
                    elinewidth=2, capsize=3, capthick=1.2,
                    ecolor=colors[1], zorder=2)

    rng = np.random.default_rng()
    # list2 points
    x2 = rng.normal(x, jitter, size=len(l2))
    ax.scatter(x2, l2, s=point_size, alpha=0.6, edgecolor='none',
               facecolor=colors[1], zorder=3, label=f'{colname[1]}_n={len(l2)}')

    # list1 points + median line  # 2/5/26 remove scatter points for shuffle data
    if l1.size:
        x1 = rng.normal(x, jitter, size=len(l1))
        # ax.scatter(x1, l1, s=point_size, alpha=0.7, edgecolor='none',
        #            facecolor=colors[0], zorder=4, label=f'{colname[0]}_n={len(l1)}')
        median1 = float(np.median(l1))
        ax.hlines(median1, x - bar_width*0.65, x + bar_width*0.65,
                  color=colors[0], linewidth=2, zorder=5)

    # cosmetics
    ax.set_xlim(0, 1)
    ax.set_xticks([x]); ax.set_xticklabels([''])
    if ylabel: ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(ylim)
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    if title: ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='best')

    # --- annotation block ---
    if annotation:
        y_min, y_max = ax.get_ylim()
        y_rng = y_max - y_min
        y_text = y_max - 0.03 * y_rng  # near top-right
        txt_lines = []

        # means & SEMs
        m1_txt = "n/a" if not np.isfinite(mean1) else f"{mean1:.4g}"
        s1_txt = "n/a" if not np.isfinite(sem1)  else f"{sem1:.4g}"
        m2_txt = f"{mean2:.4g}"
        s2_txt = ("n/a" if (se2 is None or not np.isfinite(se2)) else f"{se2:.4g}")

        txt_lines.append(f"{colname[0]}: mean={m1_txt}, SEM={s1_txt} (n={len(l1)})")
        txt_lines.append(f"{colname[1]}: mean={m2_txt}, SEM={s2_txt} (n={len(l2)})")

       
        # ttest_ind
        # if stats['ttest_ind'] is not None:
        #     txt_lines.append(f"ttest_ind: t={stats['ttest_ind']['stat']:.2f}, p={_fmt_p(stats['ttest_ind']['p'])}")
        # else:
        #     txt_lines.append("ttest_ind: n/a")
        # ranksums (independent rank-sum)
        if stats['ranksums'] is not None:
            txt_lines.append(f"rank-sum (ranksums): z={stats['ranksums']['z']:.3f}, p={_fmt_p(stats['ranksums']['p'])}")
        else:
            txt_lines.append("rank-sum (ranksums): n/a")
        # # paired t
        # if stats['ttest_rel'] is not None:
        #     txt_lines.append(f"ttest_rel (paired): t={stats['ttest_rel']['stat']:.2f}, p={_fmt_p(stats['ttest_rel']['p'])}")
        # else:
        #     txt_lines.append("ttest_rel (paired): n/a")
        # # Wilcoxon signed-rank
        # if stats['wilcoxon'] is not None:
        #     txt_lines.append(f"Wilcoxon (paired): W={stats['wilcoxon']['W']:.2f}, p={_fmt_p(stats['wilcoxon']['p'])}")
        # else:
        #     txt_lines.append("Wilcoxon (paired): n/a")


        ax.text(0.98, y_text,
                "\n".join(txt_lines),
                ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.6, edgecolor='none'))

    fig.tight_layout()
    return fig, ax, stats


# def plot_overlay_2bars(
#     list2, list3,               # real data for bar 1 and bar 2
#     list1=None, list4=None,     # optional shuffle data for bar 1 and bar 2
#     colnames=('condA', 'condB'),            # labels for the two bars
#     colors12=('steelblue', 'orange'),       # (shuffle1, real2)
#     colors34=('mediumpurple', 'seagreen'),  # (shuffle4, real3)
#     ylabel=None, ylim=None, title=None,
#     jitter=0.06, bar_width=0.45, point_size=15,
#     err='sem', show_legend=True, seed=None
# ):
#     """
#     Draw two single-column bars (for list2 and list3), each with:
#       - bar + error bar from the real data (list2 or list3)
#       - overlaid real-data points (same column)
#       - optional shuffle-data points + median line (list1 for bar1, list4 for bar2)

#     Parameters
#     ----------
#     list2, list3 : 1D array-like
#         Real data for bar 1 and bar 2.
#     list1, list4 : 1D array-like or None
#         Shuffle data corresponding to list2 and list3. If None or empty (after
#         removing non-finite), the shuffle overlay is skipped for that bar.
#     colnames : (str, str)
#         Labels for bar 1 and bar 2 (used in legend).
#     colors12 : (str, str)
#         (color_shuffle_bar1_points_and_median, color_real_bar1_bar_and_points).
#     colors34 : (str, str)
#         (color_shuffle_bar2_points_and_median, color_real_bar2_bar_and_points).
#     ylabel : str
#     ylim : (float, float)
#     title : str
#     jitter : float
#     bar_width : float
#     point_size : float
#     err : {'sem','std',None}
#         Error bar type for bars (real data only).
#     show_legend : bool
#     seed : int or None
#         Seed for jitter RNG (for reproducible scatter).
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     def _sanitize(arr):
#         if arr is None:
#             return np.array([], dtype=float)
#         a = np.asarray(arr).ravel()
#         return a[np.isfinite(a)]

#     def _err(y, mode):
#         if mode is None:
#             return None
#         n = len(y)
#         if n <= 1:
#             return 0.0
#         if mode == 'sem':
#             return float(np.std(y, ddof=1) / np.sqrt(n))
#         if mode == 'std':
#             return float(np.std(y, ddof=1))
#         raise ValueError("err must be one of {'sem','std',None}")

#     # sanitize inputs
#     l2 = _sanitize(list2)
#     l3 = _sanitize(list3)
#     l1 = _sanitize(list1)
#     l4 = _sanitize(list4)

#     if l2.size == 0 and l3.size == 0:
#         raise ValueError("Both list2 and list3 are empty after sanitization.")
#     if l2.size == 0:
#         raise ValueError("list2 has no finite values; cannot draw first bar.")
#     if l3.size == 0:
#         raise ValueError("list3 has no finite values; cannot draw second bar.")

#     # stats
#     mean2, mean3 = float(np.mean(l2)), float(np.mean(l3))
#     se2, se3 = _err(l2, err), _err(l3, err)
#     med1 = float(np.median(l1)) if l1.size else None
#     med4 = float(np.median(l4)) if l4.size else None

#     # positions
#     x1, x2 = 0.5, 1.25

#     # RNG
#     rng = np.random.default_rng(seed)

#     fig, ax = plt.subplots(figsize=(2, 3), dpi=300)

#     # --- Bar 1 (list2 real) ---
#     bar1 = ax.bar([x1], [mean2], width=bar_width,
#                   color=colors12[1], alpha=0.5,
#                   edgecolor='none', linewidth=1.2, zorder=1)
#     if se2 is not None:
#         ax.errorbar([x1], [mean2], yerr=[se2], fmt='none',
#                     elinewidth=2, capsize=3, capthick=1.2,
#                     ecolor=colors12[1], zorder=2)
#     # points real (list2)
#     ax.scatter(rng.normal(x1, jitter, size=len(l2)), l2,
#                s=point_size, alpha=0.7, 
#                facecolor=colors12[1], edgecolor='none',
#                zorder=3,
#                label=f'{colnames[0]} (real) n={len(l2)}')
#     # optional shuffle (list1)
#     if l1.size:
#         ax.scatter(rng.normal(x1, jitter, size=len(l1)), l1,
#                    s=point_size, alpha=0.6, 
#                    facecolor=colors12[0], edgecolor='none',
#                    zorder=4,
#                    label=f'{colnames[0]} (shuffle) n={len(l1)}')
#         ax.hlines(med1, x1 - bar_width*0.65, x1 + bar_width*0.65,
#                   color=colors12[0], linewidth=2, zorder=5)

#     # --- Bar 2 (list3 real) ---
#     bar2 = ax.bar([x2], [mean3], width=bar_width,
#                   color=colors34[1], alpha=0.8,
#                   edgecolor='none', linewidth=1.2, zorder=1)
#     if se3 is not None:
#         ax.errorbar([x2], [mean3], yerr=[se3], fmt='none',
#                     elinewidth=2, capsize=3, capthick=1.2,
#                     ecolor=colors34[1], zorder=2)
#     # points real (list3)
#     ax.scatter(rng.normal(x2, jitter, size=len(l3)), l3,
#                s=point_size, alpha=0.7, 
#                facecolor=colors34[1], 
#                edgecolor='none',
#                zorder=3,
#                label=f'{colnames[1]} (real) n={len(l3)}')
#     # optional shuffle (list4)
#     if l4.size:
#         ax.scatter(rng.normal(x2, jitter, size=len(l4)), l4,
#                    s=point_size, alpha=0.6, 
#                    facecolor=colors34[0], edgecolor='none',
#                    zorder=4,
#                    label=f'{colnames[1]} (shuffle) n={len(l4)}')
#         ax.hlines(med4, x2 - bar_width*0.65, x2 + bar_width*0.65,
#                   color=colors34[0], linewidth=2, zorder=5)

#     # cosmetics
#     ax.set_xlim(0, 1.5)
#     ax.set_xticks([x1, x2])
#     ax.set_xticklabels(colnames, rotation=0)
#     if ylabel:
#         ax.set_ylabel(ylabel)
#     if ylim:
#         ax.set_ylim(ylim)

#     for spine in ['top', 'right', 'bottom']:
#         ax.spines[spine].set_visible(False)

#     if title:
#         ax.set_title(title, fontsize=10)

#     if show_legend:
#         ax.legend(frameon=False, fontsize=8, loc='best')

#     return fig, ax

def plot_overlay_2bars(
    list2, list3,               # real data for bar 1 and bar 2
    list1=None, list4=None,     # optional shuffle data for bar 1 and bar 2
    colnames=('condA', 'condB'),            # labels for the two bars
    colors12=('steelblue', 'orange'),       # (shuffle1, real2)
    colors34=('mediumpurple', 'seagreen'),  # (shuffle4, real3)
    ylabel=None, ylim=None, title=None,
    jitter=0.06, bar_width=0.45, point_size=15,
    err='sem', show_legend=True, seed=None
):
    """
    Two bars (list2, list3) with:
      - bar + error bar from real data
      - overlaid real-data points
      - optional shuffle points + median line
      - text annotations for mean±SEM (real and shuffle) with 3 decimals
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def _sanitize(arr):
        if arr is None:
            return np.array([], dtype=float)
        a = np.asarray(arr).ravel().astype(float)
        return a[np.isfinite(a)]

    def _err(y, mode):
        if mode is None:
            return None
        n = len(y)
        if n <= 1:
            return 0.0
        if mode == 'sem':
            return float(np.std(y, ddof=1) / np.sqrt(n))
        if mode == 'std':
            return float(np.std(y, ddof=1))
        raise ValueError("err must be one of {'sem','std',None}")

    def _mean_sem(y, mode):
        if y.size == 0:
            return None, None
        m = float(np.mean(y))
        s = _err(y, mode)
        if s is None:
            s = 0.0
        return m, float(s)

    # sanitize inputs
    l2 = _sanitize(list2)  # real bar 1
    l3 = _sanitize(list3)  # real bar 2
    l1 = _sanitize(list1)  # shuffle for bar 1 (optional)
    l4 = _sanitize(list4)  # shuffle for bar 2 (optional)

    if l2.size == 0 and l3.size == 0:
        raise ValueError("Both list2 and list3 are empty after sanitization.")
    if l2.size == 0:
        raise ValueError("list2 has no finite values; cannot draw first bar.")
    if l3.size == 0:
        raise ValueError("list3 has no finite values; cannot draw second bar.")

    # stats
    mean2, se2 = _mean_sem(l2, err)
    mean3, se3 = _mean_sem(l3, err)
    med1 = float(np.median(l1)) if l1.size else None
    med4 = float(np.median(l4)) if l4.size else None
    # shuffle mean±sem
    shuf_mean1, shuf_se1 = _mean_sem(l1, err) if l1.size else (None, None)
    shuf_mean4, shuf_se4 = _mean_sem(l4, err) if l4.size else (None, None)

    # positions
    x1, x2 = 0.5, 1.25

    # RNG
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(2, 3), dpi=300)

    # --- Bar 1 (list2 real) ---
    ax.bar([x1], [mean2], width=bar_width,
           color=colors12[1], alpha=0.5,
           edgecolor='none', linewidth=1.2, zorder=1)
    if se2 is not None:
        ax.errorbar([x1], [mean2], yerr=[se2], fmt='none',
                    elinewidth=2, capsize=3, capthick=1.2,
                    ecolor='black', zorder=2)
    # points real (list2)
    ax.scatter(rng.normal(x1, jitter, size=len(l2)), l2,
               s=point_size, alpha=0.7,
               facecolor=colors12[1], edgecolor='none', zorder=3,
               label=f'{colnames[0]} (real) n={len(l2)}')

    # optional shuffle (list1)
    if l1.size:
        ax.scatter(rng.normal(x1, jitter, size=len(l1)), l1,
                   s=point_size, alpha=0.6,
                   facecolor=colors12[0], edgecolor='none', zorder=4,
                   label=f'{colnames[0]} (shuffle) n={len(l1)}')
        ax.hlines(med1, x1 - bar_width*0.65, x1 + bar_width*0.65,
                  color=colors12[0], linewidth=2, zorder=5)

    # --- Bar 2 (list3 real) ---
    ax.bar([x2], [mean3], width=bar_width,
           color=colors34[1], alpha=0.8,
           edgecolor='none', linewidth=1.2, zorder=1)
    if se3 is not None:
        ax.errorbar([x2], [mean3], yerr=[se3], fmt='none',
                    elinewidth=2, capsize=3, capthick=1.2,
                    ecolor='black', zorder=2)
    # points real (list3)
    ax.scatter(rng.normal(x2, jitter, size=len(l3)), l3,
               s=point_size, alpha=0.7,
               facecolor=colors34[1], edgecolor='none', zorder=3,
               label=f'{colnames[1]} (real) n={len(l3)}')

    # optional shuffle (list4)
    if l4.size:
        ax.scatter(rng.normal(x2, jitter, size=len(l4)), l4,
                   s=point_size, alpha=0.6,
                   facecolor=colors34[0], edgecolor='none', zorder=4,
                   label=f'{colnames[1]} (shuffle) n={len(l4)}')
        ax.hlines(med4, x2 - bar_width*0.65, x2 + bar_width*0.65,
                  color=colors34[0], linewidth=2, zorder=5)

    # ---- Auto ylim padding (only if not provided) ----
    if ylim is None:
        y_all = []
        y_all.extend(l2.tolist()); y_all.extend(l3.tolist())
        if l1.size: y_all.extend(l1.tolist())
        if l4.size: y_all.extend(l4.tolist())
        # include bars+errors
        y_all.extend([mean2 + (se2 or 0.0), mean3 + (se3 or 0.0)])
        y_all.extend([mean2, mean3])
        if len(y_all) == 0:
            y_lo, y_hi = 0.0, 1.0
        else:
            y_lo, y_hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        rng_y = max(1e-6, y_hi - y_lo)
        pad_top = 0.20 * rng_y  # room for annotations
        pad_bot = 0.05 * rng_y
        ylim = (min(0.0, y_lo) - pad_bot, y_hi + pad_top)

    # cosmetics
    ax.set_xlim(0, 1.5)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(colnames, rotation=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)

    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc='best')

    # ---- Text annotations: mean ± SEM (3 decimals) ----
    # small vertical bump relative to current axis range
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    bump = 0.02 * yrange
    def _fmt3(x): return f"{x:.3f}"

    # Real bars (above bar top)
    ax.text(x1, mean2 + (se2 or 0.0) + bump,
            f"{_fmt3(mean2)} ± {_fmt3(se2 or 0.0)}",
            ha='center', va='bottom', fontsize=7, color=colors12[1])
    ax.text(x2, mean3 + (se3 or 0.0) + bump,
            f"{_fmt3(mean3)} ± {_fmt3(se3 or 0.0)}",
            ha='center', va='bottom', fontsize=7, color=colors34[1])

    # Shuffle means (near shuffle median line if available; otherwise above shuffle cloud center)
    if l1.size and (shuf_mean1 is not None):
        y_anchor = med1 if med1 is not None else shuf_mean1
        ax.text(x1, y_anchor + bump,
                f"shuf: {_fmt3(shuf_mean1)} ± {_fmt3(shuf_se1 or 0.0)}",
                ha='center', va='bottom', fontsize=7, color=colors12[0])
    if l4.size and (shuf_mean4 is not None):
        y_anchor = med4 if med4 is not None else shuf_mean4
        ax.text(x2, y_anchor + bump,
                f"shuf: {_fmt3(shuf_mean4)} ± {_fmt3(shuf_se4 or 0.0)}",
                ha='center', va='bottom', fontsize=7, color=colors34[0])

    return fig, ax
#%% plot FOVs
import math

def scaler(actual_len_um, zoom):
    """
    Convert a physical length (µm) to pixels given the zoom.
    NOTE: using your zoom==3 calibration (512 px ↔ 321 µm). 
    Extend the elifs if you have other zooms.
    """
    if zoom == 3:
        return (512 / 321.0) * actual_len_um
    raise ValueError(f"Unsupported zoom={zoom}. Add your calibration above.")

def scalbar(ax, actual_len_um=50, zoom=3,
            x_right_px=None, y_bottom_px=None,
            pad_px=10,                 # base inset from edges
            bar_offset_y=20,           # extra lift from bottom (px)
            bar_offset_x=20,            # extra inset from right (px)
            label=None, color='white', lw=1,
            label_position="above",    # "above" or "below"
            shrink_to_fit=True):
    """
    Draw a scale bar near the bottom-right of the current axes,
    respecting crop windows and preventing overflow.

    - pad_px:       base padding from right/bottom (≥0 OK)
    - bar_offset_y: additional vertical lift from bottom (px)
    - bar_offset_x: additional horizontal inset from right (px)
    - shrink_to_fit: if True, reduces bar length to fit if needed
    """
    # length in pixels from your zoom calibration
    pix_len = scaler(actual_len_um, zoom)
    if label is None:
        label = f"{actual_len_um} µm"

    # axis limits (handle inverted axes)
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()
    x_left, x_right = min(x0, x1), max(x0, x1)
    y_top,  y_bot   = min(y0, y1), max(y0, y1)

    # defaults to the visible box edges if not provided
    if x_right_px is None: x_right_px = x_right
    if y_bottom_px is None: y_bottom_px = y_bot

    # keep at least 1 px inside so it doesn't clip even when pad_px=0
    pad = max(pad_px, 1)

    # target right edge of the bar after padding + extra inset
    target_right = x_right_px - pad - bar_offset_x

    # available horizontal space from left edge to target_right
    avail_w = (target_right - (x_left + 1))
    if shrink_to_fit and pix_len > avail_w:
        # shrink if it wouldn't fit; minimum 5 px to remain visible
        pix_len = max(5, avail_w)

    # place bar flush to the (inset) right
    xmin = target_right - pix_len
    xmax = target_right

    # if still out of bounds (e.g., extreme crop), clamp inside
    if xmin < x_left + 1:
        shift = (x_left + 1) - xmin
        xmin += shift
        xmax += shift
    if xmax > x_right - 1:
        shift = xmax - (x_right - 1)
        xmin -= shift
        xmax -= shift

    # vertical position (lifted above bottom)
    y = y_bottom_px - pad - bar_offset_y

    # draw the bar
    ax.hlines(y=y, xmin=xmin, xmax=xmax, colors=color, linewidths=lw)

    # fontsize scales with width (your requested rule)
    fontsize = max(10, int((x_right - x_left) * 0.05))

    # label gap relative to FOV height (prevents crowding)
    label_gap = max(5, 0.02 * (y_bot - y_top))

    x_text = 0.5 * (xmin + xmax)
    if label_position == "above":
        ax.text(x_text, y - label_gap, label,
                fontsize=fontsize, color=color,
                ha='center', va='bottom')
    else:  # "below"
        ax.text(x_text, y + label_gap, label,
                fontsize=fontsize, color=color,
                ha='center', va='top')
        
def plot_grid(ax,
              grid_stats,
              img_height,
              img_width,
              grid_size=16,
              grid_color='tab:red',
              grid_alpha=1,
              grid_lw = 3,
              show_coordinates=False,
              grid_start=None,   # (row, col) inclusive
              grid_end=None,     # (row, col) inclusive
              crop_to_window=True,
              add_scalebar=False,
              scalebar_len_um=50,
              scalebar_zoom=3,
              scalebar_color='white'):
    """
    Draw grid boxes indicated by grid_stats['roi_id'] == (row, col).
    Optionally restrict to a window [grid_start .. grid_end] and crop the view.
    Can also add a scalebar (bottom-right).
    """

    # Validate optional window
    use_window = (grid_start is not None) and (grid_end is not None)
    if use_window:
        r0, c0 = grid_start
        r1, c1 = grid_end
        if r0 > r1: r0, r1 = r1, r0
        if c0 > c1: c0, c1 = c1, c0
        # filter rows to the requested window
        mask = grid_stats['roi_id'].apply(lambda rc: (r0 <= rc[0] <= r1) and (c0 <= rc[1] <= c1))
        grid_stats_iter = grid_stats.loc[mask]
        # pixel bounds of the window
        x_min = c0 * grid_size
        x_max = min((c1 + 1) * grid_size, img_width)
        y_min = r0 * grid_size
        y_max = min((r1 + 1) * grid_size, img_height)
    else:
        grid_stats_iter = grid_stats
        x_min, y_min = 0, 0
        x_max, y_max = img_width, img_height

    # Draw rectangles (and optional labels)
    for _, grid in grid_stats_iter.iterrows():
        r, c = grid['roi_id']
        x_start = c * grid_size
        y_start = r * grid_size
        rect = plt.Rectangle((x_start, y_start),
                             grid_size, grid_size,
                             fill=False, edgecolor=grid_color, alpha=grid_alpha, linewidth=grid_lw)
        ax.add_patch(rect)

        if show_coordinates:
            coord_text = f"{r},{c}"
            ax.text(x_start + grid_size/2, y_start + grid_size/2,
                    coord_text, color=grid_color, fontsize=6,
                    ha='center', va='center', weight='bold')

    # Crop the view if requested
    if use_window and crop_to_window:
        # imshow usually uses y increasing downward; to keep the image upright,
        # put the larger y first in ax.set_ylim
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)

    # Optional scalebar
    if add_scalebar:
        scalbar(ax,
                actual_len_um=scalebar_len_um,
                zoom=scalebar_zoom,
                x_right_px=x_max if (use_window and crop_to_window) else img_width,
                y_bottom_px=y_max if (use_window and crop_to_window) else img_height,
                pad_px=0,
                label=f"{scalebar_len_um} µm",
                color=scalebar_color,
                lw=6)
        
def plot_tiled_grid(
    ax,
    img_height,
    img_width,
    grid_size=16,
    grid_color='tab:red',
    grid_alpha=1.0,
    linewidth=3,
    show_coordinates=False,
    grid_start=None,   # (row, col) inclusive in grid units
    grid_end=None,     # (row, col) inclusive in grid units
    crop_to_window=True,
    add_scalebar=False,
    scalebar_len_um=50,
    scalebar_zoom=3,
    scalebar_color='white',
    return_indices=False
):
    """
    Tile the field of view with grid boxes of size `grid_size` pixels.
    - If `grid_start` and `grid_end` are given, only draws that window (inclusive).
    - Crops the axes view to the window if `crop_to_window=True`.
    - Optionally labels each tile with its (row, col) grid index.
    - Optionally adds a scalebar in the bottom-right.
    - Returns the list of (row, col) indices drawn if `return_indices=True`.

    Notes:
      * Edges are clipped so tiles don't extend past image borders.
      * Y axis is set to image coordinates (origin at top-left).
    """
    # How many full tiles would cover the image
    n_rows = math.ceil(img_height / grid_size)
    n_cols = math.ceil(img_width  / grid_size)

    # Resolve window
    use_window = (grid_start is not None) and (grid_end is not None)
    if use_window:
        r0, c0 = grid_start
        r1, c1 = grid_end
        # normalize ordering
        if r0 > r1: r0, r1 = r1, r0
        if c0 > c1: c0, c1 = c1, c0
        # clamp to valid grid indices
        r0 = max(0, min(r0, n_rows-1))
        r1 = max(0, min(r1, n_rows-1))
        c0 = max(0, min(c0, n_cols-1))
        c1 = max(0, min(c1, n_cols-1))

        # pixel bounds of the window (clip to image size)
        x_min = c0 * grid_size
        x_max = min((c1 + 1) * grid_size, img_width)
        y_min = r0 * grid_size
        y_max = min((r1 + 1) * grid_size, img_height)
    else:
        r0, c0 = 0, 0
        r1, c1 = n_rows - 1, n_cols - 1
        x_min, y_min = 0, 0
        x_max, y_max = img_width, img_height

    drawn_indices = []

    # Draw tiles
    for r in range(r0, r1 + 1):
        y_start = r * grid_size
        if y_start >= img_height:
            continue
        tile_h = min(grid_size, img_height - y_start)
        if tile_h <= 0:
            continue

        for c in range(c0, c1 + 1):
            x_start = c * grid_size
            if x_start >= img_width:
                continue
            tile_w = min(grid_size, img_width - x_start)
            if tile_w <= 0:
                continue

            rect = plt.Rectangle(
                (x_start, y_start),
                tile_w, tile_h,
                fill=False,
                edgecolor=grid_color,
                alpha=grid_alpha,
                linewidth=linewidth
            )
            ax.add_patch(rect)

            if show_coordinates:
                ax.text(
                    x_start + tile_w/2,
                    y_start + tile_h/2,
                    f"{r},{c}",
                    color=grid_color,
                    fontsize=6,
                    ha='center', va='center', weight='bold'
                )

            drawn_indices.append((r, c))

    # Set view (image-style coords: y increases downward)
    if use_window and crop_to_window:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)

    # Optional scalebar (uses your existing `scalbar` helper)
    if add_scalebar:
        scalbar(
            ax,
            actual_len_um=scalebar_len_um,
            zoom=scalebar_zoom,
            x_right_px=x_max if (use_window and crop_to_window) else img_width,
            y_bottom_px=y_max if (use_window and crop_to_window) else img_height,
            pad_px=0,
            label=f"{scalebar_len_um} µm",
            color=scalebar_color,
            lw=linewidth
        )

    if return_indices:
        return drawn_indices
    
def show_target_grid(target_grid, grid_size, ref_img, fiber_mask=None,
                         q_min=1, q_max=99,
                         margin=20, lw=1.5):
    # Calculate crop boundaries around target grid
    crop_margin = grid_size + margin  # grid size plus 20 pixels margin
    x_center = target_grid[1] * grid_size + grid_size // 2
    y_center = target_grid[0] * grid_size + grid_size // 2

    x_min = max(0, x_center - crop_margin)
    x_max = min(ref_img.shape[1], x_center + crop_margin)
    y_min = max(0, y_center - crop_margin)
    y_max = min(ref_img.shape[0], y_center + crop_margin)

    # Crop the reference image
    ref_img_cropped = ref_img[y_min:y_max, x_min:x_max]
    
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    # Left subplot: Reference image overlay
    ax.imshow(ref_img_cropped, 
              vmin = np.percentile(ref_img, q_min),
              vmax = np.percentile(ref_img, q_max),
              cmap='gray', alpha=1)
    
    if fiber_mask is not None:
        fiber_mask = fiber_mask[y_min:y_max, x_min:x_max]
        ax.imshow(fiber_mask, 
                  cmap='Set1', alpha=0.4)
    
    # Overlay significant grids on left subplot
    img_height, img_width = ref_img_cropped.shape

    x_start = target_grid[1] * grid_size - x_min
    y_start = target_grid[0] * grid_size - y_min
    rect = plt.Rectangle((x_start, y_start), grid_size, grid_size, 
                        fill=False, edgecolor='red', linewidth=lw)
    ax.add_patch(rect)

    # ax.set_title(f'Reference Image with Significant Grids (n={len(sig_grid)})')
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.axis('off')

    # plt.show()
    return fig


def plot_grid_with_mask(
    ax,
    grid_stats,
    img_height,
    img_width,
    grid_size=16,
    show_grid = True,
    grid_color='tab:red',
    grid_alpha=1,
    grid_lw=3,
    show_coordinates=False,
    grid_start=None,   # (row, col) inclusive
    grid_end=None,     # (row, col) inclusive
    crop_to_window=True,
    add_scalebar=False,
    scalebar_len_um=50,
    scalebar_zoom=3,
    scalebar_color='white',
    # --- mask overlay controls (imshow) ---
    axon_mask=None,           # (H, W) bool
    mask_cmap='Set1',
    mask_alpha=0.5,
    mask_interp='none'
):
    """
    grid_stats selects WHICH tiles to draw (and overlay).
    grid_start/grid_end only bound the view (and optionally crop it).
    The mask overlay is shown only inside (grid_stats ∩ window).
    """

    # ---- resolve window bounds for view/cropping
    use_window = (grid_start is not None) and (grid_end is not None)
    if use_window:
        r0, c0 = grid_start
        r1, c1 = grid_end
        if r0 > r1: r0, r1 = r1, r0
        if c0 > c1: c0, c1 = c1, c0
        x_min = c0 * grid_size
        x_max = min((c1 + 1) * grid_size, img_width)
        y_min = r0 * grid_size
        y_max = min((r1 + 1) * grid_size, img_height)
    else:
        r0, c0 = 0, 0
        r1 = (img_height - 1) // grid_size
        c1 = (img_width  - 1) // grid_size
        x_min, y_min = 0, 0
        x_max, y_max = img_width, img_height

    # ---- intersect grid_stats with window to get tiles to actually draw
    def _in_window(rc):
        r, c = rc
        return (r0 <= r <= r1) and (c0 <= c <= c1)

    grid_stats_iter = grid_stats.loc[grid_stats['roi_id'].apply(_in_window)]

    # ---- build a selection mask ONLY from the selected tiles (grid_stats_iter)
    if axon_mask is not None:
        if axon_mask.shape != (img_height, img_width):
            raise ValueError(
                f"axon_mask shape {axon_mask.shape} != ({img_height},{img_width})"
            )

        tile_sel = np.zeros((img_height, img_width), dtype=bool)
        for _, g in grid_stats_iter.iterrows():
            r, c = g['roi_id']
            y0, y1 = r * grid_size, min((r + 1) * grid_size, img_height)
            x0, x1 = c * grid_size, min((c + 1) * grid_size, img_width)
            if y0 < img_height and x0 < img_width:
                tile_sel[y0:y1, x0:x1] = True

        masked_in_tiles = (axon_mask.astype(bool) & tile_sel)

        # # masked array so non-selected pixels are transparent
        # imshow_mask = np.ma.masked_where(~masked_in_tiles, masked_in_tiles.astype(float))

        # # make NaNs/masked transparent
        # cmap = plt.get_cmap(mask_cmap).copy()
        # cmap.set_bad(alpha=0)
        
        imshow_mask = np.where(masked_in_tiles>0, 1, np.nan)
        # draw overlay BEFORE rectangles so lines stay on top
        ax.imshow(
            imshow_mask,
            cmap=mask_cmap,
            alpha=mask_alpha,
            interpolation=mask_interp,
            # vmin=0, vmax=1,
            origin='upper',   # match image coordinates
            zorder=2
        )

    # ---- draw rectangles & optional labels (only for selected tiles)
    if show_grid:
        for _, g in grid_stats_iter.iterrows():
            r, c = g['roi_id']
            x0, y0 = c * grid_size, r * grid_size
            rect = plt.Rectangle(
                (x0, y0),
                grid_size, grid_size,
                fill=False, edgecolor=grid_color, alpha=grid_alpha, linewidth=grid_lw
            )
            ax.add_patch(rect)
            if show_coordinates:
                ax.text(
                    x0 + grid_size/2, y0 + grid_size/2,
                    f"{r},{c}",
                    color=grid_color, fontsize=6, ha='center', va='center', weight='bold'
                )

    # ---- set view
    if use_window and crop_to_window:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)

    # ---- optional scalebar
    if add_scalebar:
        scalbar(
            ax,
            actual_len_um=scalebar_len_um,
            zoom=scalebar_zoom,
            x_right_px=x_max if (use_window and crop_to_window) else img_width,
            y_bottom_px=y_max if (use_window and crop_to_window) else img_height,
            pad_px=0,
            label=f"{scalebar_len_um} µm",
            color=scalebar_color,
            lw=6
        )

def plot_random_dff_traces(F_soma, n_show=30, seed=None, q_baseline=10, eps=1e-6,
                           use_common_ylim=False, figsize=(12, 14), dpi=200,
                           title='',
                           out_path_fig=None):
    """
    F_soma: (n_rois, n_frames) fluorescence array (numpy).
    n_show: number of ROIs to plot
    q_baseline: percentile baseline per ROI for dF/F
    use_common_ylim: if True, use a common y-lim across panels (robust)
    """
    F = np.asarray(F_soma)
    n_rois, n_frames = F.shape
    n_show = min(n_show, n_rois)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rois, size=n_show, replace=False)
    
    dff=F_soma


    # robust scaling helpers
    def robust_ylim(y):
        lo, hi = np.nanpercentile(y, [1, 99])
        pad = 0.10 * (hi - lo + 1e-12)
        return lo - pad, hi + pad

    if use_common_ylim:
        ylo, yhi = robust_ylim(dff.reshape(-1))
    else:
        ylo = yhi = None

    fig, axes = plt.subplots(
        n_show, 1, sharex=True,
        figsize=figsize, dpi=dpi,
        gridspec_kw={"hspace": 0.05}
    )
    if n_show == 1:
        axes = [axes]

    x = np.arange(n_frames)

    for i, ax in enumerate(axes):
        y = dff[idx[i]]
        ax.plot(x, y, lw=0.8)
        # ax.set_ylabel(f"ROI {idx[i]}", rotation=0, labelpad=25, va="center")

        if use_common_ylim:
            ax.set_ylim(ylo, yhi)
        else:
            lo, hi = robust_ylim(y)
            ax.set_ylim(lo, hi)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # keep y ticks minimal to reduce clutter
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(axis="y", labelsize=7)

    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"{title}")

    plt.tight_layout()
    if out_path_fig is None:
        return fig, axes, idx
    else:
        plt.savefig(out_path_fig, dpi=dpi)
        plt.close()

#%%
def plot_pyr_sorted_heatmap(df_rec, rec_id, bef, aft, s, 
                            prefix='', suffix='', show_pyr_lines=True, 
                            sorted_idx=None,
                            activity_profile = 'mean_profile',
                            ratio = 'response_ratio',
                            cross_sess_norm=False,
                            norm = True, sm=True, plot_mean=False):
    if len(suffix)>0:
        suffix = f'_{suffix}'
    if len(prefix)>0:
        prefix = f'{prefix}_'
            
    if plot_mean:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    else:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
    activity_profile = f'{activity_profile}'
    
    # Sort data by run_ratio
    if sorted_idx is not None:
       # session_for_sorting = s
        df_rec['roi_id'] = pd.Categorical(
            df_rec['roi_id'],
            categories=sorted_idx,
            ordered=True
        )

        df_sorted = df_rec.sort_values('roi_id').reset_index(drop=True)       
    else:   
        df_sorted = df_rec.sort_values(by=f'{ratio}', ascending=False)
    # df_sorted = df_rec.dropna(subset=[f'{ratio}_{session_for_sorting}']).sort_values(by=f'{ratio}_{session_for_sorting}')
    run_aligned_sorted = np.stack(df_sorted[activity_profile]).squeeze()
    if sm:
        run_aligned_sorted_sm = gaussian_filter1d(run_aligned_sorted, sigma=1, axis=-1)
    else:
        run_aligned_sorted_sm = run_aligned_sorted
    if norm:
        run_aligned_sorted_sm = utl.normalize(run_aligned_sorted_sm)
    else:
        run_aligned_sorted_sm = run_aligned_sorted_sm
    # run_aligned_sorted_sm = robust_std_filter(run_aligned_sorted_sm, robust_sd_factor)
    tot_roi = run_aligned_sorted.shape[0]
    
    # Plot heatmap
    ax.imshow(run_aligned_sorted_sm, 
              cmap='Greys', 
              aspect='auto', 
              extent=[-bef, aft, 0, tot_roi],
              interpolation=None)
    
    # Add lines to indicate pyrUp and pyrDown boundaries
    if show_pyr_lines:
        # Get the sorted indices for pyrUp and pyrDown
        pyr_up_col = f'pyrUp'
        pyr_down_col = f'pyrDown'
        
        if pyr_up_col in df_sorted.columns and pyr_down_col in df_sorted.columns:
            # Count pyrUp cells (they should be at the top since sorted by ratio descending)
            n_pyr_up = df_sorted[pyr_up_col].sum()
            # Count pyrDown cells (they should be at the bottom)
            n_pyr_down = df_sorted[pyr_down_col].sum()
            
            # Draw horizontal line at the boundary of pyrUp cells
            if n_pyr_up > 0:
                ax.axhline(y=tot_roi-n_pyr_up, color='purple', linestyle='-', linewidth=1.5, 
                          alpha=0.7, label=f'pyrUp boundary (n={n_pyr_up})')
                print(n_pyr_up)
            # Draw horizontal line at the boundary of pyrDown cells
            if n_pyr_down > 0:
                ax.axhline(y=n_pyr_down, color='darkorange', linestyle='-', 
                          linewidth=1.5, alpha=0.7, label=f'pyrDown boundary (n={n_pyr_down})')
                print(n_pyr_down)
                
            # Add legend if lines were drawn
            # if n_pyr_up > 0 or n_pyr_down > 0:
            #     ax.legend(loc='upper right', fontsize=6, frameon=False)
    
    # if plot_mean:
    #     # pyr_up_col = f'pyrUp_{s}'
    #     # pyr_down_col = f'pyrDown_{s}'
        
    #     pyr_up_col = 'pyrUp_ss1'
    #     pyr_down_col = 'pyrDown_ss1'
    #     pyr_up_prof = np.stack(df_sorted.loc[df_sorted[pyr_up_col], 'run_cal_profile_raw_good_ss1'])
    #     pyr_down_prof = np.stack(df_sorted.loc[df_sorted[pyr_down_col], 'run_cal_profile_raw_good_ss1'])
        
    #     tax = ax.twinx()
    #     xaxis = np.arange((bef+aft)*30)/30-bef
    #     pf.plot_mean_trace(pyr_up_prof, tax, xaxis, color='purple', label='pyrUp_ss1')
    #     pf.plot_mean_trace(pyr_down_prof, tax, xaxis, color='darkorange', label='pyrDown_ss2')
        
    #     pyr_up_col = 'pyrUp_ss2'
    #     pyr_down_col = 'pyrDown_ss2'
    #     pyr_up_prof = np.stack(df_sorted.loc[df_sorted[pyr_up_col], 'run_cal_profile_raw_good_ss2'])
    #     pyr_down_prof = np.stack(df_sorted.loc[df_sorted[pyr_down_col], 'run_cal_profile_raw_good_ss2'])
    #     pf.plot_mean_trace(pyr_up_prof, tax, xaxis, color='indigo', label='pyrUp_ss2')
    #     pf.plot_mean_trace(pyr_down_prof, tax, xaxis, color='chocolate', label='pyrDown_ss2')
        
    #     tax.legend(frameon=False, loc=(1.2, .8), prop={'size': 6})
    #     tax.set(ylabel='raw dFF')
    
    # ax.axvline(0, 0, lw=1, ls='--', color='darkred')
    
    # ax.set(xlabel='time (s)', 
    #        ylabel='roi sorted by {}'.format(session_for_sorting), 
    #        xlim=(-1, 4),
    #        title='{}{}{}'.format(prefix,
    #                              rec_id,
    #                              suffix)
    #        )
    fig.tight_layout()
    return fig, ax

    



    
