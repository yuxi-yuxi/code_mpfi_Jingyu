"""PySide6 desktop application for comparing processed GCaMP sessions."""
from __future__ import annotations

import os
import sys
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("QT_FONT_DPI", "96")

from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget, QScrollArea, QSizePolicy, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.patches import Rectangle
import numpy as np

from .analysis import (
    DEFAULT_DATA_ROOT, GroupedMaps, discover_sessions, group_session,
    load_session, load_session_fov, normalized_sorted_maps,
    session_treatment_label, trial_type_metrics,
)
from .export import export_html
from .property_statistics import (
    available_trial_types, build_property_comparison,
)


class WorkerSignals(QObject):
    completed = Signal(object, object)


class LoadWorker(QRunnable):
    def __init__(self, root: str, sessions: list[str], options: dict):
        super().__init__()
        self.root, self.sessions, self.options = root, sessions, options

    def run(self):
        groups, errors = [], []
        for rec_id in self.sessions:
            try:
                session = load_session(rec_id, self.root)
                try:
                    load_session_fov(session)
                except Exception as fov_exc:
                    errors.append(f"{rec_id} FOV: {fov_exc}")
                group = group_session(session, **self.options)
                group.trial_metrics = trial_type_metrics(group)
                groups.append(group)
            except Exception as exc:
                errors.append(f"{rec_id}: {exc}")
        self.signals.completed.emit(groups, errors)

    @property
    def signals(self):
        if not hasattr(self, "_signals"):
            self._signals = WorkerSignals()
        return self._signals


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(facecolor="#1b1e23", constrained_layout=True)
        super().__init__(self.figure)
        self.setObjectName("plotCanvas")
        self.setStyleSheet("background: #1b1e23; border: 0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.resize_for_rows(1)

    def resize_for_rows(self, rows: int):
        """Give every session a readable row; the enclosing view scrolls."""
        width_px = 1180
        height_px = max(430, 330 * max(1, rows))
        self.setFixedSize(width_px, height_px)
        self.figure.set_size_inches(
            width_px / self.figure.dpi, height_px / self.figure.dpi, forward=True
        )

    def show_groups(self, groups: list[GroupedMaps]):
        self.figure.clear()
        if not groups:
            ax = self.figure.subplots()
            ax.text(.5, .5, "Select one or more sessions and click Load",
                    ha="center", va="center", color="#8a95aa", transform=ax.transAxes)
            ax.set_axis_off(); self.draw_idle(); return
        single_condition = all(not group.label_b for group in groups)
        ncols = 2 if single_condition else 3
        axes = self.figure.subplots(len(groups), ncols, squeeze=False)
        for row, group in enumerate(groups):
            maps_a, maps_b = normalized_sorted_maps(group)
            heatmaps = [(axes[row, 0], maps_a, group.label_a, len(group.lap_idx_a))]
            if not single_condition:
                heatmaps.append((axes[row, 1], maps_b, group.label_b, len(group.lap_idx_b)))
            for ax, values, label, lap_count in heatmaps:
                ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                          extent=[0, 180, len(values), 0], interpolation="none")
                ax.set_title(f"{group.session.display_name}\n{label} (laps={lap_count})",
                             fontsize=8, pad=4)
                ax.set(xlabel="Position (cm)", ylabel="Cells")
            ax = axes[row, -1]
            if len(group.lap_idx_a):
                ax.plot(group.session.bin_centres, np.nanmean(group.maps_a, axis=0),
                        color="#00c2ff", label=group.label_a)
            if group.label_b and len(group.lap_idx_b):
                ax.plot(group.session.bin_centres, np.nanmean(group.maps_b, axis=0),
                        color="#ff5f87", label=group.label_b)
            ax.set_title(f"{group.session.display_name}\nPopulation mean | n={len(group.cell_indices)} cells",
                         fontsize=8, pad=4)
            ax.set(xlabel="Position (cm)", ylabel="Mean dF/F")
            ax.legend(fontsize=7); ax.grid(alpha=.15)
        for ax in axes.flat:
            ax.set_facecolor("#272c36"); ax.tick_params(colors="#d8dee9", labelsize=7)
            ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
            ax.xaxis.label.set_color("#d8dee9"); ax.yaxis.label.set_color("#d8dee9")
            ax.title.set_color("white")
            for spine in ax.spines.values(): spine.set_color("#4b5363")
        self.resize_for_rows(len(groups))
        self.draw_idle()
class CombinedHeatmapCanvas(FigureCanvasQTAgg):
    """Pool all selected cells and globally sort by place-field location."""

    def __init__(self):
        self.figure = Figure(facecolor="#1b1e23", constrained_layout=True)
        super().__init__(self.figure)
        self.setObjectName("plotCanvas")
        self.setStyleSheet("background: #1b1e23; border: 0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.total_cells = 0
        self.sorted_field_positions = np.array([], dtype=float)
        self.sorted_session_ids = np.array([], dtype=object)
        self.sorted_cell_ids = np.array([], dtype=int)
        self._resize(1)

    def _resize(self, n_sessions: int):
        width_px = 1180
        height_px = max(650, 180 * max(1, n_sessions))
        self.setFixedSize(width_px, height_px)
        self.figure.set_size_inches(
            width_px / self.figure.dpi, height_px / self.figure.dpi, forward=True
        )

    @staticmethod
    def _normalize(group: GroupedMaps):
        maps_a = np.asarray(group.maps_a, dtype=float)
        maps_b = np.asarray(group.maps_b, dtype=float)
        both = np.concatenate([maps_a, maps_b], axis=1)
        finite_both = np.isfinite(both)
        row_max = np.max(np.where(finite_both, both, -np.inf), axis=1)
        valid_max = np.isfinite(row_max) & (row_max != 0)
        norm_a = np.divide(
            maps_a, row_max[:, None], out=np.full_like(maps_a, np.nan),
            where=valid_max[:, None] & np.isfinite(maps_a),
        )
        norm_b = np.divide(
            maps_b, row_max[:, None], out=np.full_like(maps_b, np.nan),
            where=valid_max[:, None] & np.isfinite(maps_b),
        )
        count = np.isfinite(maps_a).astype(int) + np.isfinite(maps_b).astype(int)
        reference = np.divide(
            np.nan_to_num(maps_a, nan=0.0) + np.nan_to_num(maps_b, nan=0.0),
            count, out=np.full_like(maps_a, np.nan), where=count > 0,
        )
        peak_idx = np.argmax(
            np.where(np.isfinite(reference), reference, -np.inf), axis=1
        )
        fallback_positions = group.session.bin_centres[peak_idx]
        return norm_a, norm_b, fallback_positions

    def show_groups(self, groups: list[GroupedMaps]):
        self.figure.clear()
        self._resize(len(groups))
        if not groups:
            ax = self.figure.subplots()
            ax.text(.5, .5, "Load multiple sessions to build pooled heatmaps",
                    ha="center", va="center", color="#8a95aa",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.total_cells = 0
            self.sorted_field_positions = np.array([], dtype=float)
            self.sorted_session_ids = np.array([], dtype=object)
            self.sorted_cell_ids = np.array([], dtype=int)
            self.draw_idle()
            return

        pooled_a, pooled_b = [], []
        field_positions, session_ids, cell_ids = [], [], []
        for group in groups:
            if not len(group.cell_indices):
                continue
            norm_a, norm_b, fallback = self._normalize(group)
            rows = group.session.dataframe.iloc[group.cell_indices]
            positions = np.asarray(
                np.array(rows["place_field_position_cm"], dtype=float), dtype=float
            )
            positions = np.where(np.isfinite(positions), positions, fallback)
            ids = np.asarray(rows["cell_id"], dtype=int)
            pooled_a.append(norm_a); pooled_b.append(norm_b)
            field_positions.append(positions)
            cell_ids.append(ids)
            session_ids.append(np.full(len(ids), group.session.rec_id, dtype=object))
        if not pooled_a:
            self.show_groups([])
            return

        pooled_a = np.vstack(pooled_a)
        pooled_b = np.vstack(pooled_b)
        positions = np.concatenate(field_positions)
        sessions = np.concatenate(session_ids)
        cells = np.concatenate(cell_ids)
        global_order = np.argsort(positions, kind="stable")
        pooled_a = pooled_a[global_order]
        pooled_b = pooled_b[global_order]
        self.sorted_field_positions = positions[global_order]
        self.sorted_session_ids = sessions[global_order]
        self.sorted_cell_ids = cells[global_order]
        self.total_cells = len(global_order)

        entries = [(pooled_a, groups[0].label_a)]
        if groups[0].label_b:
            entries.append((pooled_b, groups[0].label_b))
        axes = self.figure.subplots(1, len(entries), squeeze=False, sharey=True)[0]
        cmap = colormaps["viridis"].copy()
        cmap.set_bad("#1b1e23")
        image = None
        for ax, (values, title) in zip(axes, entries):
            image = ax.imshow(
                np.ma.masked_invalid(values), aspect="auto", cmap=cmap,
                vmin=0, vmax=1, extent=[0, 180, self.total_cells, 0],
                interpolation="none",
            )
            ax.axvline(150, color="gold", linewidth=.7, linestyle="--", alpha=.7)
            ax.set_title(
                f"{title} | {self.total_cells} pooled cells from {len(groups)} sessions",
                fontsize=9, color="white", pad=6,
            )
            ax.set_xlabel("Position (cm)", fontsize=8, color="#d8dee9")
            ax.tick_params(colors="#d8dee9", labelsize=7)
            ax.set_facecolor("#272c36")
            for spine in ax.spines.values():
                spine.set_color("#4b5363")
        axes[0].set_ylabel(
            "All pooled cells (globally sorted by field location)",
            fontsize=8, color="#d8dee9",
        )
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        colorbar = self.figure.colorbar(image, ax=axes.tolist(), shrink=.55, pad=.02)
        colorbar.set_label("Cell-normalized activity", color="#d8dee9", fontsize=8)
        colorbar.ax.tick_params(colors="#d8dee9", labelsize=7)
        self.draw_idle()
class MetricsCanvas(FigureCanvasQTAgg):
    """Quantitative summaries for the threshold-defined place-cell population."""

    def __init__(self):
        self.figure = Figure(facecolor="#1b1e23", constrained_layout=True)
        super().__init__(self.figure)
        self.setObjectName("plotCanvas")
        self.setStyleSheet("background: #1b1e23; border: 0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.metric_data = []
        self._resize(1)

    def _resize(self, n_sessions: int):
        width_px = max(1180, 115 * max(1, n_sessions))
        height_px = 780
        self.setFixedSize(width_px, height_px)
        self.figure.set_size_inches(
            width_px / self.figure.dpi, height_px / self.figure.dpi, forward=True
        )

    @staticmethod
    def _boxplot(ax, metrics: list[dict], key: str, title: str, ylabel: str):
        entries = [(m["rec_id"], np.asarray(m[key], dtype=float)) for m in metrics]
        entries = [(label, values[np.isfinite(values)])
                   for label, values in entries if np.any(np.isfinite(values))]
        if not entries:
            ax.text(.5, .5, "No finite values", ha="center", va="center",
                    color="#8a95aa", transform=ax.transAxes)
            ax.set_title(title)
            return
        labels, values = zip(*entries)
        result = ax.boxplot(values, labels=labels, patch_artist=True,
                            showfliers=False, medianprops={"color": "white"})
        for patch in result["boxes"]:
            patch.set_facecolor("#168aad"); patch.set_alpha(.75)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=45)

    def show_groups(self, groups: list[GroupedMaps]):
        self.figure.clear()
        self.metric_data = [
            metric for group in groups for metric in trial_type_metrics(group)
        ]
        self._resize(len(self.metric_data))
        if not groups:
            ax = self.figure.subplots()
            ax.text(.5, .5, "Load sessions to quantify place-cell properties",
                    ha="center", va="center", color="#8a95aa",
                    transform=ax.transAxes)
            ax.set_axis_off(); self.draw_idle(); return

        metrics = self.metric_data
        labels = [f'{m["rec_id"]} [{m["treatment"]}]\n{m["trial_type"]} (laps={m["lap_count"]})'
                  for m in metrics]
        x = np.arange(len(metrics))
        axes = self.figure.subplots(2, 4, squeeze=False)

        ax = axes[0, 0]
        ax.bar(x, [m["place_cell_percent"] for m in metrics], color="#00c2ff")
        ax.set_title("Shared place-cell cohort")
        ax.set_ylabel("Place cells (%)")
        ax.set_xticks(x, labels, rotation=45, ha="right")

        self._boxplot(axes[0, 1], metrics, "spatial_information",
                      "Spatial information", "Bits/event")

        ax = axes[0, 2]
        bins = np.arange(0, 180.01, 15.0)
        centres = (bins[:-1] + bins[1:]) / 2
        pooled_locations = np.concatenate(
            [m["field_locations"] for m in metrics if len(m["field_locations"])]
        ) if any(len(m["field_locations"]) for m in metrics) else np.array([])
        if len(pooled_locations):
            counts, _ = np.histogram(pooled_locations, bins=bins)
            ax.bar(centres, 100 * counts / counts.sum(), width=14,
                   color="#6c63ff", alpha=.75, label="Pooled")
        if len(metrics) <= 6:
            for m in metrics:
                if not len(m["field_locations"]):
                    continue
                counts, _ = np.histogram(m["field_locations"], bins=bins)
                ax.step(centres, 100 * counts / counts.sum(), where="mid",
                        linewidth=1,
                        label=f'{m["rec_id"]} | {m["trial_type"]}')
            ax.legend(fontsize=5)
        ax.axvspan(0, 30, color="#00c2ff", alpha=.08)
        ax.axvspan(150, 180, color="gold", alpha=.10)
        ax.set(title="Place-field location", xlabel="Position (cm)",
               ylabel="Fields per bin (%)", xlim=(0, 180))

        ax = axes[0, 3]
        width = .36
        reward = [100 * m["reward_fraction"] for m in metrics]
        onset = [100 * m["onset_fraction"] for m in metrics]
        ax.bar(x - width / 2, reward, width, label="Reward 150-180 cm",
               color="gold")
        ax.bar(x + width / 2, onset, width, label="Onset 0-30 cm",
               color="#00c2ff")
        expected = 100 * metrics[0]["expected_zone_fraction"]
        ax.axhline(expected, color="white", linestyle="--", linewidth=.8,
                   label=f"Uniform expectation ({expected:.1f}%)")
        ax.set_title("Field over-representation")
        ax.set_ylabel("Fields in zone (%)")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.legend(fontsize=6)

        self._boxplot(axes[1, 0], metrics, "field_width",
                      "Place-field width", "Width (cm)")
        self._boxplot(axes[1, 1], metrics, "odd_even_stability",
                      "Odd-even stability", "Pearson r")
        self._boxplot(axes[1, 2], metrics, "consecutive_stability",
                      "Consecutive-lap stability", "Pearson r")
        self._boxplot(axes[1, 3], metrics, "in_out_ratio",
                      "In-field / out-of-field ratio", "Ratio")

        for ax in axes.flat:
            ax.set_facecolor("#272c36")
            ax.tick_params(colors="#d8dee9", labelsize=6)
            ax.xaxis.label.set_color("#d8dee9")
            ax.yaxis.label.set_color("#d8dee9")
            ax.title.set_color("white"); ax.title.set_fontsize(8)
            ax.grid(axis="y", alpha=.12)
            for spine in ax.spines.values():
                spine.set_color("#4b5363")
        self.draw_idle()

class PropertyStatisticsCanvas(FigureCanvasQTAgg):
    """Nine place-cell property comparisons using Jingyu plotting helpers."""

    def __init__(self):
        self.figure = Figure(facecolor="#1b1e23", constrained_layout=True)
        super().__init__(self.figure)
        self.setObjectName("plotCanvas")
        self.setStyleSheet("background: #1b1e23; border: 0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(1180, 920)
        self.figure.set_size_inches(11.8, 9.2, forward=True)
        self.last_comparison = None

    @staticmethod
    def _dark_style(ax):
        ax.set_facecolor("#272c36")
        ax.tick_params(colors="#d8dee9", labelsize=6)
        ax.xaxis.label.set_color("#d8dee9")
        ax.yaxis.label.set_color("#d8dee9")
        ax.title.set_color("white"); ax.title.set_fontsize(7)
        for text_item in ax.texts:
            cleaned = (text_item.get_text().replace("Â±", "+/-")
                       .replace("±", "+/-"))
            text_item.set_text(cleaned); text_item.set_color("#f5f6f7")
        for line in ax.lines:
            color = str(line.get_color()).lower()
            if color in {"k", "black", "#000000"}:
                line.set_color("#aeb8ca")
        for spine in ax.spines.values():
            spine.set_color("#4b5363")
        ax.grid(axis="y", alpha=.10)

    def show_comparison(self, comparison: dict | None):
        self.figure.clear()
        self.last_comparison = comparison
        if comparison is None:
            ax = self.figure.subplots()
            ax.text(.5, .5, "Load sessions to compare place-cell properties",
                    ha="center", va="center", color="#8a95aa",
                    transform=ax.transAxes)
            ax.set_axis_off(); self.draw_idle(); return

        from common import plotting_functions_Jingyu as pf
        axes = self.figure.subplots(3, 3, squeeze=False)
        labels = comparison["labels"]
        colors = ("#8a95aa", "#ff5f87")
        for ax, metric in zip(axes.flat, comparison["metrics"]):
            values_a, values_b = metric["values_a"], metric["values_b"]
            if not len(values_a) or not len(values_b):
                ax.text(.5, .5, "Not enough selected sessions",
                        ha="center", va="center", color="#8a95aa",
                        transform=ax.transAxes)
                ax.set_title(metric["title"])
                ax.set_ylabel(metric["ylabel"])
                self._dark_style(ax)
                continue
            ylim = ((-1.0, 1.35) if metric["key"] in
                    {"odd_even_stability", "consecutive_stability"} else None)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    if comparison["paired"]:
                        pf.plot_bar_with_paired_scatter(
                            ax, values_a, values_b, colors=colors,
                            title=metric["title"], ylabel=metric["ylabel"],
                            xticklabels=labels, ylim=ylim,
                        )
                    else:
                        pf.plot_bar_with_unpaired_scatter(
                            ax, values_a, values_b, colors=colors,
                            title=metric["title"], ylabel=metric["ylabel"],
                            xticklabels=labels, ylim=ylim,
                        )
            except (AssertionError, ValueError) as exc:
                ax.clear()
                ax.text(.5, .5, str(exc), ha="center", va="center",
                        color="#8a95aa", transform=ax.transAxes, wrap=True)
                ax.set_title(metric["title"]); ax.set_ylabel(metric["ylabel"])
            self._dark_style(ax)
        self.draw_idle()

class SequenceCanvas(FigureCanvasQTAgg):
    """Clickable place-cell sequence sorted by each cell's field location."""
    cellSelected = Signal(int)

    def __init__(self):
        self.figure = Figure(facecolor="#1b1e23", constrained_layout=True)
        super().__init__(self.figure)
        self.setObjectName("plotCanvas")
        self.setStyleSheet("background: #1b1e23; border: 0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(430, 520)
        self.figure.set_size_inches(4.3, 5.2, forward=True)
        self._axis = None
        self._sorted_cell_ids = np.array([], dtype=int)
        self._row_by_cell = {}
        self._selection_patch = None
        self.sorted_field_positions = np.array([], dtype=float)
        self.sequence_matrix = np.empty((0, 0), dtype=float)
        self.mpl_connect("button_press_event", self._on_click)

    def set_group(self, group: GroupedMaps | None):
        self.figure.clear()
        self._axis = self.figure.subplots()
        self._selection_patch = None
        self._sorted_cell_ids = np.array([], dtype=int)
        self._row_by_cell = {}
        self.sorted_field_positions = np.array([], dtype=float)
        self.sequence_matrix = np.empty((0, 0), dtype=float)
        sequence_indices = (None if group is None else
                            group.place_cell_indices if group.place_cell_indices is not None
                            else group.cell_indices)
        if group is None or not len(sequence_indices):
            self._axis.text(
                .5, .5, "No place cells at the current thresholds",
                ha="center", va="center", color="#8a95aa",
                transform=self._axis.transAxes,
            )
            self._axis.set_axis_off()
            self.draw_idle()
            return

        profiles, positions, cell_ids = [], [], []
        n_bins = group.session.tensor.shape[2]
        for row_idx in np.asarray(sequence_indices, dtype=int):
            row = group.session.dataframe.iloc[int(row_idx)]
            profile = np.asarray(row.get("place_field_map_norm", []), dtype=float).ravel()
            if profile.size != n_bins or not np.any(np.isfinite(profile)):
                profile = np.nanmean(group.session.tensor[int(row_idx)], axis=0)
            finite = profile[np.isfinite(profile)]
            if finite.size:
                lo, hi = float(np.min(finite)), float(np.max(finite))
                profile = ((profile - lo) / (hi - lo) if hi > lo
                           else np.zeros_like(profile, dtype=float))
            else:
                profile = np.full(n_bins, np.nan, dtype=float)
            peak_idx = int(np.argmax(np.where(np.isfinite(profile), profile, -np.inf)))
            fallback = float(group.session.bin_centres[peak_idx])
            position = float(row.get("place_field_position_cm", np.nan))
            profiles.append(profile)
            positions.append(position if np.isfinite(position) else fallback)
            cell_ids.append(int(row.get("cell_id", row_idx)))

        positions = np.asarray(positions, dtype=float)
        cell_ids = np.asarray(cell_ids, dtype=int)
        order = np.argsort(positions, kind="stable")
        self.sequence_matrix = np.vstack(profiles)[order]
        self.sorted_field_positions = positions[order]
        self._sorted_cell_ids = cell_ids[order]
        self._row_by_cell = {
            int(cell_id): row for row, cell_id in enumerate(self._sorted_cell_ids)
        }
        n_cells = len(self._sorted_cell_ids)
        cmap = colormaps["viridis"].copy()
        cmap.set_bad("#1b1e23")
        self._axis.imshow(
            np.ma.masked_invalid(self.sequence_matrix), aspect="auto", cmap=cmap,
            vmin=0, vmax=1, extent=[0, 180, n_cells, 0], interpolation="none",
        )
        self._axis.axvline(150, color="gold", linewidth=.7,
                          linestyle="--", alpha=.7)
        self._axis.set_title(
            f"{group.session.display_name}\nPlace-cell sequence - click a row",
            fontsize=8, color="white", pad=5,
        )
        self._axis.set_xlabel("Position (cm)", fontsize=8, color="#d8dee9")
        self._axis.set_ylabel(
            f"{n_cells} cells sorted by field location",
            fontsize=8, color="#d8dee9",
        )
        self._axis.tick_params(colors="#d8dee9", labelsize=7)
        self._axis.set_facecolor("#272c36")
        for spine in self._axis.spines.values():
            spine.set_color("#4b5363")
        self.draw_idle()

    def select_cell(self, cell_id: int | None):
        if self._selection_patch is not None:
            self._selection_patch.remove()
            self._selection_patch = None
        if cell_id is None or self._axis is None or cell_id not in self._row_by_cell:
            self.draw_idle()
            return
        row = self._row_by_cell[int(cell_id)]
        self._selection_patch = Rectangle(
            (0, row), 180, 1, linewidth=1.8, edgecolor="cyan",
            facecolor=(0, 1, 1, .12),
        )
        self._axis.add_patch(self._selection_patch)
        self.draw_idle()

    def _on_click(self, event):
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        if event.inaxes is not self._axis or event.ydata is None:
            return
        row = int(np.floor(event.ydata))
        if 0 <= row < len(self._sorted_cell_ids):
            self.cellSelected.emit(int(self._sorted_cell_ids[row]))

class FOVCanvas(FigureCanvasQTAgg):
    """Clickable FOV matching the ROI selection behavior of the original HTML."""
    cellSelected = Signal(int)

    def __init__(self):
        self.figure = Figure(facecolor="#1b1e23", constrained_layout=True)
        super().__init__(self.figure)
        self.setObjectName("plotCanvas")
        self.setStyleSheet("background: #1b1e23; border: 0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(520, 520)
        self.figure.set_size_inches(5.2, 5.2, forward=True)
        self._axis = None
        self._id_map = None
        self._fov = None
        self._selection_patch = None
        self.mpl_connect("button_press_event", self._on_click)

    def set_group(self, group: GroupedMaps | None):
        self.figure.clear()
        self._axis = self.figure.subplots()
        self._id_map = None
        self._fov = None
        self._selection_patch = None
        if group is None or group.session.fov is None:
            self._axis.text(.5, .5, "FOV unavailable", ha="center", va="center",
                            color="#8a95aa", transform=self._axis.transAxes)
            self._axis.set_axis_off()
            self.draw_idle()
            return

        fov = group.session.fov
        self._fov = fov
        image = np.asarray(fov.mean_image, dtype=float)
        lo, hi = np.nanpercentile(image, [1, 99])
        image_norm = np.clip((image - lo) / (hi - lo), 0, 1) if hi > lo else image
        self._axis.imshow(image_norm, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")

        overlay = np.full(image.shape, np.nan, dtype=float)
        id_map = np.full(image.shape, -1, dtype=np.int32)
        info_values = []
        for row_idx in range(len(group.session.dataframe)):
            row = group.session.dataframe.iloc[int(row_idx)]
            cell_id = int(row.get("cell_id", row_idx))
            if cell_id not in fov.roi_xpix:
                continue
            xpix, ypix = fov.roi_xpix[cell_id], fov.roi_ypix[cell_id]
            inside = ((xpix >= 0) & (xpix < image.shape[1]) &
                      (ypix >= 0) & (ypix < image.shape[0]))
            xpix, ypix = xpix[inside], ypix[inside]
            value = float(row.get("spatial_information_bits", 0.0))
            if not np.isfinite(value):
                value = 0.0
            overlay[ypix, xpix] = value
            id_map[ypix, xpix] = cell_id
            info_values.append(value)
        self._id_map = id_map
        vmax = np.nanpercentile(info_values, 95) if info_values else 1.0
        vmax = max(float(vmax), 1e-9)
        masked = np.ma.masked_invalid(overlay)
        self._axis.imshow(masked, cmap=colormaps["viridis"], vmin=0, vmax=vmax,
                          alpha=.9, interpolation="nearest")
        self._axis.set_title(
            f"{group.session.display_name} FOV - click an ROI\n"
            f"{len(info_values)} displayed cells | color: spatial information",
            fontsize=8, color="white", pad=5,
        )
        self._axis.set_axis_off()
        self.draw_idle()

    def select_cell(self, cell_id: int | None):
        if self._selection_patch is not None:
            self._selection_patch.remove()
            self._selection_patch = None
        if cell_id is None or self._fov is None or cell_id not in self._fov.roi_xpix:
            self.draw_idle()
            return
        xpix, ypix = self._fov.roi_xpix[cell_id], self._fov.roi_ypix[cell_id]
        if not len(xpix):
            return
        self._selection_patch = Rectangle(
            (float(np.min(xpix)) - 2, float(np.min(ypix)) - 2),
            float(np.max(xpix) - np.min(xpix)) + 4,
            float(np.max(ypix) - np.min(ypix)) + 4,
            linewidth=2.0, edgecolor="cyan", facecolor=(0, 1, 1, .12),
        )
        self._axis.add_patch(self._selection_patch)
        self.draw_idle()

    def _on_click(self, event):
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        if event.inaxes is not self._axis or self._id_map is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= y < self._id_map.shape[0] and 0 <= x < self._id_map.shape[1]:
            cell_id = int(self._id_map[y, x])
            if cell_id >= 0:
                self.cellSelected.emit(cell_id)

class CellCanvas(PlotCanvas):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1180, 500)
        self.figure.set_size_inches(
            1180 / self.figure.dpi, 500 / self.figure.dpi, forward=True
        )

    def show_cell(self, group: GroupedMaps | None, source_idx: int = 0):
        self.figure.clear()
        self.last_heatmap_raw_range = (np.nan, np.nan)
        if group is None:
            self.show_groups([])
            return
        source_idx = int(np.clip(source_idx, 0, group.session.n_cells - 1))
        has_second_group = bool(group.label_b)
        ncols = 3 if has_second_group else 2
        axes = self.figure.subplots(1, ncols, squeeze=False)[0]
        trace_ax = axes[-1]
        x = group.session.bin_centres
        n_bins = group.session.tensor.shape[2]
        trials_a = (group.session.tensor[source_idx, group.lap_idx_a]
                    if len(group.lap_idx_a) else np.empty((0, n_bins)))
        trials_b = (group.session.tensor[source_idx, group.lap_idx_b]
                    if has_second_group and len(group.lap_idx_b)
                    else np.empty((0, n_bins)))

        finite_parts = [values[np.isfinite(values)]
                        for values in (trials_a, trials_b) if values.size]
        finite_parts = [values for values in finite_parts if len(values)]
        if finite_parts:
            pooled_values = np.concatenate(finite_parts)
            shared_min = float(np.min(pooled_values))
            shared_max = float(np.max(pooled_values))
        else:
            shared_min, shared_max = 0.0, 1.0
        self.last_heatmap_raw_range = (shared_min, shared_max)
        shared_range = shared_max - shared_min

        def joint_normalize(values):
            if not values.size:
                return values
            if shared_range > 0:
                return (values - shared_min) / shared_range
            return np.zeros_like(values, dtype=float)

        trial_entries = [(axes[0], trials_a, group.label_a)]
        if has_second_group:
            trial_entries.append((axes[1], trials_b, group.label_b))
        images = []
        for ax, trials, label in trial_entries:
            if len(trials):
                image = ax.imshow(
                    joint_normalize(trials), aspect="auto", cmap="viridis",
                    vmin=0, vmax=1, extent=[0, 180, len(trials), 0],
                    interpolation="none",
                )
                images.append(image)
            else:
                ax.text(.5, .5, "No trials", ha="center", va="center",
                        transform=ax.transAxes)
            suffix = "shared normalization" if has_second_group else "normalized"
            ax.set_title(f"{group.session.display_name}\n{label} laps | {suffix}", fontsize=8, pad=4)
            ax.set(xlabel="Position (cm)", ylabel="Lap")

        if len(trials_a):
            trace_ax.plot(x, np.nanmean(trials_a, axis=0),
                          color="#00c2ff", label=group.label_a)
        if has_second_group and len(trials_b):
            trace_ax.plot(x, np.nanmean(trials_b, axis=0),
                          color="#ff5f87", label=group.label_b)
        cell_id = group.session.dataframe.iloc[source_idx].get("cell_id", source_idx)
        trace_ax.set_title(f"{group.session.display_name}\ncell_id={cell_id} | raw activity", fontsize=8, pad=4)
        trace_ax.set(xlabel="Position (cm)", ylabel="dF/F")
        if len(trials_a) or len(trials_b):
            trace_ax.legend(fontsize=7)
        for ax in axes:
            ax.set_facecolor("#272c36")
            ax.tick_params(colors="#d8dee9", labelsize=7)
            ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
            ax.xaxis.label.set_color("#d8dee9"); ax.yaxis.label.set_color("#d8dee9")
            ax.title.set_color("white")
        if images:
            heat_axes = [entry[0] for entry in trial_entries]
            colorbar = self.figure.colorbar(images[0], ax=heat_axes, shrink=.55, pad=.02)
            colorbar_label = ("Normalized activity (shared scale)"
                              if has_second_group else "Normalized activity")
            colorbar.set_label(colorbar_label, color="#d8dee9", fontsize=8)
            colorbar.ax.tick_params(colors="#d8dee9", labelsize=7)
        self.draw_idle()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.groups: list[GroupedMaps] = []
        self.pool = QThreadPool.globalInstance()
        self.setWindowTitle("GCaMP Session Explorer"); self.resize(1500, 900)
        self._build(); self._load_theme(); self.refresh_sessions()

    def _build(self):
        root = QWidget(objectName="root"); self.setCentralWidget(root)
        outer = QHBoxLayout(root); outer.setContentsMargins(0, 0, 0, 0)
        sidebar = QFrame(objectName="sidebar"); sidebar.setMinimumWidth(250); sidebar.setMaximumWidth(480)
        side = QVBoxLayout(sidebar); side.setContentsMargins(16, 18, 16, 18)
        side.addWidget(QLabel("GCaMP Explorer", objectName="title"))
        side.addWidget(QLabel("Multi-session place-cell viewer", objectName="subtitle"))
        data_box = QGroupBox("Data source"); data_l = QVBoxLayout(data_box)
        self.root_edit = QLineEdit(str(DEFAULT_DATA_ROOT)); data_l.addWidget(self.root_edit)
        refresh = QPushButton("Refresh sessions"); refresh.clicked.connect(self.refresh_sessions); data_l.addWidget(refresh)
        self.session_list = QListWidget(); self.session_list.setSelectionMode(QListWidget.ExtendedSelection); data_l.addWidget(self.session_list, 1)
        side.addWidget(data_box, 1)
        split_box = QGroupBox("Trial groups"); split_l = QVBoxLayout(split_box)
        self.mode = QComboBox(); self.mode.addItem("All valid trials - no split", "all_valid"); self.mode.addItem("Early / late - median", "lick_median"); self.mode.addItem("Early / late - thresholds", "lick_threshold"); self.mode.addItem("Stim / control", "stim_control"); split_l.addWidget(self.mode)
        row = QHBoxLayout(); self.early = QDoubleSpinBox(); self.early.setRange(0,180); self.early.setValue(100); self.early.setSuffix(" cm"); self.late = QDoubleSpinBox(); self.late.setRange(0,180); self.late.setValue(120); self.late.setSuffix(" cm"); row.addWidget(self.early); row.addWidget(self.late); split_l.addLayout(row)
        self.speed_match_lick = QCheckBox("Speed-match early / late trials")
        self.speed_match_lick.setChecked(False)
        split_l.addWidget(self.speed_match_lick)
        speed_row = QHBoxLayout()
        self.speed_tolerance_label = QLabel("Tolerance:")
        self.speed_tolerance = QDoubleSpinBox()
        self.speed_tolerance.setRange(0.1, 5.0)
        self.speed_tolerance.setDecimals(1)
        self.speed_tolerance.setSingleStep(0.1)
        self.speed_tolerance.setValue(2.0)
        self.speed_tolerance.setSuffix(" SD")
        speed_row.addWidget(self.speed_tolerance_label)
        speed_row.addWidget(self.speed_tolerance)
        split_l.addLayout(speed_row)
        self.mode.currentIndexChanged.connect(self._update_trial_controls)
        self.speed_match_lick.toggled.connect(self._update_trial_controls)
        self.significant = QCheckBox("Significant place cells only")
        self.significant.setChecked(True)
        split_l.addWidget(self.significant)
        side.addWidget(split_box)

        self.significance_box = QGroupBox("Place-cell significance")
        significance_l = QFormLayout(self.significance_box)
        significance_l.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.pc_min_peak = QDoubleSpinBox()
        self.pc_min_peak.setRange(0, 10); self.pc_min_peak.setDecimals(3)
        self.pc_min_peak.setSingleStep(0.01); self.pc_min_peak.setValue(0.1)
        self.pc_min_ratio = QDoubleSpinBox()
        self.pc_min_ratio.setRange(0, 100); self.pc_min_ratio.setDecimals(2)
        self.pc_min_ratio.setSingleStep(0.1); self.pc_min_ratio.setValue(2.0)
        self.pc_min_width = QDoubleSpinBox()
        self.pc_min_width.setRange(0, 180); self.pc_min_width.setDecimals(1)
        self.pc_min_width.setSingleStep(1); self.pc_min_width.setValue(18.0)
        self.pc_min_width.setSuffix(" cm")
        self.pc_min_transient = QDoubleSpinBox()
        self.pc_min_transient.setRange(0, 1); self.pc_min_transient.setDecimals(3)
        self.pc_min_transient.setSingleStep(0.01); self.pc_min_transient.setValue(0.1)
        significance_l.addRow("Min peak dF/F", self.pc_min_peak)
        significance_l.addRow("Min in/out ratio", self.pc_min_ratio)
        significance_l.addRow("Min field width", self.pc_min_width)
        significance_l.addRow("Min transient frac.", self.pc_min_transient)
        reset_pc = QPushButton("Reset defaults")
        reset_pc.clicked.connect(self._reset_pc_thresholds)
        significance_l.addRow(reset_pc)
        apply_pc = QPushButton("Apply and reload")
        apply_pc.clicked.connect(self.load_selected)
        significance_l.addRow(apply_pc)
        side.addWidget(self.significance_box)
        self.load_button = QPushButton("Load selected sessions", objectName="primary"); self.load_button.clicked.connect(self.load_selected); side.addWidget(self.load_button)
        export_button = QPushButton("Export interactive HTML"); export_button.clicked.connect(self.export); side.addWidget(export_button)
        self.main_splitter = QSplitter(Qt.Horizontal)

        content = QWidget(); content_l = QVBoxLayout(content); content_l.setContentsMargins(16, 14, 16, 10)
        top = QFrame(objectName="topBar"); top_l = QHBoxLayout(top); top_l.addWidget(QLabel("Session comparison", objectName="title")); top_l.addStretch(); self.summary = QLabel("No data loaded", objectName="subtitle"); top_l.addWidget(self.summary); content_l.addWidget(top)
        self.tabs = QTabWidget(); self.overview = PlotCanvas(); self.combined_canvas = CombinedHeatmapCanvas(); self.metrics_canvas = MetricsCanvas(); self.stats_canvas = PropertyStatisticsCanvas(); self.sequence_canvas = SequenceCanvas(); self.fov_canvas = FOVCanvas(); self.cell_canvas = CellCanvas(); self.log = QTextEdit(); self.log.setReadOnly(True)
        self.overview_scroll = self._scrollable(self.overview)
        self.tabs.addTab(self._plot_panel(self.overview, self.overview_scroll), "Overview")
        self.combined_scroll = self._scrollable(self.combined_canvas)
        self.tabs.addTab(self._plot_panel(self.combined_canvas, self.combined_scroll), "Combined heatmaps")
        metrics_page = QWidget()
        metrics_l = QVBoxLayout(metrics_page)
        metrics_note = QLabel(
            "Rows compare trial types using the same all-lap threshold-defined place-cell cohort. "
            "All other metrics are recomputed from each trial subset; continuous values are mean +/- SEM. "
            "Reward: 150-180 cm; run onset: 0-30 cm; enrichment uses a 16.7% uniform expectation."
        )
        metrics_note.setWordWrap(True)
        metrics_l.addWidget(metrics_note)
        self.metrics_table = QTableWidget(0, 14)
        self.metrics_table.setHorizontalHeaderLabels([
            "Session", "Treatment", "Trial type", "PCs / total", "PC (%)", "Spatial info",
            "Reward fields (%)", "Reward enrich.", "Onset fields (%)",
            "Onset enrich.", "Width (cm)", "Odd-even r",
            "Consecutive r", "In/out ratio",
        ])
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.metrics_table.setMinimumHeight(190)
        self.metrics_table.setMaximumHeight(280)
        metrics_l.addWidget(self.metrics_table)
        self.metrics_scroll = self._scrollable(self.metrics_canvas)
        metrics_l.addWidget(self._plot_panel(self.metrics_canvas, self.metrics_scroll), 1)
        self.tabs.addTab(metrics_page, "Place-cell metrics")
        self.metrics_canvas.show_groups([])

        stats_page = QWidget()
        stats_l = QVBoxLayout(stats_page)
        stats_controls = QHBoxLayout()
        self.stats_mode = QComboBox()
        self.stats_mode.addItem("Baseline vs drug sessions", "baseline_drug")
        self.stats_mode.addItem("Drug vs saline sessions", "drug_saline")
        self.stats_mode.addItem("Trial types within sessions", "trial_types")
        self.stats_drug = QComboBox()
        self.stats_drug.addItems(["Any drug", "SCH", "Prazosin", "Propranolol"])
        self.stats_session_trial = QComboBox()
        self.stats_trial_a = QComboBox()
        self.stats_trial_b = QComboBox()
        stats_controls.addWidget(QLabel("Comparison:"))
        stats_controls.addWidget(self.stats_mode)
        stats_controls.addWidget(QLabel("Drug:"))
        stats_controls.addWidget(self.stats_drug)
        stats_controls.addWidget(QLabel("Session metric trial type:"))
        stats_controls.addWidget(self.stats_session_trial)
        stats_controls.addWidget(QLabel("Trial A:"))
        stats_controls.addWidget(self.stats_trial_a)
        stats_controls.addWidget(QLabel("Trial B:"))
        stats_controls.addWidget(self.stats_trial_b)
        stats_controls.addStretch()
        stats_l.addLayout(stats_controls)
        self.stats_summary = QLabel(
            "Session-level statistics; cells are averaged within each recording."
        )
        self.stats_summary.setWordWrap(True)
        stats_l.addWidget(self.stats_summary)
        self.stats_scroll = self._scrollable(self.stats_canvas)
        stats_l.addWidget(self._plot_panel(self.stats_canvas, self.stats_scroll), 1)
        self.tabs.addTab(stats_page, "Property statistics")
        self.stats_canvas.show_comparison(None)
        for combo in (self.stats_mode, self.stats_drug, self.stats_session_trial,
                      self.stats_trial_a, self.stats_trial_b):
            combo.currentIndexChanged.connect(self._update_statistics_controls)
        self._update_statistics_controls()

        cell_page = QWidget()
        cell_l = QVBoxLayout(cell_page)
        controls = QHBoxLayout()
        self.cell_session = QComboBox()
        self.cell_index = QComboBox()
        self.cell_session.currentIndexChanged.connect(self._update_cells)
        self.cell_index.currentIndexChanged.connect(self._plot_cell)
        controls.addWidget(QLabel("Session:"))
        controls.addWidget(self.cell_session)
        controls.addWidget(QLabel("Cell:"))
        controls.addWidget(self.cell_index)
        controls.addStretch()
        cell_l.addLayout(controls)
        self.fov_canvas.cellSelected.connect(self._select_cell_from_plot)
        self.sequence_canvas.cellSelected.connect(self._select_cell_from_plot)
        self.fov_scroll = self._scrollable(self.fov_canvas)
        self.sequence_scroll = self._scrollable(self.sequence_canvas)
        self.cell_scroll = self._scrollable(self.cell_canvas)
        self.cell_splitter = QSplitter(Qt.Horizontal)
        self.cell_map_splitter = QSplitter(Qt.Vertical)
        self.fov_panel = self._plot_panel(self.fov_canvas, self.fov_scroll)
        self.sequence_panel = self._plot_panel(self.sequence_canvas, self.sequence_scroll)
        self.activity_panel = self._plot_panel(self.cell_canvas, self.cell_scroll)
        self.fov_panel.setMinimumSize(360, 240)
        self.sequence_panel.setMinimumSize(360, 220)
        self.activity_panel.setMinimumWidth(500)
        self.cell_map_splitter.addWidget(self.fov_panel)
        self.cell_map_splitter.addWidget(self.sequence_panel)
        self.cell_map_splitter.setSizes([430, 390])
        self.cell_map_splitter.setStretchFactor(0, 1)
        self.cell_map_splitter.setStretchFactor(1, 1)
        self.cell_splitter.addWidget(self.cell_map_splitter)
        self.cell_splitter.addWidget(self.activity_panel)
        self.cell_splitter.setSizes([460, 940])
        self.cell_splitter.setStretchFactor(1, 1)
        cell_l.addWidget(self.cell_splitter)
        self.tabs.addTab(cell_page, "Cell activity")
        self.tabs.addTab(self.log, "Details")
        content_l.addWidget(self.tabs, 1)
        self.main_splitter.addWidget(sidebar)
        self.main_splitter.addWidget(content)
        self.main_splitter.setSizes([320, 1180])
        self.main_splitter.setStretchFactor(1, 1)
        outer.addWidget(self.main_splitter)
        self._update_trial_controls()
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _scrollable(canvas: FigureCanvasQTAgg) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("plotScroll")
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: #1b1e23; border: 0; } "
            "QScrollArea > QWidget > QWidget { background: #1b1e23; }"
        )
        scroll.viewport().setStyleSheet("background: #1b1e23;")
        scroll.setWidget(canvas)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

    @staticmethod
    def _plot_panel(canvas: FigureCanvasQTAgg, scroll: QScrollArea) -> QWidget:
        panel = QWidget()
        panel.setObjectName("plotPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = NavigationToolbar2QT(canvas, panel)
        toolbar.setObjectName("plotToolbar")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        layout.addWidget(toolbar)
        hint = QLabel(
            "Pan: click Pan, then drag  |  Rescale: click Zoom, then drag a box  |  Home: reset"
        )
        hint.setObjectName("plotHint")
        layout.addWidget(hint)
        layout.addWidget(scroll, 1)
        return panel

    def _load_theme(self):
        self.setStyleSheet((Path(__file__).with_name("theme.qss")).read_text(encoding="utf-8"))

    def refresh_sessions(self):
        selected = {
            i.data(Qt.UserRole) or i.text() for i in self.session_list.selectedItems()
        }
        self.session_list.clear()
        sessions = discover_sessions(self.root_edit.text())
        for rec in sessions:
            treatment = session_treatment_label(rec)
            item = QListWidgetItem(f"{rec}  [{treatment}]")
            item.setData(Qt.UserRole, rec)
            self.session_list.addItem(item)
            item.setSelected(rec in selected)
        self.statusBar().showMessage(f"Found {len(sessions)} complete sessions")

    def _update_trial_controls(self, *_):
        mode = self.mode.currentData()
        is_lick = isinstance(mode, str) and mode.startswith("lick_")
        is_threshold = mode == "lick_threshold"
        self.early.setEnabled(is_threshold)
        self.late.setEnabled(is_threshold)
        self.speed_match_lick.setEnabled(is_lick)
        tolerance_enabled = is_lick and self.speed_match_lick.isChecked()
        self.speed_tolerance_label.setEnabled(tolerance_enabled)
        self.speed_tolerance.setEnabled(tolerance_enabled)
    def _reset_pc_thresholds(self):
        self.pc_min_peak.setValue(0.1)
        self.pc_min_ratio.setValue(2.0)
        self.pc_min_width.setValue(18.0)
        self.pc_min_transient.setValue(0.1)

    def load_selected(self):
        sessions=[i.data(Qt.UserRole) or i.text() for i in self.session_list.selectedItems()]
        if not sessions: QMessageBox.information(self,"Select sessions","Select at least one session."); return
        options = dict(
            mode=self.mode.currentData(),
            early_threshold=self.early.value(),
            late_threshold=self.late.value(),
            significant_only=self.significant.isChecked(),
            min_peak_dff=self.pc_min_peak.value(),
            min_in_out_ratio=self.pc_min_ratio.value(),
            min_width_cm=self.pc_min_width.value(),
            min_transient_fraction=self.pc_min_transient.value(),
            speed_match_lick=self.speed_match_lick.isChecked(),
            speed_tolerance=self.speed_tolerance.value(),
        )
        self.load_button.setEnabled(False); self.statusBar().showMessage(f"Loading {len(sessions)} session(s)...")
        worker=LoadWorker(self.root_edit.text(),sessions,options); worker.signals.completed.connect(self._loaded); self.pool.start(worker)

    def _loaded(self, groups, errors):
        self.groups=list(groups); self.load_button.setEnabled(True); self.overview.show_groups(self.groups); self.combined_canvas.show_groups(self.groups); self.metrics_canvas.show_groups(self.groups); self._update_metrics_table(); self._refresh_statistics_options()
        self.cell_session.clear(); self.cell_session.addItems([g.session.display_name for g in self.groups]); self._update_cells()
        lines=[f"{g.session.display_name}: displayed={len(g.cell_indices)} cells; place cells={len(g.place_cell_indices) if g.place_cell_indices is not None else len(g.cell_indices)}; {g.label_a}={len(g.lap_idx_a)} laps; {g.label_b}={len(g.lap_idx_b)} laps; {g.details}" for g in self.groups]
        if errors: lines += ["", "Skipped:", *errors]
        self.log.setPlainText("\n".join(lines)); self.summary.setText(f"{len(self.groups)} session(s) loaded"); self.statusBar().showMessage("Loaded" if self.groups else "No sessions could be loaded")

    @staticmethod
    def _mean_sem_text(values, digits=3):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return "-"
        mean = float(np.mean(values))
        sem = (float(np.std(values, ddof=1) / np.sqrt(len(values)))
               if len(values) > 1 else 0.0)
        return f"{mean:.{digits}f} +/- {sem:.{digits}f}"

    def _update_metrics_table(self):
        metrics = self.metrics_canvas.metric_data
        self.metrics_table.setRowCount(len(metrics))
        for row, values in enumerate(metrics):
            fields = [
                values["rec_id"],
                values["treatment"],
                f'{values["trial_type"]} (laps={values["lap_count"]})',
                f'{values["n_place_cells"]} / {values["n_cells"]}',
                f'{values["place_cell_percent"]:.2f}',
                self._mean_sem_text(values["spatial_information"]),
                f'{100 * values["reward_fraction"]:.2f}',
                f'{values["reward_enrichment"]:.2f}x',
                f'{100 * values["onset_fraction"]:.2f}',
                f'{values["onset_enrichment"]:.2f}x',
                self._mean_sem_text(values["field_width"], 2),
                self._mean_sem_text(values["odd_even_stability"]),
                self._mean_sem_text(values["consecutive_stability"]),
                self._mean_sem_text(values["in_out_ratio"], 2),
            ]
            for column, value in enumerate(fields):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.metrics_table.setItem(row, column, item)
    @staticmethod
    def _set_combo_values(combo: QComboBox, values: list[str], default_index=0):
        previous = combo.currentText()
        combo.blockSignals(True)
        combo.clear(); combo.addItems(values)
        restored = combo.findText(previous)
        if restored >= 0:
            combo.setCurrentIndex(restored)
        elif values:
            combo.setCurrentIndex(min(default_index, len(values) - 1))
        combo.blockSignals(False)

    def _refresh_statistics_options(self):
        trial_types = available_trial_types(self.metrics_canvas.metric_data)
        self._set_combo_values(self.stats_session_trial, trial_types, 0)
        self._set_combo_values(self.stats_trial_a, trial_types, 0)
        self._set_combo_values(self.stats_trial_b, trial_types, 1)
        self._update_statistics_controls()

    def _update_statistics_controls(self, *_):
        mode = self.stats_mode.currentData()
        session_comparison = mode in {"baseline_drug", "drug_saline"}
        self.stats_drug.setEnabled(session_comparison)
        self.stats_session_trial.setEnabled(session_comparison)
        self.stats_trial_a.setEnabled(mode == "trial_types")
        self.stats_trial_b.setEnabled(mode == "trial_types")
        self._plot_statistics()

    def _plot_statistics(self):
        rows = self.metrics_canvas.metric_data
        if not rows:
            self.stats_canvas.show_comparison(None)
            self.stats_summary.setText(
                "Load sessions to calculate recording-level property statistics."
            )
            return
        try:
            comparison = build_property_comparison(
                rows, comparison_mode=self.stats_mode.currentData(),
                drug_family=self.stats_drug.currentText(),
                session_trial_type=self.stats_session_trial.currentText(),
                trial_type_a=self.stats_trial_a.currentText(),
                trial_type_b=self.stats_trial_b.currentText(),
            )
            self.stats_canvas.show_comparison(comparison)
            if comparison["paired"]:
                sample_text = f'n={comparison["n_a"]} paired recordings'
            else:
                sample_text = (f'n={comparison["n_a"]} {comparison["labels"][0]} and '
                               f'n={comparison["n_b"]} {comparison["labels"][1]} sessions')
            self.stats_summary.setText(
                f'{comparison["description"]}; {sample_text}. '
                "Each point is one recording-level mean; cell-level values are not treated as independent samples."
            )
        except Exception as exc:
            self.stats_canvas.show_comparison(None)
            self.stats_summary.setText(f"Statistics unavailable: {exc}")
    def _update_cells(self):
        self.cell_index.blockSignals(True)
        self.cell_index.clear()
        idx = self.cell_session.currentIndex()
        group = self.groups[idx] if 0 <= idx < len(self.groups) else None
        if group is not None:
            for row_idx in range(group.session.n_cells):
                cell_id = int(group.session.dataframe.iloc[row_idx].get("cell_id", row_idx))
                self.cell_index.addItem(str(cell_id), row_idx)
        self.cell_index.blockSignals(False)
        self.fov_canvas.set_group(group)
        self.sequence_canvas.set_group(group)
        self._plot_cell()

    def _select_cell_from_plot(self, cell_id: int):
        combo_idx = self.cell_index.findText(str(int(cell_id)), Qt.MatchExactly)
        if combo_idx >= 0:
            self.cell_index.setCurrentIndex(combo_idx)
            self.statusBar().showMessage(f"Selected cell {cell_id} from map")

    def _plot_cell(self):
        idx = self.cell_session.currentIndex()
        group = self.groups[idx] if 0 <= idx < len(self.groups) else None
        source_idx = self.cell_index.currentData() if group is not None else None
        self.cell_canvas.show_cell(group, int(source_idx) if source_idx is not None else 0)
        cell_id = self.cell_index.currentText() if group is not None else ""
        self.fov_canvas.select_cell(int(cell_id) if cell_id else None)
        self.sequence_canvas.select_cell(int(cell_id) if cell_id else None)
    def export(self):
        if not self.groups: QMessageBox.information(self,"Nothing to export","Load at least one session first."); return
        path,_=QFileDialog.getSaveFileName(self,"Export HTML",str(Path.home()/"gcamp_session_comparison.html"),"HTML files (*.html)")
        if not path: return
        try: export_html(self.groups, path, property_comparison=self.stats_canvas.last_comparison, statistics_figure=self.stats_canvas.figure); self.statusBar().showMessage(f"Exported {path}")
        except Exception as exc: QMessageBox.critical(self,"Export failed",str(exc))


def main():
    app=QApplication(sys.argv); window=MainWindow(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


















