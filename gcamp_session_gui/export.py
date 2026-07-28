"""Plotly HTML report generation."""
from __future__ import annotations

from pathlib import Path
from html import escape
from io import BytesIO
import base64
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import GroupedMaps, normalized_sorted_maps, trial_type_metrics


def _metrics_table_html(groups: list[GroupedMaps]) -> str:
    def mean_sem(values, digits=3):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return "-"
        mean = float(np.mean(values))
        sem = (float(np.std(values, ddof=1) / np.sqrt(len(values)))
               if len(values) > 1 else 0.0)
        return f"{mean:.{digits}f} +/- {sem:.{digits}f}"

    headers = [
        "Session", "Treatment", "Trial type", "PCs / total", "PC (%)", "Spatial information",
        "Reward fields (%)", "Reward enrichment", "Onset fields (%)",
        "Onset enrichment", "Field width (cm)", "Odd-even r",
        "Consecutive r", "In/out ratio",
    ]
    rows = []
    for group in groups:
        for m in trial_type_metrics(group):
            cells = [
                m["rec_id"], m["treatment"],
                f'{m["trial_type"]} (laps={m["lap_count"]})',
                f'{m["n_place_cells"]} / {m["n_cells"]}',
                f'{m["place_cell_percent"]:.2f}', mean_sem(m["spatial_information"]),
                f'{100 * m["reward_fraction"]:.2f}', f'{m["reward_enrichment"]:.2f}x',
                f'{100 * m["onset_fraction"]:.2f}', f'{m["onset_enrichment"]:.2f}x',
                mean_sem(m["field_width"], 2), mean_sem(m["odd_even_stability"]),
                mean_sem(m["consecutive_stability"]), mean_sem(m["in_out_ratio"], 2),
            ]
            rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in cells) + "</tr>")
    header = "<tr>" + "".join(f"<th>{escape(v)}</th>" for v in headers) + "</tr>"
    grouping_details = "".join(
        f"<li><strong>{escape(group.session.display_name)}</strong>: {escape(group.details)}</li>"
        for group in groups
    )
    return f"""
    <section class="metrics">
      <h2>Place-cell quantitative summary</h2>
      <p>Trial types use the same all-lap threshold-defined place-cell cohort; all other metrics are
      recomputed from each trial subset. Reward zone: 150-180 cm; run-onset zone: 0-30 cm.
      Each spans 16.7% of the track; enrichment is observed field fraction divided by this uniform
      expectation. Continuous metrics are mean +/- SEM across the shared place-cell cohort.</p>
      <ul>{grouping_details}</ul>
      <div class="table-wrap"><table>{header}{''.join(rows)}</table></div>
    </section>
    """

def _statistics_figure_html(comparison: dict | None, figure=None) -> str:
    if comparison is None or figure is None:
        return ""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, facecolor="#1b1e23")
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    labels = comparison["labels"]
    if comparison["paired"]:
        sample_text = f'n={comparison["n_a"]} paired recordings'
    else:
        sample_text = (f'n={comparison["n_a"]} {labels[0]} sessions; '
                       f'n={comparison["n_b"]} {labels[1]} sessions')
    return f"""
    <section class="metrics statistics">
      <h2>Place-cell property statistics</h2>
      <p>{escape(comparison["description"])}; {escape(sample_text)}.
      Each point is one recording-level mean.</p>
      <img alt="Place-cell property statistics" src="data:image/png;base64,{image_data}">
    </section>
    """

def export_html(groups: list[GroupedMaps], output: Path | str,
                property_comparison: dict | None = None,
                statistics_figure=None) -> Path:
    if not groups:
        raise ValueError("No loaded results to export")
    single_condition = all(not group.label_b for group in groups)
    ncols = 2 if single_condition else 3
    titles = []
    for group in groups:
        titles.append(f"{group.session.display_name}<br>{group.label_a}")
        if not single_condition:
            titles.append(f"{group.session.display_name}<br>{group.label_b}")
        titles.append(f"{group.session.display_name}<br>Population mean")
    fig = make_subplots(rows=len(groups), cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.05, vertical_spacing=0.12)
    for row, group in enumerate(groups, 1):
        maps_a, maps_b = normalized_sorted_maps(group)
        x = group.session.bin_centres
        fig.add_trace(
            go.Heatmap(
                z=maps_a, x=x, colorscale="Viridis", zmin=0, zmax=1,
                showscale=False,
                hovertemplate="cell row=%{y}<br>position=%{x:.1f} cm<br>norm=%{z:.3f}<extra></extra>",
            ), row=row, col=1,
        )
        mean_col = ncols
        if not single_condition:
            fig.add_trace(
                go.Heatmap(
                    z=maps_b, x=x, colorscale="Viridis", zmin=0, zmax=1,
                    showscale=False,
                    hovertemplate="cell row=%{y}<br>position=%{x:.1f} cm<br>norm=%{z:.3f}<extra></extra>",
                ), row=row, col=2,
            )
        fig.add_trace(
            go.Scatter(
                x=x, y=np.nanmean(group.maps_a, axis=0),
                name=group.label_a, legendgroup=group.label_a,
                showlegend=row == 1, line=dict(color="#00c2ff"),
            ), row=row, col=mean_col,
        )
        if not single_condition and len(group.lap_idx_b):
            fig.add_trace(
                go.Scatter(
                    x=x, y=np.nanmean(group.maps_b, axis=0),
                    name=group.label_b, legendgroup=group.label_b,
                    showlegend=row == 1, line=dict(color="#ff5f87"),
                ), row=row, col=mean_col,
            )
        for col in range(1, ncols + 1):
            fig.update_xaxes(title_text="Position (cm)", row=row, col=col)
        fig.update_yaxes(title_text="Cells", row=row, col=1)
        fig.update_yaxes(title_text="Mean dF/F", row=row, col=mean_col)
    fig.update_layout(template="plotly_dark", title="GCaMP session comparison",
                      height=max(500, 390 * len(groups)), margin=dict(t=80),
                      paper_bgcolor="#1b1e23", plot_bgcolor="#272c36")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_html = fig.to_html(include_plotlyjs=True, full_html=False)
    statistics_html = _statistics_figure_html(property_comparison, statistics_figure)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GCaMP session comparison</title>
<style>
body {{ margin: 0; background: #1b1e23; color: #d8dee9; font-family: Arial, sans-serif; }}
.metrics {{ padding: 20px 28px 4px; }}
.metrics h2 {{ color: white; margin: 0 0 8px; }}
.metrics p {{ color: #aeb8ca; font-size: 13px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #4b5363; padding: 7px 9px; text-align: center; white-space: nowrap; }}
th {{ background: #272c36; color: #00c2ff; }}
tr:nth-child(even) {{ background: #22262e; }}
.statistics img {{ display: block; width: 100%; max-width: 1500px; margin: 10px auto; }}
</style></head><body>{_metrics_table_html(groups)}{statistics_html}{plot_html}</body></html>"""
    output.write_text(document, encoding="utf-8")
    return output

