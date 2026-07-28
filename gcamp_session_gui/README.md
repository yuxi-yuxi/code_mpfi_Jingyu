# GCaMP Session Explorer

A PySide6 desktop viewer for the processed drug-infusion place-cell sessions. Its dark sidebar, blue accent, rounded panels, and high-DPI setup follow the visual conventions of the MIT-licensed PyDracula project.

## Features

- Multi-select one or more of the 129 complete session folders, each labeled as Baseline, Saline, SCH, Prazosin, or Propranolol from `drug_infusion.rec_lst_infusion`.
- Compare place-cell maps using one shared per-cell min-max scale across trial types, with population means shown in raw dF/F.
- View pooled trial-type heatmaps across all selected sessions, globally sorted by field location with matched cell ordering.
- Adjust minimum peak dF/F, in/out ratio, field width, and transient fraction used for place-cell significance.
- Compare each selected session and trial type in a dedicated metrics tab: place-cell percentage, spatial information, field-location histogram, reward/run-onset field enrichment, field width, odd-even and consecutive stability, and in-field/out-of-field ratio. Exact values are also included in HTML exports.
- Plot recording-level place-cell property statistics in a separate tab using `common.plotting_functions_Jingyu`: paired baseline/drug sessions, drug-family-filtered unpaired drug/saline sessions, or paired trial types within sessions. The selected nine-panel statistics figure is embedded in HTML exports.
- View all behavior-valid trials without a split, or split laps by median first lick, adjustable early/late thresholds, or stim/control labels. Early/late groups can optionally be distance-speed matched with an adjustable SD tolerance.
- Select a cell from the cached FOV, synchronized dropdown, or clickable place-cell sequence sorted by field location; the selected ROI and sequence row are highlighted in cyan.
- Browse every recorded cell's lap-by-position activity with heatmaps jointly normalized across trial types; mean traces remain raw dF/F.
- Export the current multi-session comparison as a self-contained interactive HTML file.

Property statistics use one recording as one sample. Cell-level properties are averaged within recording before testing. Baseline/drug comparisons pair sessions from the same animal/date; trial-type comparisons pair conditions within recording; drug/saline comparisons are unpaired and saline sessions are restricted to the selected drug experiment family. Jingyu's paired plotting helper reports paired t-test and Wilcoxon results (plus reference unpaired tests), while the unpaired helper reports t-test and rank-sum results.
Session treatment labels are built from the concatenated `rec_SCH`/`rec_SCH_ctrl`, `rec_praz`/`rec_praz_ctrl`, and `rec_prop`/`rec_prop_ctrl` dataframes. Their paired `session` and `label` arrays are expanded into full `{animal}-{date}-{session}` IDs. `ctrl` and `ctrl2` are displayed as Saline, repeated drug labels such as `SCH2` retain the drug name, and baseline sessions remain Baseline. All 129 processed sessions are covered without conflicts.
Trial-type comparisons use one paired cohort: place cells are classified once from all valid laps with the current significance thresholds, so the place-cell percentage is intentionally shared across trial types. Spatial information (uniform occupancy), peak location, field width, in/out ratio, odd-even stability, and consecutive stability are then recomputed from each trial subset. The condition-specific tentative field containing the condition-specific peak defines width and in/out ratio. Reward fields fall in 150-180 cm and run-onset fields in 0-30 cm; enrichment is the observed field fraction divided by the 16.7% uniform expectation. Continuous table entries are mean +/- SEM across the shared place-cell cohort.
Speed matching follows the original single-session script: distance-aligned speed is summarized in six 30-cm bins, and each group retains trials lying within the other group's mean +/- the selected tolerance times its SD. The default tolerance is 2 SD. Retained lap counts and the post-match speed-bin p-value range are reported in Details and HTML exports.

Lap/trial mapping follows `test_place_cell_by_lick_single_session.py`: `lap_trial_idx[lap]` is used to look up that lap's behavioral trial label. For these processed pickles, stim/control is derived from the trial-aligned `pulse_descriptions` because all 129 files lack pulse start/end timestamps; non-empty descriptions are stim trials. If timestamp fields are added later, the loader automatically uses `align_pulses()` from `test_lc_stim_gcamp.py`.

## Install and run

From the repository root:

```powershell
python -m pip install -r gcamp_session_gui/requirements.txt
python -m gcamp_session_gui.app
```
```spyder IDE
%cd r"Z:\Jingyu\code_mpfi_Jingyu"
%run -m gcamp_session_gui
```

The default data root is already set to:

`Z:\Jingyu\raw_data\gcamp_drug_infusion\processed_data`

You can change it in the sidebar and click **Refresh sessions**.

## Theme attribution

Interface styling is inspired by [PyDracula](https://github.com/Wanderson-Magalhaes/Modern_GUI_PyDracula_PySide6_or_PyQt6), copyright Wanderson Magalhaes, MIT License. No upstream code or image assets are bundled.



## Session-local FOV cache

Each processed session folder contains `{rec_id}_fov_data.npz`. It stores the mean FOV image and a flattened, pickle-free mapping from dataframe `cell_id` values to ROI pixels. The GUI reads this compact cache first, so interactive FOV viewing does not depend on the original Suite2p or raw-signals folders.






