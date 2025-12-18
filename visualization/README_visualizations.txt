Visualization Output Guide
Generated: 2025-12-01 18:55:36 EST
Last recorded command: python plot_all_py.py
Last recorded run time: 2025-12-01 18:55:36 EST (rerun and update after generating fresh plots)
PNG storage: every script saves PNGs into a "png" subfolder that mirrors the PDF location (e.g., plots_nov27/png/...).

Scripts and Outputs
- plot_color_legend.py → color_legend
- plot_cell_overall.py → cell_overall_ap
- plot_cell_per_type.py → cell_per_type
- plot_marker_comp.py → marker_variant_comparison
- plot_marker.py → marker_comparison_pooled / per_type / compact
- plot_nuclei_overall.py → nuclei_overall_ap
- plot_side_by_side.py → cell_vs_nuclei_side_by_side
- more_metrics.py → metrics_boxplot_pooled, metrics_boundary_per_type, metrics_precision_per_type, metrics_recall_per_type, metrics_boxplot_comprehensive, metrics_violin_pooled
- plot_nuclei_marker_all.py → plots/nuclei/*, plots/marker/*, metrics_boxplot_comprehensive_nuclei, metrics_boxplot_comprehensive_marker

Root-Level Exports (plots_nov27)
1. color_legend (PDF: color_legend.pdf | PNG: png/color_legend.png)
   Caption: Reference palette and style guide.
   Description: Shows the approved colors, line styles, and markers for each algorithm so you can verify every plot uses config-defined styling.
2. cell_overall_ap (PDF/PNG)
   Caption: Precision vs IoU for overall cell segmentation.
   Description: Summarizes mean AP curves for all algorithms to compare global cell performance.
3. cell_per_type (PDF/PNG)
   Caption: Precision vs IoU per cell marker.
   Description: Faceted AP curves filtered by OLIG2, NEUN, IBA1, and GFAP images for cross-marker trends.
4. marker_variant_comparison (PDF/PNG)
   Caption: OLIG2 2-channel vs marker-only comparison.
   Description: Side-by-side OLIG2 curves contrasting both input variants.
5. marker_comparison_pooled (PDF/PNG)
   Caption: Pooled variant comparison.
   Description: Two-panel figure (2-channel vs marker-only) using all cell types combined.
6. marker_comparison_per_type (PDF/PNG)
   Caption: Variant comparison per cell type.
   Description: Grid with cell types on rows and variants on columns to inspect variant-specific shifts.
7. marker_comparison_compact (PDF/PNG)
   Caption: Compact pooled + per-type overview.
   Description: Two rows of subplots overlaying both variants per panel so you can read all cell types at once.
8. nuclei_overall_ap (PDF/PNG)
   Caption: Precision vs IoU for nuclei segmentation.
   Description: Same overall curve view but for nuclei benchmarks (plots from plot_nuclei_overall.py).
9. cell_vs_nuclei_side_by_side (PDF/PNG)
   Caption: Cell vs nuclei benchmarks in parallel.
   Description: Two synchronized panels allowing quick comparison of algorithm ordering across tasks.
10. metrics_boxplot_pooled (PDF/PNG)
    Caption: Pooled distribution summary.
    Description: Box plots for boundary F-score, precision, and recall across all images.
11. metrics_boundary_per_type (PDF/PNG)
    Caption: Boundary F-score split by cell type.
    Description: Box plots per marker plus pooled panel for the boundary F metric.
12. metrics_precision_per_type (PDF/PNG)
    Caption: Precision@0.50 split by cell type.
    Description: Shows how each algorithm’s precision varies by marker.
13. metrics_recall_per_type (PDF/PNG)
    Caption: Recall@0.50 split by cell type.
    Description: Companion figure for recall trends per marker.
14. metrics_boxplot_comprehensive (PDF/PNG)
    Caption: All metrics vs all cell types grid.
    Description: 3×5 matrix (pooled + four markers) covering boundary F-score, precision, and recall.
15. metrics_violin_pooled (PDF/PNG)
    Caption: Pooled distribution shapes.
    Description: Violin plots for boundary F-score, precision, and recall to highlight distribution skewness.
16. metrics_boxplot_comprehensive_nuclei (PDF/PNG)
    Caption: Nuclei-only pooled distributions.
    Description: Stack of boundary F-score, precision, and recall box plots derived from nuclei results (no per-type facets).
17. metrics_boxplot_comprehensive_marker (PDF/PNG)
    Caption: Marker-only pooled distributions.
    Description: Stack of boundary F-score, precision, and recall box plots for marker-only runs without per-type facets.

Nuclei Folder (plots_nov27/nuclei)
1. nuclei_overall_ap (PDF | PNG: png/nuclei_overall_ap.png)
   Caption: Precision vs IoU for nuclei (duplicate of root-level copy but stored with other nuclei assets).
   Description: Handy when sharing only nuclei deliverables.
2. nuclei_per_type (PDF/PNG)
   Caption: Per-marker nuclei curves.
   Description: Precision vs IoU per cell marker using nuclei masks.
3. nuclei_metric_boundary_fscore_pooled (PDF/PNG)
   Caption: Boundary F-score pooled distributions.
   Description: Box plot of nuclei boundary F-score across algorithms.
4. nuclei_metric_boundary_fscore_per_celltype (PDF/PNG)
   Caption: Boundary F-score by cell type.
   Description: Same metric but split by marker panels.
5. nuclei_metric_precision_iou_pooled (PDF/PNG)
   Caption: Precision@0.50 pooled distributions.
   Description: Box plot of nuclei precision for every algorithm.
6. nuclei_metric_precision_iou_per_celltype (PDF/PNG)
   Caption: Precision@0.50 by cell type.
   Description: Marker-specific precision panels for nuclei.
7. nuclei_metric_recall_iou_pooled (PDF/PNG)
   Caption: Recall@0.50 pooled distributions.
   Description: Box plot of nuclei recall for each algorithm.
8. nuclei_metric_recall_iou_per_celltype (PDF/PNG)
   Caption: Recall@0.50 by cell type.
   Description: Marker-specific recall view for nuclei segmentation.

Marker Folder (plots_nov27/marker)
1. marker_overall_ap (PDF/PNG)
   Caption: Precision vs IoU for marker-only segmentation.
   Description: Pooled AP curves for marker-only inputs.
2. marker_per_type (PDF/PNG)
   Caption: Marker-only per cell type.
   Description: AP curves filtered by OLIG2/NEUN/IBA1/GFAP for marker-only runs.
3. marker_metric_boundary_fscore_pooled (PDF/PNG)
   Caption: Boundary F-score pooled distributions (marker-only).
   Description: Box plot of algorithm boundary accuracy on marker-only masks.
4. marker_metric_boundary_fscore_per_celltype (PDF/PNG)
   Caption: Boundary F-score by cell type (marker-only).
   Description: Marker-level boundary comparisons.
5. marker_metric_precision_iou_pooled (PDF/PNG)
   Caption: Precision@0.50 pooled (marker-only).
   Description: Box plot of pooled precision for marker-only results.
6. marker_metric_precision_iou_per_celltype (PDF/PNG)
   Caption: Precision@0.50 by cell type (marker-only).
   Description: Marker-level precision breakdown.
7. marker_metric_recall_iou_pooled (PDF/PNG)
   Caption: Recall@0.50 pooled (marker-only).
   Description: Box plot of pooled recall for marker-only results.
8. marker_metric_recall_iou_per_celltype (PDF/PNG)
   Caption: Recall@0.50 by cell type (marker-only).
   Description: Marker-level recall breakdown.

Notes
- PNG copies mirror the PDF names inside each "png" subfolder so reviews can quickly load raster versions.
- Each PDF/PNG directory also includes a "no_legend" subfolder that stores the legend-free variants requested for publication layouts.
- Captions above match the in-code comments and should be reused verbatim in manuscripts or slides.
- Update the "Last recorded run time" after executing python plot_all_py.py with fresh evaluation outputs.
