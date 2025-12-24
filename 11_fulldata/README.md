# ROSMAP Immunofluorescence Image Analysis Pipeline

## Overview

This project analyzes immunofluorescence (IF) microscopy images from the ROSMAP (Religious Orders Study and Memory and Aging Project) study to quantify cell type-specific marker expression in brain tissue and correlate with Alzheimer's disease clinical outcomes.

## Data

### Raw Data
- **Location**: `/ihome/jbwang/liy121/ifimage/11_fulldata/raw/`
- **Markers**: 5 cell type markers
  | Marker | Cell Type | Files |
  |--------|-----------|-------|
  | GFAP | Astrocytes | ~6,144 |
  | iba1 | Microglia | ~6,126 |
  | NeuN | Neurons | ~8,596 |
  | Olig2 | Oligodendrocytes | ~5,954 |
  | PECAM | Endothelial cells | ~5,347 |
  
- **Total**: ~32,000 image pairs
- **Image format**: TIFF, 1040 x 1388 pixels, 3 channels (RGB)
- **File naming**:
  - Most markers: `*_b0c0x0-1388y0-1040.tiff` (nucleus), `*_b0c1x0-1388y0-1040.tiff` (marker)
  - NeuN: `*_dapi-b0c0x0-1388y0-1040.tiff` (nucleus), `*_NeuN-b0c1x0-1388y0-1040.tiff` (marker)

### Clinical Metadata
- **File**: `ROSMAP_clinical_n69.csv`
- **Donors**: 69
- **Key variables**:
  - `cogdx`: Cognitive diagnosis (1=NCI, 2=MCI, 3=MCI+, 4=AD, 5=AD+, 6=Other)
  - `braaksc`: Braak stage (0-6)
  - `ceradsc`: CERAD score (1=Definite, 2=Probable, 3=Possible, 4=No AD)
  - `cts_mmse30_lv`: MMSE score at last visit
  - `plaq_d`, `plaq_n`, `nft`, `gpath`: Pathology measures

---

## Pipeline

### Step 1: Data Extraction
```bash
cd /ihome/jbwang/liy121/ifimage/11_fulldata/raw
for f in *.zip; do unzip -o "$f" -d "${f%.zip}"; done
```

### Step 2: Cell Segmentation with Cellpose-SAM

**Script**: `batch_cellpose.py`

Uses Cellpose 3.0 (Cellpose-SAM) for automated cell segmentation:
- Segments nucleus channel (c0)
- Segments marker channel (c1)  
- Segments merged image (c0 + c1)

**Output**: `/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation/`
```
segmentation/
├── GFAP/[participant_id]/[sample]/
│   ├── mask_nuc.npy
│   ├── mask_marker.npy
│   └── mask_merged.npy
├── iba1/...
├── NeuN/...
├── Olig2/...
├── PECAM/...
├── file_mapping.csv
├── file_mapping_with_stats.csv
└── checkpoint.json
```

**Run**:
```bash
sbatch submit_cellpose.sh
```

### Step 3: Refinement with Marker Intensity

**Script**: `refine_segmentation.py` (or `refine_all.py`)

Refines segmentation using marker channel intensity:
1. Load merged mask (segmentation result)
2. Load marker channel intensity image
3. Calculate Otsu threshold on masked pixels
4. For each cell: if mean(marker intensity) ≥ Otsu threshold → marker positive
5. Calculate ratio = marker_positive_cells / total_cells

**Output**: `/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation_refined/`

**Run**:
```bash
sbatch submit_refine.sh
# or
python refine_all.py
```

### Step 4: Aggregation to Donor Level

**Script**: `aggregate_refined.py`

Aggregates image-level results to donor (projid) level:
- Mean, std, sum of cell counts per marker
- Mean marker positive ratio per donor
- Merges with clinical metadata

**Output**: `donor_level_aggregation_refined.csv`

**Run**:
```bash
python aggregate_refined.py
```

### Step 5: Clinical Association Analysis

**Script**: `clinical_analysis_refined.py`

Generates publication-quality figures (Genome Biology style):

| Figure | Description |
|--------|-------------|
| fig1 | Correlation heatmap (markers vs clinical variables) |
| fig2 | Scatter: Marker ratio vs Braak stage |
| fig3 | Scatter: Marker ratio vs CERAD score |
| fig4 | Scatter: Marker ratio vs MMSE |
| fig5 | Scatter: Marker ratio vs pathology measures |
| fig6 | Box plots by cognitive diagnosis (NCI/MCI/AD) |
| fig7 | Marker-marker correlation matrix |
| fig8 | Cell counts vs Braak stage |
| fig10 | Stacked bar plot (cell type composition per donor) |

**Output**: `/ihome/jbwang/liy121/ifimage/11_fulldata/analysis_refined/`

**Run**:
```bash
python clinical_analysis_refined.py
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `batch_cellpose.py` | Batch Cellpose-SAM segmentation |
| `redo_neun.py` | Re-segment NeuN (different file naming) |
| `refine_segmentation.py` | Refine masks with Otsu thresholding |
| `aggregate_refined.py` | Aggregate to donor level |
| `clinical_analysis_refined.py` | Generate figures and statistics |
| `test_refine.py` | Test refinement on 5 images |
| `merge_and_aggregate.py` | Merge NeuN results into main data |

## SLURM Submission Scripts

| Script | Cluster | Time | Resources |
|--------|---------|------|-----------|
| `submit_cellpose.sh` | gpu/a100 | 24h | 1 GPU |
| `submit_neun.sh` | gpu/a100 | 24h | 1 GPU |
| `submit_refine.sh` | htc | 8h | 64G RAM |

---

## Output Summary

### Final Data Files
1. `segmentation/file_mapping_with_stats.csv` - Image-level segmentation stats
2. `segmentation/donor_level_aggregation.csv` - Donor-level (original ratio)
3. `segmentation_refined/file_mapping_refined.csv` - Image-level refined stats
4. `segmentation_refined/donor_level_aggregation_refined.csv` - Donor-level (refined ratio)

### Key Columns in Final Dataset
- `{marker}_n_images` - Number of images per donor
- `{marker}_n_total_cells_sum` - Total segmented cells
- `{marker}_n_marker_positive_sum` - Marker positive cells (refined)
- `{marker}_marker_positive_ratio_refined_mean` - Mean ratio across images
- `{marker}_ratio_from_sum` - Ratio from summed counts

---

## Methods Summary

### Segmentation
- **Algorithm**: Cellpose-SAM (Cellpose 3.0)
- **Model**: CellposeModel with GPU acceleration
- **Parameters**: flow_threshold=0.4, cellprob_threshold=0.0, batch_size=32

### Marker Positive Classification
- **Method**: Otsu thresholding on marker channel intensity
- **Threshold**: Calculated per image on all segmented cell pixels
- **Criterion**: Cell is marker+ if mean intensity ≥ Otsu threshold
- **Filter**: Minimum cell area = 100 pixels

### Statistical Analysis
- **Correlation**: Spearman's ρ (ordinal clinical variables)
- **Group comparison**: Kruskal-Wallis test (NCI vs MCI vs AD)
- **Significance**: *p<0.05, **p<0.01, ***p<0.001

---

## Dependencies

```
cellpose>=3.0
numpy
pandas
tifffile
scikit-image
scipy
matplotlib
seaborn
tqdm
```