```markdown
# Heart Tissue Nuclei Segmentation Benchmark

## Project Overview

This project benchmarks multiple nuclei and cell segmentation algorithms on immunofluorescence microscopy images of human heart tissue from the ROSMAP study. The dataset contains 5-channel images with manual annotations of three cell types (Epicardial, Immune, and Mural cells) across five cardiac regions.

## Dataset Information

### Tissue Regions
- **LA** - Left Atrium
- **RA** - Right Atrium  
- **LV** - Left Ventricle
- **RV** - Right Ventricle
- **SEP** - Septum

### Image Channels
Multi-page TIFF files with 5 channels:
- **Channel 0**: DAPI (nuclei staining)
- **Channel 1**: ALDH1A2 (epicardial cell marker)
- **Channel 2**: WGA (cell membrane marker)
- **Channel 3**: CD45 (immune cell marker)
- **Channel 4**: PDGFRB (mural cell marker - pericytes and smooth muscle cells)

### Cell Type Annotations
Manual annotations for three cell types:
- **Epi** - Epicardial cells
- **Immune** - Immune cells  
- **Mural** - Mural cells (pericytes and smooth muscle)

Each region contains 2-3 areas, with 3 vertical positions per area, annotated for all three cell types using ImageJ ROI format.

## Directory Structure

```
heart/
├── raw/                          # Original multi-channel TIFF images
│   ├── LA/
│   │   ├── LA1.tif              # 5-channel TIFF (5, H, W)
│   │   ├── Epi-LA1.zip          # ImageJ ROI annotations
│   │   ├── Immune-LA1.zip
│   │   ├── Mural-LA1.zip
│   │   └── ...
│   ├── RA/, LV/, RV/, SEP/
│   └── readme.txt
│
├── ground_truth_masks/           # Converted GT instance masks
│   ├── LA/
│   │   ├── Epi-LA1_mask.npy     # Instance mask (uint16)
│   │   ├── Immune-LA1_mask.npy
│   │   └── Mural-LA1_mask.npy
│   ├── RA/, LV/, RV/, SEP/
│   ├── file_mapping.csv         # Links images → ROIs → masks
│   ├── file_mapping.json
│   └── annotation_stats.csv     # GT statistics
│
├── processed/                    # Extracted single-channel TIFFs
│   ├── LA/
│   │   ├── LA1_dapi.tif         # Single channel (H, W)
│   │   ├── LA1_aldh1a2.tif
│   │   ├── LA1_wga.tif
│   │   ├── LA1_cd45.tif
│   │   ├── LA1_pdgfrb.tif
│   │   └── ...
│   ├── RA/, LV/, RV/, SEP/
│   ├── data_info.csv            # All channel file paths
│   ├── channel_statistics.csv   # Channel quality metrics
│   ├── complete_mapping.csv     # Links channels → GT masks
│   └── README.md
│
├── benchmark_results/            # Algorithm predictions and evaluation
│   ├── cellpose_predictions/
│   │   ├── LA/, RA/, LV/, RV/, SEP/
│   │   └── ...
│   ├── stardist_predictions/
│   ├── omnipose_predictions/
│   ├── watershed_predictions/
│   ├── mesmer_predictions/
│   ├── lacss_predictions/
│   ├── microsam_predictions/
│   ├── cellsam_predictions/
│   ├── cellpose_predictions.csv
│   ├── stardist_predictions.csv
│   ├── ...
│   ├── heart_evaluation_all.csv
│   └── gt_channel_summary.csv
│
├── config_heart.py              # Configuration file
├── gt_generation.py             # Convert ROI → numpy masks
├── prepare_data.py              # Extract 5 channels → single TIFFs
├── find_gt_channel.py           # Identify GT-marker correspondence
│
├── run_cellpose.py              # Cellpose algorithm
├── run_stardist.py              # StarDist algorithm
├── run_omnipose.py              # Omnipose algorithm
├── run_watershed.py             # Watershed algorithm
├── run_mesmer.py                # Mesmer/Deepcell algorithm
├── run_lacss.py                 # LACSS algorithm
├── run_microsam.py              # MicroSAM algorithm
├── run_cellsam.py               # CellSAM algorithm
│
├── evaluate_all.py              # Evaluation script
├── run_all_benchmarks.sh        # Master pipeline script
└── README.md                    # This file
```

## Pipeline Workflow

### 1. Ground Truth Generation

Convert ImageJ ROI annotations to instance segmentation masks:

```bash
conda activate cellpose
python gt_generation.py
```

**Output**: `ground_truth_masks/` with numpy arrays where each nucleus has a unique ID

**Statistics**:
- Total masks: 42
- Total annotated nuclei: 1,055
  - Epi: 261 nuclei (mean: 18.6 ± 9.3 per image)
  - Immune: 391 nuclei (mean: 27.9 ± 20.9 per image)
  - Mural: 403 nuclei (mean: 28.8 ± 14.4 per image)

### 2. Channel Extraction

Extract 5-channel TIFF into single-channel files:

```bash
conda activate ifimage
python prepare_data.py
```

**Output**: `processed/` with separate TIFF files for each channel
- 14 images × 5 channels = 70 single-channel files
- File naming: `{area}_{channel}.tif` (e.g., `LA1_dapi.tif`)

### 3. GT-Marker Correspondence Analysis

Identify which marker channel corresponds to each cell type annotation:

```bash
conda activate ifimage
python find_gt_channel.py
```

**Analysis metrics**:
- Overlap ratio: Signal intensity within GT regions
- Coverage: GT coverage of high-signal areas
- Correlation: GT mask vs channel image correlation
- Dice coefficient: Overlap similarity

**Expected results**:
- Epi cells → ALDH1A2 channel
- Immune cells → CD45 channel
- Mural cells → PDGFRB channel

### 4. Algorithm Benchmarking

Run multiple segmentation algorithms on DAPI channel for nuclei segmentation:

```bash
# Run all algorithms
./run_all_benchmarks.sh

# Or run individually with specific environments
conda activate cellpose && python run_cellpose.py
conda activate ifimage_stardist && python run_stardist.py
conda activate cellpose && python run_omnipose.py
conda activate ifimage && python run_watershed.py
conda activate deepcell_retinamask && python run_mesmer.py
conda activate lacss && python run_lacss.py
conda activate microsam-cuda && python run_microsam.py
conda activate ifimage_cellsam && python run_cellsam.py
```

**Algorithms tested**:
1. **Cellpose** - Deep learning, flow-based
2. **StarDist** - Star-convex polygons
3. **Omnipose** - Improved Cellpose for diverse shapes
4. **Watershed** - Classical segmentation
5. **Mesmer** - Deep learning for multiplexed images
6. **LACSS** - Learning-based cell segmentation
7. **MicroSAM** - Segment Anything for microscopy
8. **CellSAM** - SAM-based cell segmentation

### 5. Evaluation

Evaluate all predictions against ground truth:

```bash
conda activate ifimage_evaluation
python evaluate_all.py
```

**Evaluation metrics**:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **Average IoU**: Mean IoU of matched instances
- **True Positives (TP)**: Correctly detected nuclei (IoU ≥ 0.5)
- **False Positives (FP)**: Incorrectly detected nuclei
- **False Negatives (FN)**: Missed nuclei

**Output**: `benchmark_results/heart_evaluation_all.csv` with comprehensive metrics

## Conda Environments

| Algorithm | Environment | Purpose |
|-----------|-------------|---------|
| Cellpose | `cellpose` | Cellpose & Omnipose |
| StarDist | `ifimage_stardist` | StarDist |
| Watershed | `ifimage` | Classical methods |
| Mesmer | `deepcell_retinamask` | Deepcell |
| LACSS | `lacss` | LACSS |
| MicroSAM | `microsam-cuda` | MicroSAM |
| CellSAM | `ifimage_cellsam` | CellSAM |
| Evaluation | `ifimage_evaluation` | Metrics calculation |

## Key Files

### Configuration
- `config_heart.py` - Central configuration (paths, regions, cell types)

### Data Processing
- `gt_generation.py` - ROI → numpy mask conversion
- `prepare_data.py` - Multi-channel TIFF → single-channel extraction
- `find_gt_channel.py` - GT-marker correspondence analysis

### Algorithms (8 scripts)
- `run_cellpose.py`, `run_stardist.py`, `run_omnipose.py`, etc.
- Each script is independent and uses its own conda environment

### Evaluation
- `evaluate_all.py` - Unified evaluation using Hungarian matching algorithm

### Automation
- `run_all_benchmarks.sh` - Run entire pipeline

## Data Files

### Core Mapping Files
1. **ground_truth_masks/file_mapping.csv**
   - Links: original image → ROI zip → GT mask
   - Columns: region, area, cell_type, image paths, GT paths

2. **processed/data_info.csv**
   - All extracted channel file paths
   - Columns: region, area, channel paths

3. **processed/complete_mapping.csv**
   - Complete mapping: channels → GT masks
   - Columns: region, area, cell_type, all channel paths, GT path

4. **benchmark_results/heart_evaluation_all.csv**
   - All algorithms' evaluation results
   - Columns: algorithm, region, area, cell_type, metrics

## Usage Examples

### Quick Start - Full Pipeline
```bash
cd /ihome/jbwang/liy121/ifimage/heart
chmod +x run_all_benchmarks.sh
./run_all_benchmarks.sh
```

### Run Specific Steps
```bash
# Step 1: Generate GT masks
conda activate cellpose
python gt_generation.py

# Step 2: Extract channels
python prepare_data.py

# Step 3: Analyze GT correspondence
python find_gt_channel.py

# Step 4: Run one algorithm
conda activate cellpose
python run_cellpose.py

# Step 5: Evaluate
conda activate ifimage_evaluation
python evaluate_all.py
```

### Load Results in Python
```python
import pandas as pd
from pathlib import Path

# Load GT mapping
gt_mapping = pd.read_csv('ground_truth_masks/file_mapping.csv')

# Load processed data mapping
data_mapping = pd.read_csv('processed/complete_mapping.csv')

# Load evaluation results
results = pd.read_csv('benchmark_results/heart_evaluation_all.csv')

# Compare algorithms
algo_summary = results.groupby('algorithm')[
    ['precision', 'recall', 'f1_score', 'avg_iou']
].mean()
print(algo_summary)
```

## Dataset Statistics

- **Total regions**: 5 (LA, RA, LV, RV, SEP)
- **Total images**: 14 unique areas
- **Total annotations**: 42 (14 areas × 3 cell types)
- **Total annotated nuclei**: 1,055
- **Image dimensions**: Variable (328-1312 pixels height, 840-2508 pixels width)
- **Bit depth**: 16-bit unsigned integer
- **File format**: Multi-page TIFF (5 channels)

## Notes

1. **File Format**: Original images are multi-page TIFF files. Use `tifffile` library for proper reading.
2. **GT Format**: Ground truth masks are numpy arrays (uint16) with instance IDs (0=background, 1-N=nuclei).
3. **Evaluation**: Hungarian matching algorithm with IoU threshold of 0.5 for TP/FP/FN calculation.
4. **Missing Data**: RV1 Epi annotation is missing (file not found).
5. **Coordinate System**: ImageJ ROI coordinates are preserved in the conversion process.

## References

- **Cellpose**: Stringer et al., Nature Methods 2021
- **StarDist**: Schmidt et al., MICCAI 2018
- **Omnipose**: Cutler et al., Nature Methods 2022
- **Mesmer**: Greenwald et al., Nature Biotechnology 2022

## Citation

If you use this dataset or pipeline, please cite the original ROSMAP study and relevant algorithm papers.

## Contact

For questions or issues, contact the project maintainer.
