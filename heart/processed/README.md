# Heart Dataset - Processed Data

## Channel Information

- **Channel 0**: DAPI (nuclei)
- **Channel 1**: ALDH1A2 (epicardial cell)
- **Channel 2**: WGA (cell membrane)
- **Channel 3**: CD45 (immune cell)
- **Channel 4**: PDGFRB (mural cells)

## File Naming Convention

Files are named as: `{area}_{channel}.tif`

Examples:
- `LA1_dapi.tif` - Left Atrium area 1, DAPI channel
- `LA1_cd45.tif` - Left Atrium area 1, CD45 channel
- `RV2_aldh1a2.tif` - Right Ventricle area 2, ALDH1A2 channel

## File Structure

```
processed/
├── LA/
│   ├── LA1_dapi.tif
│   ├── LA1_aldh1a2.tif
│   ├── LA1_wga.tif
│   ├── LA1_cd45.tif
│   ├── LA1_pdgfrb.tif
│   └── ...
├── RA/, LV/, RV/, SEP/
├── data_info.csv            # All extracted channel files
├── channel_statistics.csv   # Channel quality statistics
├── complete_mapping.csv     # Links to ground truth masks
└── README.md
```

## Data Files

- **data_info.csv**: 14 images with all channel paths
- **complete_mapping.csv**: 42 entries linking channels to GT masks
- **channel_statistics.csv**: Quality metrics for each channel in each image

## Statistics

- Total images: 14
- Total channel files: 70
- Regions: LA, RA, LV, RV, SEP
- Cell types with GT: Epi, Immune, Mural
