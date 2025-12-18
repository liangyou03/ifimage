"""
Redo NeuN segmentation only
Loose matching, overwrite existing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from cellpose import models, core, io
from tifffile import imread
from tqdm import tqdm
import json
import time
from datetime import datetime, timedelta

io.logger_setup()

# Check GPU
if not core.use_gpu():
    raise RuntimeError("No GPU access!")
print("✓ GPU available!")

# Paths
base_path = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/raw/NeuN')
out_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation/NeuN')
out_base.mkdir(parents=True, exist_ok=True)

error_log = out_base / 'errors_neun.log'

def log_error(msg):
    with open(error_log, 'a') as f:
        f.write(f"{datetime.now()} - {msg}\n")

# Load Cellpose-SAM model
print("Loading Cellpose-SAM model...")
model = models.CellposeModel(gpu=True)
print("✓ Model loaded!")

# Parameters
flow_threshold = 0.4
cellprob_threshold = 0.0

# Find ALL -1040.tiff files loosely
all_tiffs = list(base_path.glob('**/*-1040.tiff'))
print(f"Found {len(all_tiffs)} total tiff files")

# Group by parent folder (each folder should have paired images)
from collections import defaultdict
folders = defaultdict(list)
for f in all_tiffs:
    folders[f.parent].append(f)

print(f"Found {len(folders)} image folders")

# Process each folder
file_mapping = []
start_time = time.time()
processed = 0
errors = 0

pbar = tqdm(folders.items(), desc="NeuN Segmenting")

for folder, files in pbar:
    try:
        # Find nuc (dapi/c0) and marker (NeuN/c1) files
        nuc_file = None
        marker_file = None
        
        for f in files:
            fname = f.name.lower()
            if 'dapi' in fname or ('c0' in fname and 'c1' not in fname):
                nuc_file = f
            elif 'neun' in fname or 'c1' in fname:
                marker_file = f
        
        if nuc_file is None or marker_file is None:
            log_error(f"Cannot find pair in {folder}: {[f.name for f in files]}")
            errors += 1
            continue
        
        # Extract participant ID
        parts = folder.parts
        participant_id = None
        for p in parts:
            if p.isdigit() and len(p) >= 6:
                participant_id = p
                break
        if participant_id is None:
            participant_id = 'unknown'
        
        sample_name = folder.name.replace('.tiff_files', '')
        
        # Output directory
        out_dir = out_base / participant_id / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Read images
        img_nuc = imread(nuc_file)
        img_marker = imread(marker_file)
        
        # Segment nucleus
        mask_nuc, _, _ = model.eval(img_nuc, batch_size=32,
                                    flow_threshold=flow_threshold,
                                    cellprob_threshold=cellprob_threshold)
        
        # Segment marker
        mask_marker, _, _ = model.eval(img_marker, batch_size=32,
                                       flow_threshold=flow_threshold,
                                       cellprob_threshold=cellprob_threshold)
        
        # Segment merged
        img_merged = np.stack([img_nuc[:,:,0], img_marker[:,:,0],
                               np.zeros_like(img_nuc[:,:,0])], axis=-1)
        mask_merged, _, _ = model.eval(img_merged, batch_size=32,
                                       flow_threshold=flow_threshold,
                                       cellprob_threshold=cellprob_threshold)
        
        # Save (overwrite)
        np.save(out_dir / 'mask_nuc.npy', mask_nuc)
        np.save(out_dir / 'mask_marker.npy', mask_marker)
        np.save(out_dir / 'mask_merged.npy', mask_merged)
        
        # Track mapping
        file_mapping.append({
            'marker': 'NeuN',
            'participant_id': participant_id,
            'sample_name': sample_name,
            'c0_path': str(nuc_file),
            'c1_path': str(marker_file),
            'c1_exists': True,
            'out_dir': str(out_dir),
            'nuc_cell_count': int(mask_nuc.max()),
            'marker_cell_count': int(mask_marker.max()),
            'merged_cell_count': int(mask_merged.max()),
            'mask_exists': True
        })
        
        processed += 1
        pbar.set_postfix({'done': processed, 'err': errors})
        
    except Exception as e:
        log_error(f"{folder}: {e}")
        errors += 1
        continue

# Save mapping
df = pd.DataFrame(file_mapping)
df.to_csv(out_base / 'file_mapping_neun.csv', index=False)

# Summary
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"✓ NeuN Completed!")
print(f"  Processed: {processed}")
print(f"  Errors: {errors}")
print(f"  Time: {timedelta(seconds=int(elapsed))}")
print(f"  Mapping saved: {out_base / 'file_mapping_neun.csv'}")
print(f"{'='*60}")