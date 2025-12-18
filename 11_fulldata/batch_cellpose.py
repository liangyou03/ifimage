"""
Batch Cellpose-SAM segmentation for all markers
GFAP, iba1, NeuN, Olig2, PECAM

Features:
- Checkpoint/resume support
- Real-time progress tracking
- Error logging
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
base_path = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/raw')
out_base = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation')
out_base.mkdir(parents=True, exist_ok=True)

# Checkpoint file
checkpoint_file = out_base / 'checkpoint.json'
error_log = out_base / 'errors.log'

# All markers
markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# Load checkpoint
def load_checkpoint():
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return set(json.load(f)['completed'])
    return set()

def save_checkpoint(completed):
    with open(checkpoint_file, 'w') as f:
        json.dump({'completed': list(completed), 'last_update': str(datetime.now())}, f)

def log_error(msg):
    with open(error_log, 'a') as f:
        f.write(f"{datetime.now()} - {msg}\n")

# Load completed files
completed = load_checkpoint()
print(f"✓ Checkpoint loaded: {len(completed)} files already processed")

# Load Cellpose-SAM model
print("Loading Cellpose-SAM model...")
model = models.CellposeModel(gpu=True)
print("✓ Model loaded!")

# Parameters
flow_threshold = 0.4
cellprob_threshold = 0.0

# Gather all files first - loose search like R code
all_files = []
file_mapping = []  # Track file relationships

for marker in markers:
    marker_path = base_path / marker
    # 递归查找所有 *c0*-1040.tiff，不限制目录结构
    c0_files = list(marker_path.glob('**/*c0*-1040.tiff'))
    for c0 in c0_files:
        c1 = Path(str(c0).replace('c0', 'c1'))
        
        # Extract participant ID
        parts = c0.parts
        participant_id = None
        for p in parts:
            if p.isdigit() and len(p) >= 6:
                participant_id = p
                break
        if participant_id is None:
            participant_id = 'unknown'
        
        sample_name = c0.parent.name.replace('.tiff_files', '')
        
        all_files.append((marker, c0))
        file_mapping.append({
            'marker': marker,
            'participant_id': participant_id,
            'sample_name': sample_name,
            'c0_path': str(c0),
            'c1_path': str(c1),
            'c1_exists': c1.exists(),
            'out_dir': str(out_base / marker / participant_id / sample_name)
        })
    
    print(f"  {marker}: {len([f for m,f in all_files if m==marker])} files")

# Save file mapping
import pandas as pd
df_mapping = pd.DataFrame(file_mapping)
mapping_file = out_base / 'file_mapping.csv'
df_mapping.to_csv(mapping_file, index=False)
print(f"\n✓ File mapping saved: {mapping_file}")
print(f"  Total pairs: {len(df_mapping)}")
print(f"  Missing c1: {(~df_mapping['c1_exists']).sum()}")

total_files = len(all_files)
remaining = [(m, f) for m, f in all_files if str(f) not in completed]

print(f"\n{'='*60}")
print(f"Total files: {total_files}")
print(f"Already done: {len(completed)}")
print(f"Remaining: {len(remaining)}")
print(f"{'='*60}\n")

# Process
start_time = time.time()
processed = 0
errors = 0

pbar = tqdm(remaining, desc="Segmenting", 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

for marker, c0_file in pbar:
    try:
        # Find paired c1 file
        c1_file = Path(str(c0_file).replace('c0', 'c1'))
        
        if not c1_file.exists():
            log_error(f"No c1 pair: {c0_file}")
            errors += 1
            continue
        
        # Extract participant ID and sample name (flexible extraction)
        # Look for numeric participant ID in path
        parts = c0_file.parts
        participant_id = None
        for p in parts:
            if p.isdigit() and len(p) >= 6:  # participant IDs are long numbers
                participant_id = p
                break
        if participant_id is None:
            participant_id = 'unknown'
        
        sample_name = c0_file.parent.name.replace('.tiff_files', '')
        
        # Output directory
        out_dir = out_base / marker / participant_id / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Read images
        img_nuc = imread(c0_file)
        img_marker = imread(c1_file)
        
        # Segment nucleus (c0)
        mask_nuc, _, _ = model.eval(img_nuc, batch_size=32, 
                                    flow_threshold=flow_threshold,
                                    cellprob_threshold=cellprob_threshold)
        
        # Segment marker (c1)
        mask_marker, _, _ = model.eval(img_marker, batch_size=32,
                                       flow_threshold=flow_threshold,
                                       cellprob_threshold=cellprob_threshold)
        
        # Segment merged
        img_merged = np.stack([img_nuc[:,:,0], img_marker[:,:,0], 
                               np.zeros_like(img_nuc[:,:,0])], axis=-1)
        mask_merged, _, _ = model.eval(img_merged, batch_size=32,
                                       flow_threshold=flow_threshold,
                                       cellprob_threshold=cellprob_threshold)
        
        # Save
        np.save(out_dir / 'mask_nuc.npy', mask_nuc)
        np.save(out_dir / 'mask_marker.npy', mask_marker)
        np.save(out_dir / 'mask_merged.npy', mask_merged)
        
        # Update checkpoint
        completed.add(str(c0_file))
        processed += 1
        
        # Save checkpoint every 50 files
        if processed % 50 == 0:
            save_checkpoint(completed)
        
        # Update progress bar
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - processed) / rate if rate > 0 else 0
        pbar.set_postfix({
            'marker': marker,
            'done': f"{len(completed)}/{total_files}",
            'err': errors
        })
        
    except Exception as e:
        log_error(f"{c0_file}: {e}")
        errors += 1
        continue

# Final checkpoint save
save_checkpoint(completed)

# Summary
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"✓ Completed!")
print(f"  Processed: {processed}")
print(f"  Errors: {errors}")
print(f"  Time: {timedelta(seconds=int(elapsed))}")
print(f"  Output: {out_base}")
print(f"{'='*60}")