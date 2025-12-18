"""
Quick test: Cellpose-SAM segmentation on a few sample images
"""

import numpy as np
from pathlib import Path
from cellpose import models, core, io
from tifffile import imread

io.logger_setup()

# Check GPU
if not core.use_gpu():
    raise RuntimeError("No GPU access!")
print("GPU available!")

# Paths
base_path = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/raw')
out_path = Path('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation_test')
out_path.mkdir(parents=True, exist_ok=True)

# Find a few test images (2 pairs from GFAP)
test_files = list(base_path.glob('GFAP/GFAP/*/Grey/*_files/*c0*-1040.tiff'))[:2]

print(f"Found {len(test_files)} test images")
for f in test_files:
    print(f"  {f}")

# Load Cellpose-SAM model
print("\nLoading Cellpose-SAM model...")
model = models.CellposeModel(gpu=True)

# Parameters
flow_threshold = 0.4
cellprob_threshold = 0.0

for c0_file in test_files:
    # Find paired c1 file
    c1_file = Path(str(c0_file).replace('c0', 'c1'))
    
    if not c1_file.exists():
        print(f"Skipping {c0_file.name}, no c1 pair")
        continue
    
    print(f"\nProcessing: {c0_file.parent.name}")
    
    # Read images
    img_nuc = imread(c0_file)
    img_marker = imread(c1_file)
    
    print(f"  Image shapes: nuc={img_nuc.shape}, marker={img_marker.shape}")
    
    # Segment nucleus (c0)
    print("  Segmenting nucleus (c0)...")
    mask_nuc, flows, styles = model.eval(img_nuc, batch_size=32, 
                                          flow_threshold=flow_threshold,
                                          cellprob_threshold=cellprob_threshold)
    
    # Segment marker (c1)
    print("  Segmenting marker (c1)...")
    mask_marker, flows, styles = model.eval(img_marker, batch_size=32,
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=cellprob_threshold)
    
    # Segment merged (combine into single RGB: nuc=red, marker=green)
    print("  Segmenting merged...")
    # Take first channel from each (they're RGB but likely grayscale)
    img_merged = np.stack([img_nuc[:,:,0], img_marker[:,:,0], np.zeros_like(img_nuc[:,:,0])], axis=-1)
    mask_merged, flows, styles = model.eval(img_merged, batch_size=32,
                                             flow_threshold=flow_threshold,
                                             cellprob_threshold=cellprob_threshold)
    
    # Save as npy
    sample_name = c0_file.parent.name.replace('.tiff_files', '')
    np.save(out_path / f'{sample_name}_mask_nuc.npy', mask_nuc)
    np.save(out_path / f'{sample_name}_mask_marker.npy', mask_marker)
    np.save(out_path / f'{sample_name}_mask_merged.npy', mask_merged)
    
    print(f"  Saved! Cells found: nuc={mask_nuc.max()}, marker={mask_marker.max()}, merged={mask_merged.max()}")

print("\nDone! Check outputs in:", out_path)