#!/usr/bin/env python3
"""
Script to process scribbles dataset and organize it for ifimage_tools.IfImageDataset

This script:
1. Extracts DAPI (b0c0) and marker (b0c1) channels from TIFF files
2. Converts ROI ZIP files to NPY format 
3. Organizes files with correct naming for IfImageDataset.load_data()

Expected input structure: scribbles/{CELLTYPE}_Scribbled/{CELLTYPE}_{ID}/files
Expected output structure: 
- images/{celltype}_{id}.tiff (DAPI channel)
- images/{celltype}_{id}_marker.tiff (marker channel) 
- masks/{celltype}_{id}_cellbodies.npy (converted from RoiSet_CellBodies_Final.zip)
- masks/{celltype}_{id}_dapimultimask.npy (if DAPI masks exist)
"""

import os
import re
import shutil
import numpy as np
from pathlib import Path
from preprocessing import rois_to_mask
import skimage.io as skio
from tqdm import tqdm

class ScribblesProcessor:
    def __init__(self, scribbles_dir="scribbles", output_dir="processed_dataset", 
                 width=1388, height=1040):
        """
        Initialize the processor
        
        Args:
            scribbles_dir: Path to scribbles directory
            output_dir: Path to output directory
            width: Image width for mask conversion
            height: Image height for mask conversion
        """
        self.scribbles_dir = Path(scribbles_dir)
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.processed_samples = 0
        self.errors = []
        
    def extract_sample_info(self, sample_dir_name):
        """Extract celltype and sample_id from directory name like 'GFAP_3527'"""
        match = re.match(r"^([A-Za-z0-9]+)_(\d+)$", sample_dir_name)
        if match:
            celltype = match.group(1).lower()
            sample_id = match.group(2)
            return celltype, sample_id
        return None, None
    
    def process_tiff_files(self, sample_path, celltype, sample_id):
        """Process TIFF files to extract DAPI and marker channels"""
        dapi_file = None
        marker_file = None
        
        # Find b0c0 (DAPI) and b0c1 (marker) files
        for file_path in sample_path.glob("*.tiff"):
            filename = file_path.name
            if "b0c0" in filename:
                dapi_file = file_path
            elif "b0c1" in filename:
                marker_file = file_path
        
        # Copy files with correct naming
        if dapi_file:
            dapi_output = self.images_dir / f"{celltype}_{sample_id}.tiff"
            shutil.copy2(dapi_file, dapi_output)
            print(f"  ✓ DAPI: {dapi_file.name} → {dapi_output.name}")
        else:
            self.errors.append(f"No DAPI file (b0c0) found in {sample_path}")
            
        if marker_file:
            marker_output = self.images_dir / f"{celltype}_{sample_id}_marker.tiff"
            shutil.copy2(marker_file, marker_output)
            print(f"  ✓ Marker: {marker_file.name} → {marker_output.name}")
        else:
            self.errors.append(f"No marker file (b0c1) found in {sample_path}")
            
        return dapi_file is not None, marker_file is not None
    
    def process_roi_files(self, sample_path, celltype, sample_id):
        """Process ROI ZIP files and convert to NPY masks"""
        roi_files_processed = 0
        
        for roi_file in sample_path.glob("*.zip"):
            filename = roi_file.name
            
            try:
                # Convert ROI to mask
                mask = rois_to_mask(str(roi_file), self.width, self.height)
                
                # Determine output filename based on ROI file name
                if "CellBodies" in filename:
                    output_name = f"{celltype}_{sample_id}_cellbodies.npy"
                elif "DAPI" in filename:
                    output_name = f"{celltype}_{sample_id}_dapimultimask.npy"
                elif "ALLDAPI" in filename:
                    output_name = f"{celltype}_{sample_id}_dapimultimask.npy"
                else:
                    # Generic naming for other ROI files
                    base_name = roi_file.stem
                    output_name = f"{celltype}_{sample_id}_{base_name.lower()}.npy"
                
                output_path = self.masks_dir / output_name
                np.save(output_path, mask)
                
                print(f"  ✓ ROI: {filename} → {output_name} (max_label: {mask.max()})")
                roi_files_processed += 1
                
            except Exception as e:
                error_msg = f"Failed to process ROI {roi_file}: {str(e)}"
                self.errors.append(error_msg)
                print(f"  ✗ Error: {error_msg}")
        
        return roi_files_processed
    
    def process_sample(self, sample_path):
        """Process a single sample directory"""
        sample_dir_name = sample_path.name
        celltype, sample_id = self.extract_sample_info(sample_dir_name)
        
        if not celltype or not sample_id:
            error_msg = f"Could not parse celltype and sample_id from {sample_dir_name}"
            self.errors.append(error_msg)
            print(f"  ✗ {error_msg}")
            return False
        
        print(f"Processing {celltype}_{sample_id}...")
        
        # Process TIFF files
        has_dapi, has_marker = self.process_tiff_files(sample_path, celltype, sample_id)
        
        # Process ROI files
        roi_count = self.process_roi_files(sample_path, celltype, sample_id)
        
        if has_dapi and has_marker and roi_count > 0:
            self.processed_samples += 1
            return True
        else:
            missing = []
            if not has_dapi:
                missing.append("DAPI")
            if not has_marker:
                missing.append("marker")
            if roi_count == 0:
                missing.append("ROI masks")
            
            error_msg = f"Sample {celltype}_{sample_id} missing: {', '.join(missing)}"
            self.errors.append(error_msg)
            print(f"  ⚠ {error_msg}")
            return False
    
    def process_all(self):
        """Process all samples in the scribbles directory"""
        print(f"Processing scribbles from: {self.scribbles_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        # Find all celltype directories
        celltype_dirs = [d for d in self.scribbles_dir.iterdir() 
                        if d.is_dir() and d.name.endswith("_Scribbled")]
        
        total_samples = 0
        for celltype_dir in celltype_dirs:
            sample_dirs = [d for d in celltype_dir.iterdir() if d.is_dir()]
            total_samples += len(sample_dirs)
        
        print(f"Found {len(celltype_dirs)} cell types with {total_samples} total samples")
        print("-" * 60)
        
        # Process each celltype directory
        for celltype_dir in tqdm(celltype_dirs, desc="Processing cell types"):
            print(f"\nProcessing {celltype_dir.name}...")
            
            # Process each sample in this celltype
            sample_dirs = [d for d in celltype_dir.iterdir() if d.is_dir()]
            for sample_dir in sample_dirs:
                self.process_sample(sample_dir)
        
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Successfully processed: {self.processed_samples} samples")
        print(f"Errors encountered: {len(self.errors)}")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"\nOutput structure:")
        print(f"  Images: {self.images_dir}")
        print(f"  Masks: {self.masks_dir}")
        
        # Count output files
        image_files = len(list(self.images_dir.glob("*.tiff")))
        mask_files = len(list(self.masks_dir.glob("*.npy")))
        
        print(f"\nGenerated files:")
        print(f"  Image files: {image_files}")
        print(f"  Mask files: {mask_files}")

def main():
    """Main function to run the processing"""
    processor = ScribblesProcessor(
        scribbles_dir="scribbles",
        output_dir="processed_dataset",
        width=1388,
        height=1040
    )
    
    processor.process_all()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("To load the processed dataset, use:")
    print("```python")
    print("from ifimage_tools import IfImageDataset")
    print("dataset = IfImageDataset(")
    print("    image_dir='processed_dataset/images',")
    print("    nuclei_masks_dir='processed_dataset/masks',")
    print("    cell_masks_dir='processed_dataset/masks'")
    print(")")
    print("dataset.load_data()")
    print("```")

if __name__ == "__main__":
    main()
