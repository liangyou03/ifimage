#!/usr/bin/env python3
"""
Direct Scribbles Review Panel

This script directly reviews the scribbles directory structure to verify
that DAPI channels, marker channels, and ROI masks are correctly matched
before processing them into the final dataset.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
from roifile import ImagejRoi
from skimage.draw import polygon
import skimage.io as skio
import warnings
warnings.filterwarnings('ignore')

class ScribblesReviewPanel:
    def __init__(self, scribbles_dir="scribbles", width=1388, height=1040):
        """
        Initialize the review panel for scribbles directory
        
        Args:
            scribbles_dir: Path to scribbles directory
            width: Image width for ROI conversion
            height: Image height for ROI conversion
        """
        self.scribbles_dir = Path(scribbles_dir)
        self.width = width
        self.height = height
        
        # Scan and organize samples
        self.samples = self._scan_samples()
        print(f"Found {len(self.samples)} samples across {len(self._get_celltypes())} cell types")
    
    def _scan_samples(self):
        """Scan scribbles directory and organize sample information"""
        samples = {}
        
        # Find all celltype directories
        celltype_dirs = [d for d in self.scribbles_dir.iterdir() 
                        if d.is_dir() and d.name.endswith("_Scribbled")]
        
        for celltype_dir in celltype_dirs:
            # Extract celltype from directory name
            celltype = celltype_dir.name.replace("_Scribbled", "").lower()
            
            # Find all sample directories
            sample_dirs = [d for d in celltype_dir.iterdir() if d.is_dir()]
            
            for sample_dir in sample_dirs:
                # Extract sample info
                match = re.match(r"^([A-Za-z0-9]+)_(\d+)$", sample_dir.name)
                if not match:
                    continue
                    
                sample_celltype = match.group(1).lower()
                sample_id = match.group(2)
                
                # Scan files in sample directory
                sample_info = {
                    'celltype': sample_celltype,
                    'sample_id': sample_id,
                    'path': sample_dir,
                    'dapi_file': None,
                    'marker_file': None,
                    'roi_files': [],
                    'other_files': []
                }
                
                for file_path in sample_dir.iterdir():
                    if file_path.is_file():
                        filename = file_path.name.lower()
                        
                        if filename.endswith('.tiff') or filename.endswith('.tif'):
                            if 'b0c0' in filename:
                                sample_info['dapi_file'] = file_path
                            elif 'b0c1' in filename:
                                sample_info['marker_file'] = file_path
                            else:
                                sample_info['other_files'].append(file_path)
                        elif filename.endswith('.zip') and 'roi' in filename:
                            sample_info['roi_files'].append(file_path)
                        else:
                            sample_info['other_files'].append(file_path)
                
                samples[f"{sample_celltype}_{sample_id}"] = sample_info
        
        return samples
    
    def _get_celltypes(self):
        """Get list of unique cell types"""
        celltypes = set()
        for sample_info in self.samples.values():
            celltypes.add(sample_info['celltype'])
        return sorted(list(celltypes))
    
    def _convert_roi_to_mask(self, roi_file):
        """Convert ROI ZIP file to numpy mask"""
        try:
            mask = np.zeros((self.height, self.width), dtype=np.uint16)
            
            with zipfile.ZipFile(roi_file, 'r') as zf:
                label = 1
                for name in zf.namelist():
                    if name.endswith('.roi'):
                        try:
                            roi_bytes = zf.read(name)
                            roi = ImagejRoi.frombytes(roi_bytes)
                            coords = roi.coordinates()
                            
                            if coords is None or len(coords) == 0:
                                if hasattr(roi, 'roitype') and roi.roitype == 10:  # Point ROI
                                    cx, cy = int(roi.left), int(roi.top)
                                    coords = np.array([
                                        [cx - 1, cy - 1], [cx + 1, cy - 1],
                                        [cx + 1, cy + 1], [cx - 1, cy + 1]
                                    ])
                                else:
                                    coords = np.array([
                                        [roi.left, roi.top], [roi.right, roi.top],
                                        [roi.right, roi.bottom], [roi.left, roi.bottom]
                                    ])
                            
                            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=mask.shape)
                            mask[rr, cc] = label
                            label += 1
                            
                        except Exception as e:
                            print(f"Error processing ROI {name}: {e}")
                            continue
            
            return mask
            
        except Exception as e:
            print(f"Error converting ROI file {roi_file}: {e}")
            return None
    
    def display_sample(self, sample_key):
        """Display comprehensive view of a single sample"""
        if sample_key not in self.samples:
            print(f"Sample {sample_key} not found!")
            return
        
        sample_info = self.samples[sample_key]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Sample: {sample_key}', fontsize=16, fontweight='bold')
        
        # Load and display DAPI
        dapi_img = None
        if sample_info['dapi_file']:
            try:
                dapi_img = skio.imread(sample_info['dapi_file'])
                if dapi_img.ndim == 3:
                    dapi_img = dapi_img[..., 0]
                axes[0, 0].imshow(dapi_img, cmap='gray')
                axes[0, 0].set_title(f'DAPI (b0c0)\nShape: {dapi_img.shape}')
            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f'Error loading DAPI:\n{str(e)}', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('DAPI (Error)')
        else:
            axes[0, 0].text(0.5, 0.5, 'No DAPI file found', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('DAPI (Missing)')
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        
        # Load and display Marker
        marker_img = None
        if sample_info['marker_file']:
            try:
                marker_img = skio.imread(sample_info['marker_file'])
                if marker_img.ndim == 3:
                    marker_img = marker_img[..., 0]
                axes[0, 1].imshow(marker_img, cmap='hot')
                axes[0, 1].set_title(f'Marker (b0c1)\nShape: {marker_img.shape}')
            except Exception as e:
                axes[0, 1].text(0.5, 0.5, f'Error loading Marker:\n{str(e)}', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Marker (Error)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Marker file found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Marker (Missing)')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        
        # Create overlay
        if dapi_img is not None and marker_img is not None:
            try:
                dapi_norm = dapi_img / dapi_img.max() if dapi_img.max() > 0 else dapi_img
                marker_norm = marker_img / marker_img.max() if marker_img.max() > 0 else marker_img
                
                overlay = np.zeros((*dapi_img.shape, 3))
                overlay[..., 0] = marker_norm  # Red for marker
                overlay[..., 2] = dapi_norm    # Blue for DAPI
                
                axes[0, 2].imshow(overlay)
                axes[0, 2].set_title('Overlay\n(Blue: DAPI, Red: Marker)')
            except Exception as e:
                axes[0, 2].text(0.5, 0.5, f'Error creating overlay:\n{str(e)}', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Overlay (Error)')
        else:
            axes[0, 2].text(0.5, 0.5, 'Cannot create overlay\n(missing images)', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Overlay (Missing data)')
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        
        # Display ROI masks
        roi_masks = []
        for i, roi_file in enumerate(sample_info['roi_files'][:2]):  # Show up to 2 ROI files
            mask = self._convert_roi_to_mask(roi_file)
            roi_masks.append(mask)
            
            ax = axes[1, i]
            if mask is not None:
                unique_labels = np.unique(mask)
                unique_labels = unique_labels[unique_labels != 0]
                num_regions = len(unique_labels)
                
                if num_regions > 0:
                    # Create colored mask
                    cmap = plt.get_cmap('gist_ncar', num_regions)
                    colored_mask = np.zeros((*mask.shape, 3))
                    for j, label in enumerate(unique_labels):
                        color = cmap(j)[:3]
                        colored_mask[mask == label] = color
                    
                    ax.imshow(colored_mask)
                    ax.set_title(f'{roi_file.name}\n{num_regions} regions')
                else:
                    ax.text(0.5, 0.5, 'Empty mask', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{roi_file.name}\n(Empty)')
            else:
                ax.text(0.5, 0.5, 'Error loading ROI', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{roi_file.name}\n(Error)')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # If no ROI files or only one, fill remaining spaces
        for i in range(len(sample_info['roi_files']), 2):
            ax = axes[1, i]
            ax.text(0.5, 0.5, 'No ROI file', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('ROI (Missing)')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Show ROI overlay if we have both DAPI and ROI
        if len(roi_masks) > 0 and roi_masks[0] is not None and dapi_img is not None:
            try:
                # Create overlay of DAPI and first ROI mask
                roi_binary = (roi_masks[0] > 0).astype(np.uint8)
                dapi_norm = dapi_img / dapi_img.max() if dapi_img.max() > 0 else dapi_img
                
                overlay = np.zeros((*dapi_img.shape, 3))
                overlay[..., 2] = dapi_norm  # Blue for DAPI
                overlay[roi_binary == 1, 0] = 1  # Red for ROI regions
                
                axes[1, 2].imshow(overlay)
                axes[1, 2].set_title('DAPI + ROI Overlay\n(Blue: DAPI, Red: ROI)')
            except Exception as e:
                axes[1, 2].text(0.5, 0.5, f'Error creating ROI overlay:\n{str(e)}', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('ROI Overlay (Error)')
        else:
            axes[1, 2].text(0.5, 0.5, 'Cannot create ROI overlay', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('ROI Overlay (Missing data)')
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        
        # Add file information
        file_info = self._get_file_info(sample_info)
        fig.text(0.02, 0.02, file_info, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
        return fig
    
    def _get_file_info(self, sample_info):
        """Get formatted file information for a sample"""
        info = [
            f"Sample: {sample_info['celltype']}_{sample_info['sample_id']}",
            f"Path: {sample_info['path']}",
            "",
            "Files found:"
        ]
        
        if sample_info['dapi_file']:
            info.append(f"  DAPI: {sample_info['dapi_file'].name}")
        else:
            info.append("  DAPI: MISSING")
            
        if sample_info['marker_file']:
            info.append(f"  Marker: {sample_info['marker_file'].name}")
        else:
            info.append("  Marker: MISSING")
            
        if sample_info['roi_files']:
            info.append(f"  ROI files: {len(sample_info['roi_files'])}")
            for roi_file in sample_info['roi_files']:
                info.append(f"    - {roi_file.name}")
        else:
            info.append("  ROI files: NONE")
            
        if sample_info['other_files']:
            info.append(f"  Other files: {len(sample_info['other_files'])}")
        
        return "\n".join(info)
    
    def create_summary_table(self):
        """Create a summary table of all samples"""
        print("\n" + "="*100)
        print("SCRIBBLES DATASET SUMMARY")
        print("="*100)
        
        # Count by cell type
        celltype_counts = {}
        complete_samples = 0
        
        for sample_key, sample_info in self.samples.items():
            celltype = sample_info['celltype']
            if celltype not in celltype_counts:
                celltype_counts[celltype] = {'total': 0, 'complete': 0, 'missing_dapi': 0, 'missing_marker': 0, 'missing_roi': 0}
            
            celltype_counts[celltype]['total'] += 1
            
            has_dapi = sample_info['dapi_file'] is not None
            has_marker = sample_info['marker_file'] is not None
            has_roi = len(sample_info['roi_files']) > 0
            
            if has_dapi and has_marker and has_roi:
                celltype_counts[celltype]['complete'] += 1
                complete_samples += 1
            
            if not has_dapi:
                celltype_counts[celltype]['missing_dapi'] += 1
            if not has_marker:
                celltype_counts[celltype]['missing_marker'] += 1
            if not has_roi:
                celltype_counts[celltype]['missing_roi'] += 1
        
        print(f"Total samples: {len(self.samples)}")
        print(f"Complete samples: {complete_samples}")
        print(f"Cell types: {len(celltype_counts)}")
        print()
        
        # Print detailed table
        print(f"{'Cell Type':<10} {'Total':<6} {'Complete':<9} {'Missing DAPI':<13} {'Missing Marker':<15} {'Missing ROI':<12}")
        print("-" * 80)
        
        for celltype, counts in sorted(celltype_counts.items()):
            print(f"{celltype:<10} {counts['total']:<6} {counts['complete']:<9} "
                  f"{counts['missing_dapi']:<13} {counts['missing_marker']:<15} {counts['missing_roi']:<12}")
        
        # Show problematic samples
        print("\nProblematic samples:")
        print(f"{'Sample':<20} {'Missing Components':<30}")
        print("-" * 50)
        
        for sample_key, sample_info in sorted(self.samples.items()):
            missing = []
            if not sample_info['dapi_file']:
                missing.append('DAPI')
            if not sample_info['marker_file']:
                missing.append('Marker')
            if not sample_info['roi_files']:
                missing.append('ROI')
            
            if missing:
                print(f"{sample_key:<20} {', '.join(missing):<30}")
    
    def interactive_review(self):
        """Start interactive review session"""
        sample_keys = sorted(list(self.samples.keys()))
        current_index = 0
        
        print("\n" + "="*60)
        print("INTERACTIVE SCRIBBLES REVIEW")
        print("="*60)
        print("Commands:")
        print("  'next' or 'n' - Next sample")
        print("  'prev' or 'p' - Previous sample")
        print("  'goto <sample>' - Go to specific sample")
        print("  'list [celltype]' - Show sample list (optionally filtered)")
        print("  'summary' - Show dataset summary")
        print("  'quit' or 'q' - Exit")
        print("-"*60)
        
        while True:
            current_sample = sample_keys[current_index]
            print(f"\nCurrent: {current_index + 1}/{len(sample_keys)} - {current_sample}")
            
            command = input("Enter command: ").strip().lower()
            
            if command in ['quit', 'q']:
                break
            elif command in ['next', 'n']:
                current_index = (current_index + 1) % len(sample_keys)
                self.display_sample(sample_keys[current_index])
            elif command in ['prev', 'p']:
                current_index = (current_index - 1) % len(sample_keys)
                self.display_sample(sample_keys[current_index])
            elif command.startswith('goto'):
                try:
                    target = command.split()[1]
                    if target in sample_keys:
                        current_index = sample_keys.index(target)
                        self.display_sample(target)
                    else:
                        print(f"Sample {target} not found!")
                except IndexError:
                    print("Usage: goto <sample_key>")
            elif command.startswith('list'):
                parts = command.split()
                celltype_filter = parts[1] if len(parts) > 1 else None
                
                print("\nAvailable samples:")
                for i, sample_key in enumerate(sample_keys):
                    sample_info = self.samples[sample_key]
                    if celltype_filter and sample_info['celltype'] != celltype_filter:
                        continue
                    
                    marker = ">>> " if i == current_index else "    "
                    status = "✓" if (sample_info['dapi_file'] and sample_info['marker_file'] and sample_info['roi_files']) else "✗"
                    print(f"{marker}{i+1:3d}. {sample_key} {status}")
            elif command == 'summary':
                self.create_summary_table()
            elif command == '':
                # Default - show current sample
                self.display_sample(current_sample)
            else:
                print("Unknown command.")

def main():
    """Main function"""
    panel = ScribblesReviewPanel("scribbles")
    
    # Show summary first
    panel.create_summary_table()
    
    # Start interactive review
    panel.interactive_review()

if __name__ == "__main__":
    main()
