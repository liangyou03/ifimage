#!/usr/bin/env python3
"""
Simple Scribbles Analyzer

This script analyzes the scribbles directory structure without visualization
to help verify file organization and identify any issues.
"""

import os
import re
from pathlib import Path
import zipfile

class ScribblesAnalyzer:
    def __init__(self, scribbles_dir="scribbles"):
        """Initialize the analyzer"""
        self.scribbles_dir = Path(scribbles_dir)
        self.samples = self._scan_samples()
    
    def _scan_samples(self):
        """Scan scribbles directory and organize sample information"""
        samples = {}

        if not self.scribbles_dir.exists():
            print(f"Error: Scribbles directory '{self.scribbles_dir}' not found!")
            return samples

        # Find all celltype directories
        celltype_dirs = [d for d in self.scribbles_dir.iterdir()
                        if d.is_dir() and d.name.endswith("_Scribbled")]

        print(f"Found {len(celltype_dirs)} cell type directories:")
        for celltype_dir in celltype_dirs:
            print(f"  - {celltype_dir.name}")

        for celltype_dir in celltype_dirs:
            # Extract celltype from directory name
            celltype = celltype_dir.name.replace("_Scribbled", "").lower()

            # Find all sample directories
            sample_dirs = [d for d in celltype_dir.iterdir() if d.is_dir()]
            print(f"\n{celltype_dir.name}: {len(sample_dirs)} samples")

            for sample_dir in sample_dirs:
                # Extract sample info
                match = re.match(r"^([A-Za-z0-9]+)_(\d+)$", sample_dir.name)
                if not match:
                    print(f"  WARNING: Cannot parse sample directory name: {sample_dir.name}")
                    continue

                sample_celltype = match.group(1).lower()
                sample_id = match.group(2)
                sample_key = f"{sample_celltype}_{sample_id}"

                # Verify celltype consistency
                if sample_celltype != celltype:
                    print(f"  WARNING: Celltype mismatch in {sample_dir.name} (expected {celltype}, got {sample_celltype})")

                # Scan files in sample directory
                sample_info = {
                    'celltype': sample_celltype,
                    'sample_id': sample_id,
                    'path': sample_dir,
                    'dapi_file': None,
                    'marker_file': None,
                    'cellbodies_roi': None,
                    'dapi_roi': None,
                    'other_roi_files': [],
                    'other_files': [],
                    'all_files': []
                }

                for file_path in sample_dir.iterdir():
                    if file_path.is_file():
                        sample_info['all_files'].append(file_path)
                        filename = file_path.name.lower()

                        if filename.endswith('.tiff') or filename.endswith('.tif'):
                            if 'b0c0' in filename:
                                sample_info['dapi_file'] = file_path
                            elif 'b0c1' in filename:
                                sample_info['marker_file'] = file_path
                            else:
                                sample_info['other_files'].append(file_path)
                        elif filename.endswith('.zip') and 'roi' in filename:
                            # Categorize ROI files based on their names
                            if 'cellbodies' in filename or 'cell_bodies' in filename:
                                sample_info['cellbodies_roi'] = file_path
                            elif 'dapi' in filename or 'alldapi' in filename:
                                sample_info['dapi_roi'] = file_path
                            else:
                                sample_info['other_roi_files'].append(file_path)
                        else:
                            sample_info['other_files'].append(file_path)

                samples[sample_key] = sample_info

        return samples
    
    def analyze_roi_files(self, sample_key):
        """Analyze ROI files for a specific sample"""
        if sample_key not in self.samples:
            print(f"Sample {sample_key} not found!")
            return

        sample_info = self.samples[sample_key]
        print(f"\nAnalyzing ROI files for {sample_key}:")

        # Analyze cell bodies ROI
        if sample_info['cellbodies_roi']:
            print(f"\n  Cell Bodies ROI: {sample_info['cellbodies_roi'].name}")
            self._analyze_single_roi_file(sample_info['cellbodies_roi'])
        else:
            print(f"\n  Cell Bodies ROI: NOT FOUND")

        # Analyze DAPI ROI
        if sample_info['dapi_roi']:
            print(f"\n  DAPI ROI: {sample_info['dapi_roi'].name}")
            self._analyze_single_roi_file(sample_info['dapi_roi'])
        else:
            print(f"\n  DAPI ROI: NOT FOUND")

        # Analyze other ROI files
        if sample_info['other_roi_files']:
            print(f"\n  Other ROI files ({len(sample_info['other_roi_files'])}):")
            for roi_file in sample_info['other_roi_files']:
                print(f"\n    {roi_file.name}:")
                self._analyze_single_roi_file(roi_file, indent="      ")

    def _analyze_single_roi_file(self, roi_file, indent="    "):
        """Analyze a single ROI ZIP file"""
        try:
            with zipfile.ZipFile(roi_file, 'r') as zf:
                roi_names = [name for name in zf.namelist() if name.endswith('.roi')]
                print(f"{indent}Contains {len(roi_names)} ROI regions")
                print(f"{indent}File size: {roi_file.stat().st_size:,} bytes")

                if len(roi_names) <= 10:  # Show details for small numbers
                    for roi_name in roi_names:
                        print(f"{indent}  - {roi_name}")
                else:
                    print(f"{indent}  - {roi_names[0]}")
                    print(f"{indent}  - ... ({len(roi_names)-2} more)")
                    print(f"{indent}  - {roi_names[-1]}")

        except Exception as e:
            print(f"{indent}ERROR: Cannot read ZIP file - {str(e)}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("\n" + "="*80)
        print("SCRIBBLES DATASET ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        total_samples = len(self.samples)
        celltypes = set(info['celltype'] for info in self.samples.values())
        
        print(f"Total samples: {total_samples}")
        print(f"Cell types: {len(celltypes)} ({', '.join(sorted(celltypes))})")
        
        # Count by completeness - now including dapi_multi_mask requirement
        complete_samples = 0
        missing_dapi_image = 0
        missing_marker = 0
        missing_cellbodies_roi = 0
        missing_dapi_roi = 0

        for sample_info in self.samples.values():
            has_dapi_image = sample_info['dapi_file'] is not None
            has_marker = sample_info['marker_file'] is not None
            has_cellbodies_roi = sample_info['cellbodies_roi'] is not None
            has_dapi_roi = sample_info['dapi_roi'] is not None

            # Complete sample needs: DAPI image, Marker image, Cell bodies ROI, and DAPI ROI
            if has_dapi_image and has_marker and has_cellbodies_roi and has_dapi_roi:
                complete_samples += 1

            if not has_dapi_image:
                missing_dapi_image += 1
            if not has_marker:
                missing_marker += 1
            if not has_cellbodies_roi:
                missing_cellbodies_roi += 1
            if not has_dapi_roi:
                missing_dapi_roi += 1

        print(f"\nCompleteness Analysis:")
        print(f"  Complete samples (all 4 components): {complete_samples}")
        print(f"  Missing DAPI image (b0c0): {missing_dapi_image}")
        print(f"  Missing Marker image (b0c1): {missing_marker}")
        print(f"  Missing Cell Bodies ROI: {missing_cellbodies_roi}")
        print(f"  Missing DAPI ROI (for dapi_multi_mask): {missing_dapi_roi}")

        # Additional statistics
        has_any_roi = sum(1 for s in self.samples.values()
                         if s['cellbodies_roi'] or s['dapi_roi'] or s['other_roi_files'])
        print(f"  Samples with any ROI files: {has_any_roi}")
        print(f"  Samples with other ROI files: {sum(1 for s in self.samples.values() if s['other_roi_files'])}")
        
        # Count by cell type
        print(f"\nBreakdown by cell type:")
        celltype_stats = {}
        for sample_info in self.samples.values():
            celltype = sample_info['celltype']
            if celltype not in celltype_stats:
                celltype_stats[celltype] = {
                    'total': 0, 'complete': 0, 'missing_dapi_img': 0,
                    'missing_marker': 0, 'missing_cellbodies_roi': 0, 'missing_dapi_roi': 0
                }

            celltype_stats[celltype]['total'] += 1

            has_dapi_img = sample_info['dapi_file'] is not None
            has_marker = sample_info['marker_file'] is not None
            has_cellbodies_roi = sample_info['cellbodies_roi'] is not None
            has_dapi_roi = sample_info['dapi_roi'] is not None

            if has_dapi_img and has_marker and has_cellbodies_roi and has_dapi_roi:
                celltype_stats[celltype]['complete'] += 1
            if not has_dapi_img:
                celltype_stats[celltype]['missing_dapi_img'] += 1
            if not has_marker:
                celltype_stats[celltype]['missing_marker'] += 1
            if not has_cellbodies_roi:
                celltype_stats[celltype]['missing_cellbodies_roi'] += 1
            if not has_dapi_roi:
                celltype_stats[celltype]['missing_dapi_roi'] += 1

        print(f"{'Type':<8} {'Total':<6} {'Complete':<9} {'No DAPI':<8} {'No Marker':<10} {'No CellROI':<10} {'No DAPIROI':<10}")
        print("-" * 80)
        for celltype in sorted(celltype_stats.keys()):
            stats = celltype_stats[celltype]
            print(f"{celltype:<8} {stats['total']:<6} {stats['complete']:<9} "
                  f"{stats['missing_dapi_img']:<8} {stats['missing_marker']:<10} "
                  f"{stats['missing_cellbodies_roi']:<10} {stats['missing_dapi_roi']:<10}")
    
    def list_problematic_samples(self):
        """List samples with missing components"""
        print("\n" + "="*80)
        print("PROBLEMATIC SAMPLES")
        print("="*80)

        problems_found = False

        for sample_key in sorted(self.samples.keys()):
            sample_info = self.samples[sample_key]
            issues = []

            if not sample_info['dapi_file']:
                issues.append('No DAPI image (b0c0)')
            if not sample_info['marker_file']:
                issues.append('No Marker image (b0c1)')
            if not sample_info['cellbodies_roi']:
                issues.append('No Cell Bodies ROI')
            if not sample_info['dapi_roi']:
                issues.append('No DAPI ROI (needed for dapi_multi_mask)')

            if issues:
                problems_found = True
                print(f"\n{sample_key}:")
                print(f"  Path: {sample_info['path']}")
                print(f"  Issues: {', '.join(issues)}")
                print(f"  Files found:")
                print(f"    DAPI image: {'✓' if sample_info['dapi_file'] else '✗'}")
                print(f"    Marker image: {'✓' if sample_info['marker_file'] else '✗'}")
                print(f"    Cell Bodies ROI: {'✓' if sample_info['cellbodies_roi'] else '✗'}")
                print(f"    DAPI ROI: {'✓' if sample_info['dapi_roi'] else '✗'}")
                if sample_info['other_roi_files']:
                    print(f"    Other ROI files: {len(sample_info['other_roi_files'])}")
                if sample_info['other_files']:
                    print(f"    Other files: {len(sample_info['other_files'])}")

        if not problems_found:
            print("🎉 No problematic samples found! All samples have:")
            print("   - DAPI image (b0c0)")
            print("   - Marker image (b0c1)")
            print("   - Cell Bodies ROI")
            print("   - DAPI ROI (for dapi_multi_mask)")
    
    def show_file_patterns(self):
        """Show common file naming patterns"""
        print("\n" + "="*80)
        print("FILE NAMING PATTERNS")
        print("="*80)

        dapi_patterns = set()
        marker_patterns = set()
        cellbodies_roi_patterns = set()
        dapi_roi_patterns = set()
        other_roi_patterns = set()
        other_patterns = set()

        for sample_info in self.samples.values():
            if sample_info['dapi_file']:
                dapi_patterns.add(sample_info['dapi_file'].name)
            if sample_info['marker_file']:
                marker_patterns.add(sample_info['marker_file'].name)
            if sample_info['cellbodies_roi']:
                cellbodies_roi_patterns.add(sample_info['cellbodies_roi'].name)
            if sample_info['dapi_roi']:
                dapi_roi_patterns.add(sample_info['dapi_roi'].name)
            for roi_file in sample_info['other_roi_files']:
                other_roi_patterns.add(roi_file.name)
            for other_file in sample_info['other_files']:
                other_patterns.add(other_file.name)

        print(f"DAPI image patterns ({len(dapi_patterns)} unique):")
        for pattern in sorted(list(dapi_patterns)[:10]):  # Show first 10
            print(f"  {pattern}")
        if len(dapi_patterns) > 10:
            print(f"  ... and {len(dapi_patterns) - 10} more")

        print(f"\nMarker image patterns ({len(marker_patterns)} unique):")
        for pattern in sorted(list(marker_patterns)[:10]):
            print(f"  {pattern}")
        if len(marker_patterns) > 10:
            print(f"  ... and {len(marker_patterns) - 10} more")

        print(f"\nCell Bodies ROI patterns ({len(cellbodies_roi_patterns)} unique):")
        for pattern in sorted(list(cellbodies_roi_patterns)):
            print(f"  {pattern}")

        print(f"\nDAPI ROI patterns ({len(dapi_roi_patterns)} unique):")
        for pattern in sorted(list(dapi_roi_patterns)):
            print(f"  {pattern}")

        if other_roi_patterns:
            print(f"\nOther ROI patterns ({len(other_roi_patterns)} unique):")
            for pattern in sorted(list(other_roi_patterns)):
                print(f"  {pattern}")

        if other_patterns:
            print(f"\nOther file patterns ({len(other_patterns)} unique):")
            for pattern in sorted(list(other_patterns)[:10]):
                print(f"  {pattern}")
            if len(other_patterns) > 10:
                print(f"  ... and {len(other_patterns) - 10} more")
    
    def interactive_explorer(self):
        """Interactive command-line explorer"""
        print("\n" + "="*60)
        print("INTERACTIVE SCRIBBLES EXPLORER")
        print("="*60)
        print("Commands:")
        print("  'summary' - Show summary report")
        print("  'problems' - List problematic samples")
        print("  'patterns' - Show file naming patterns")
        print("  'expected' - Show expected output files")
        print("  'analyze <sample>' - Analyze specific sample")
        print("  'list [celltype]' - List samples (optionally filtered)")
        print("  'roi <sample>' - Analyze ROI files for sample")
        print("  'quit' or 'q' - Exit")
        print("-"*60)
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command in ['quit', 'q']:
                break
            elif command == 'summary':
                self.create_summary_report()
            elif command == 'problems':
                self.list_problematic_samples()
            elif command == 'patterns':
                self.show_file_patterns()
            elif command == 'expected':
                self.check_expected_outputs()
            elif command.startswith('analyze'):
                try:
                    sample_key = command.split()[1]
                    self.analyze_sample(sample_key)
                except IndexError:
                    print("Usage: analyze <sample_key>")
            elif command.startswith('roi'):
                try:
                    sample_key = command.split()[1]
                    self.analyze_roi_files(sample_key)
                except IndexError:
                    print("Usage: roi <sample_key>")
            elif command.startswith('list'):
                parts = command.split()
                celltype_filter = parts[1] if len(parts) > 1 else None
                self.list_samples(celltype_filter)
            else:
                print("Unknown command. Type 'quit' to exit.")
    
    def analyze_sample(self, sample_key):
        """Analyze a specific sample in detail"""
        if sample_key not in self.samples:
            print(f"Sample {sample_key} not found!")
            return
        
        sample_info = self.samples[sample_key]
        print(f"\nDetailed analysis for {sample_key}:")
        print(f"  Cell type: {sample_info['celltype']}")
        print(f"  Sample ID: {sample_info['sample_id']}")
        print(f"  Path: {sample_info['path']}")
        
        print(f"\n  DAPI image (b0c0):")
        if sample_info['dapi_file']:
            print(f"    ✓ {sample_info['dapi_file'].name}")
            print(f"      Size: {sample_info['dapi_file'].stat().st_size:,} bytes")
        else:
            print(f"    ✗ Not found")

        print(f"\n  Marker image (b0c1):")
        if sample_info['marker_file']:
            print(f"    ✓ {sample_info['marker_file'].name}")
            print(f"      Size: {sample_info['marker_file'].stat().st_size:,} bytes")
        else:
            print(f"    ✗ Not found")

        print(f"\n  Cell Bodies ROI (for cellbodies_multimask):")
        if sample_info['cellbodies_roi']:
            print(f"    ✓ {sample_info['cellbodies_roi'].name}")
            print(f"      Size: {sample_info['cellbodies_roi'].stat().st_size:,} bytes")
        else:
            print(f"    ✗ Not found")

        print(f"\n  DAPI ROI (for dapi_multi_mask):")
        if sample_info['dapi_roi']:
            print(f"    ✓ {sample_info['dapi_roi'].name}")
            print(f"      Size: {sample_info['dapi_roi'].stat().st_size:,} bytes")
        else:
            print(f"    ✗ Not found")

        if sample_info['other_roi_files']:
            print(f"\n  Other ROI files ({len(sample_info['other_roi_files'])}):")
            for roi_file in sample_info['other_roi_files']:
                print(f"    - {roi_file.name}")
                print(f"      Size: {roi_file.stat().st_size:,} bytes")

        if sample_info['other_files']:
            print(f"\n  Other files ({len(sample_info['other_files'])}):")
            for other_file in sample_info['other_files']:
                print(f"    - {other_file.name}")
    
    def list_samples(self, celltype_filter=None):
        """List all samples, optionally filtered by celltype"""
        print(f"\nSample list" + (f" (filtered by {celltype_filter})" if celltype_filter else "") + ":")

        count = 0
        for sample_key in sorted(self.samples.keys()):
            sample_info = self.samples[sample_key]

            if celltype_filter and sample_info['celltype'] != celltype_filter:
                continue

            count += 1
            has_dapi = "✓" if sample_info['dapi_file'] else "✗"
            has_marker = "✓" if sample_info['marker_file'] else "✗"
            has_cellbodies_roi = "✓" if sample_info['cellbodies_roi'] else "✗"
            has_dapi_roi = "✓" if sample_info['dapi_roi'] else "✗"

            # Overall status
            complete = "✓" if all([sample_info['dapi_file'], sample_info['marker_file'],
                                  sample_info['cellbodies_roi'], sample_info['dapi_roi']]) else "✗"

            print(f"  {count:3d}. {sample_key:<20} Complete:{complete} "
                  f"DAPI:{has_dapi} Marker:{has_marker} CellROI:{has_cellbodies_roi} DAPIROI:{has_dapi_roi}")

        print(f"\nTotal: {count} samples")

    def check_expected_outputs(self):
        """Check what files would be generated for ifimage_tools.py"""
        print("\n" + "="*80)
        print("EXPECTED OUTPUT FILES FOR IFIMAGE_TOOLS")
        print("="*80)
        print("Based on the current scribbles, these files should be generated:")
        print()

        expected_images = []
        expected_masks = []

        for sample_key, sample_info in sorted(self.samples.items()):
            celltype = sample_info['celltype']
            sample_id = sample_info['sample_id']

            # Expected image files
            if sample_info['dapi_file']:
                expected_images.append(f"{celltype}_{sample_id}.tiff")  # DAPI channel
            if sample_info['marker_file']:
                expected_images.append(f"{celltype}_{sample_id}_marker.tiff")  # Marker channel

            # Expected mask files
            if sample_info['cellbodies_roi']:
                expected_masks.append(f"{celltype}_{sample_id}_cellbodies.npy")  # From CellBodies ROI
            if sample_info['dapi_roi']:
                expected_masks.append(f"{celltype}_{sample_id}_dapimultimask.npy")  # From DAPI ROI

        print(f"Expected image files ({len(expected_images)}):")
        print("  These go in the images directory:")
        for img_file in sorted(expected_images)[:20]:  # Show first 20
            print(f"    {img_file}")
        if len(expected_images) > 20:
            print(f"    ... and {len(expected_images) - 20} more")

        print(f"\nExpected mask files ({len(expected_masks)}):")
        print("  These go in the masks directory:")
        for mask_file in sorted(expected_masks)[:20]:  # Show first 20
            print(f"    {mask_file}")
        if len(expected_masks) > 20:
            print(f"    ... and {len(expected_masks) - 20} more")

        # Summary for ifimage_tools usage
        complete_samples = sum(1 for s in self.samples.values()
                             if s['dapi_file'] and s['marker_file'] and
                                s['cellbodies_roi'] and s['dapi_roi'])

        print(f"\nDataset Summary for IfImageDataset:")
        print(f"  Complete samples: {complete_samples}")
        print(f"  Expected DAPI images: {sum(1 for s in self.samples.values() if s['dapi_file'])}")
        print(f"  Expected marker images: {sum(1 for s in self.samples.values() if s['marker_file'])}")
        print(f"  Expected cellbodies masks: {sum(1 for s in self.samples.values() if s['cellbodies_roi'])}")
        print(f"  Expected dapi_multi_masks: {sum(1 for s in self.samples.values() if s['dapi_roi'])}")

        print(f"\nTo load this dataset after processing:")
        print(f"```python")
        print(f"from ifimage_tools import IfImageDataset")
        print(f"dataset = IfImageDataset(")
        print(f"    image_dir='processed_dataset/images',")
        print(f"    nuclei_masks_dir='processed_dataset/masks',")
        print(f"    cell_masks_dir='processed_dataset/masks'")
        print(f")")
        print(f"dataset.load_data()")
        print(f"dataset.summary(table=True)  # Check for missing components")
        print(f"```")

def main():
    """Main function"""
    analyzer = ScribblesAnalyzer("scribbles")
    
    if not analyzer.samples:
        print("No samples found. Please check the scribbles directory structure.")
        return
    
    # Show initial summary
    analyzer.create_summary_report()
    analyzer.list_problematic_samples()
    
    # Start interactive explorer
    analyzer.interactive_explorer()

if __name__ == "__main__":
    main()
