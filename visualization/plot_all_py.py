#!/usr/bin/env python3
"""
plot_all.py

Generate all visualization plots from saved evaluation results.
This script can be run multiple times without re-evaluating.
"""

import sys
import time
import subprocess
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    
    scripts = [
        "plot_color_legend.py",      # Color reference (always first)
        "plot_cell_overall.py",
        "plot_cell_per_type.py",
        "plot_marker_comp.py",
        "plot_nuclei_overall.py",
        "plot_side_by_side.py",
    ]
    
    print("=" * 60)
    print("Generating all visualization plots")
    print("=" * 60)
    print()
    
    start = time.time()
    success = 0
    failed = 0
    
    for script in scripts:
        script_path = script_dir / script
        
        if not script_path.exists():
            print(f"⚠️  Warning: {script} not found, skipping...")
            print()
            failed += 1
            continue
        
        print(f"Running {script}...")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=script_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            print(f"✓ {script} complete")
            print()
            success += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {script}:")
            print(e.stderr)
            print()
            failed += 1
    
    elapsed = time.time() - start
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  • Successful: {success}")
    print(f"  • Failed: {failed}")
    print(f"  • Total time: {elapsed:.1f}s")
    print("=" * 60)
    print()
    
    if failed == 0:
        print("All plots generated successfully!")
        print("Check the plots directory for output files.")
    else:
        print(f"Warning: {failed} plot(s) failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
