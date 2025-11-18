#!/bin/bash
#
# plot_all.sh
#
# Generate all visualization plots from saved evaluation results.
# This script can be run multiple times without re-evaluating.
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=========================================="
echo "Generating all visualization plots"
echo "=========================================="
echo ""

# Track timing
START_TIME=$(date +%s)

# List of plot scripts in order
SCRIPTS=(
    "plot_color_legend.py"
    "plot_cell_overall.py"
    "plot_cell_per_type.py"
    "plot_marker_comparison.py"
    "plot_nuclei_overall.py"
    "plot_side_by_side.py"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo "Running $script..."
        python "$SCRIPT_DIR/$script"
        echo "✓ $script complete"
        echo ""
    else
        echo "⚠️  Warning: $script not found, skipping..."
        echo ""
    fi
done

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=========================================="
echo "All plots generated successfully!"
echo "Total time: ${ELAPSED}s"
echo "=========================================="
echo ""
echo "Check the plots directory for output files."