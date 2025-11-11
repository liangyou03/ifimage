algs=(
# /ihome/jbwang/liy121/ifimage/02_stardist_benchmark/cyto_prediction
# /ihome/jbwang/liy121/ifimage/03_cellsam_benchmark/cyto_prediction
# /ihome/jbwang/liy121/ifimage/04_mesmer_benchmark/cyto_prediction
# /ihome/jbwang/liy121/ifimage/06_watershed_benchmark/cyto_prediction
# /ihome/jbwang/liy121/ifimage/07_omnipose_benchmark/cyto_prediction
# /ihome/jbwang/liy121/ifimage/011_lacss/cyto_prediction
# /ihome/jbwang/liy121/ifimage/09_hovernet_benchmark/cyto_prediction
# /ihome/jbwang/liy121/ifimage/08_splinedist_benchmark/cyto_prediction
/ihome/jbwang/liy121/ifimage/012_microsam_benchmark/cyto_prediction
)
MARKER=/ihome/jbwang/liy121/ifimage/00_dataset

for d in "${algs[@]}"; do
python - <<PY
from pathlib import Path
import run_refilter_batch as R
R.CYTO_DIR    = Path(r"${d}")
R.OUT_ROOT    = R.CYTO_DIR.parent / "refilter_outputs"
R.MARKER_DIR  = Path(r"${MARKER}")
R.CYTO_GLOB = "*.npy"
R.REMOVE_FROM_CYTO_STEM += ["_pred_cyto","_pred_cell","_cyto","_cell","_prediction","_pred","_filtered","_filter"]

R.run_config("mean","otsu",100,"off")
PY
done
