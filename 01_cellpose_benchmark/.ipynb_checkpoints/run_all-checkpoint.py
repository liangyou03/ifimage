# run_all.py — sweep ALL experiments (no evaluation; saves masks only)

from pathlib import Path
import os
import run_refilter_batch as exp  # your batch module

# ----- Optional: override paths via env vars -----
DATA_DIR = os.getenv("DATA_DIR")        # e.g. "/ihome/jbwang/liy121/ifimage/00_dataset"
CYTO_DIR = os.getenv("CYTO_DIR")        # e.g. "/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/cyto_prediction"
OUT_ROOT = os.getenv("OUT_ROOT")        # e.g. "/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/refilter_experiment"
GATE_DIR = os.getenv("GATE_DIR")        # e.g. "/ihome/jbwang/liy121/ifimage/01_cellpose_benchmark/marker_only_prediction"

if DATA_DIR: exp.DATA_DIR = Path(DATA_DIR)
if CYTO_DIR: exp.CYTO_DIR = Path(CYTO_DIR)
if OUT_ROOT: exp.OUT_ROOT = Path(OUT_ROOT)
if GATE_DIR: exp.GATE_DIR = Path(GATE_DIR)

exp.ensure_dir(exp.OUT_ROOT)

FEATURES   = ["mean", "bgcorr_mean", "ring_mean", "zscore_mean"]
THRESHOLDS = ["gmm", "otsu"]
MIN_AREAS  = [0, 100, 170, 220]
GATES      = ["off", "auto"]

print("=== Run ALL refilter configs (no evaluation) ===")
print(f"DATA_DIR: {exp.DATA_DIR}\nCYTO_DIR: {exp.CYTO_DIR}\nOUT_ROOT: {exp.OUT_ROOT}\nGATE_DIR: {exp.GATE_DIR}")

for f in FEATURES:
    for t in THRESHOLDS:
        for a in MIN_AREAS:
            for g in GATES:
                print(f"-> feature={f}, thr={t}, min_area={a}, gate={g}")
                exp.run_config(f, t, int(a), str(g))

print("\n[Done] All refined masks saved under:", exp.OUT_ROOT.resolve())
