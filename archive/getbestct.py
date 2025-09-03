import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stardist.matching import matching

# -- 1. Load the dataset -----------------------------------------------------

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
print("✅ Dataset loaded from dataset.pkl")

# -- 2. Set up IoU thresholds and the methods we care about -----------------

iou_thresholds = np.arange(0.5, 1.0, 0.05)
# We’re only computing mAP for “cellpose2chan” here, but you could loop METHODS if you want:
METHODS = ["cellpose", "cellpose2chan", "watershed", "cell_expansion"]

# This dict will hold, for each cell type, the sample with the highest mAP
best_samples = {}

# -- 3. Loop over every sample, compute mAP for cellpose2chan ----------------
for sample_id, sample in dataset.samples.items():
    # Skip if this sample has no ground-truth cell type
    if sample.celltype is None:
        print(f"⚠️ Sample {sample_id} has no celltype, skipping.")
        continue

    # Skip if no prediction for our target method
    mask_key = "cellpose2chan"
    if mask_key not in sample.cyto_positive_masks:
        print(f"⚠️ Sample {sample_id} is missing '{mask_key}' prediction, skipping.")
        continue

    # Load the predicted mask (could be an array already or a path to a .npy)
    pred = sample.cyto_positive_masks[mask_key]
    try:
        pred_mask = pred if isinstance(pred, np.ndarray) else np.load(pred)
    except Exception as e:
        print(f"❗ Failed loading pred mask for sample {sample_id}: {e}")
        continue

    # Ground-truth multi-instance mask
    gt_mask = sample.cellbodies_multimask

    # Compute precision at each IoU threshold
    precisions = []
    for thr in iou_thresholds:
        try:
            match = matching(gt_mask, pred_mask, thresh=thr)
            precisions.append(match.precision)
        except Exception as e:
            print(f"❗ Matching error (sample {sample_id}, IoU {thr:.2f}): {e}")
            precisions.append(0)

    # If everything busted out to zero, skip
    if all(p == 0 for p in precisions):
        print(f"⚠️ All precisions 0 for sample {sample_id}, skipping.")
        continue

    # Calculate mean Average Precision
    mAP = float(np.mean(precisions))
    ctype = sample.celltype

    # Update best sample for this cell type
    prev = best_samples.get(ctype)
    if prev is None or mAP > prev["mAP"]:
        best_samples[ctype] = {
            "sample": sample,
            "sample_id": sample_id,
            "mAP": mAP,
            "precisions": precisions
        }
        print(f"🔝 Cell type '{ctype}' best sample is now {sample_id} (mAP={mAP:.3f})")

# -- 4. Prepare output folder ------------------------------------------------
output_dir = "best_annotated_images"
os.makedirs(output_dir, exist_ok=True)

# This will collect report rows for a final CSV
report_rows = []

# -- 5. For each best sample: plot Precision vs IoU & save ------------------
for ctype, info in best_samples.items():
    sample = info["sample"]
    sample_id = info["sample_id"]
    mAP = info["mAP"]
    precisions = info["precisions"]

    print(f"🖼 Generating plots for cell type '{ctype}' (sample {sample_id})…")

    # Plot precision vs IoU
    plt.figure(figsize=(6, 4))
    plt.plot(iou_thresholds, precisions, marker='o', linestyle='-')
    plt.xlabel("IoU Threshold")
    plt.ylabel("Precision")
    plt.title(f"{ctype} — Sample {sample_id} (mAP={mAP:.3f})")
    plot_file = os.path.join(output_dir, f"{ctype}_precision_vs_iou.png")
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved plot to {plot_file}")

    report_rows.append({
        "celltype": ctype,
        "sample_id": sample_id,
        "mAP": mAP,
        "precision_vs_iou_plot": plot_file
    })

# -- 6. Write the CSV report -------------------------------------------------
report_df = pd.DataFrame(report_rows)
csv_path = os.path.join(output_dir, "best_samples_report.csv")
report_df.to_csv(csv_path, index=False)
print(f"📄 Report CSV saved to {csv_path}")

print("🎉 All done! Best-sample plots & report are ready.")
