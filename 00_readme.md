# Header

data is in 00_dataset right now

looks like this:
gfap_6390_cellbodies.npy       -> marker mask manual scribble
gfap_6390_dapimultimask.npy    -> dapi mask manual scribble
gfap_6390_marker.tiff -> raw marker chan
gfap_6390.tiff -> raw dapi chan

Every image is 1388 * 1040


## Nov 27
/ihome/jbwang/liy121/ifimage/run_evaluation.py

This script evaluates all segmentation algorithms and saves results to disk.
You only need to run this ONCE, or when:
  - Adding new algorithms
  - Changing ground truth data
  - Modifying evaluation parameters (IoU thresholds, etc.)

- /ihome/jbwang/liy121/ifimage/visualization
集中管理可视化文件

- /ihome/jbwang/liy121/ifimage/visualization/config.py
管理可视化参数与部分路径



## AUG 30
mesmer env is same as cellsam
finish prediction of mesmer and stardist
i think i'll use CellViT in cellposesam env cellvit is extremely hard to implement


pecam_15971 size
pecam_15972 size issue
15973 too few

oval shape:
4201
8265
1120
2529
i edit them into _ovaldapimultimask to give better results




## AUG 25
Set up the environment of ifimage_cellsam and manage to run cellsam -> 32gb OOM 
Write a slurm file /liy121/ifimage/03_cellsam_benchmark/run_cellsam_nuc.slurm



## AUG 23
### big updates
algorithms in seperate env and seperate folders
and in every folder there is env.yml env info
utils.py for dataclass and dataset class
pre_cyto & pre_nuc as named

### Update logic of segmentation
Expected to test 10+ algorithms
make some folders in ifimage -> each correspond a environment



### update the dataset with previous lack image
move the data directory to /liy121/ifimage/images
this incluede the marker channel of the following image
neun_1120
neun_1360
neun_1492
neun_1534
neun_2144
neun_2322
neun_2529
neun_2783



## AUG 17
Slightly improve the env for mmdetection and successfully use swin-t
first transform data into

first transforn coco 80 into ssh://unnamedaug15/ihome/jbwang/liy121/ifimage/archive/get80cats.bash

"Great—this last crash is 100% due to a class-mismatch during formatting:
Your model checkpoint is COCO-80-class.
Your test dataset JSON has 1 class (cell).
When MMDet writes segm.json, it maps prediction labels (0..79) to dataset.cat_ids (length 1) → IndexError.
Let’s fix it by giving the tester a dummy 80-class COCO category list (no GT needed). Then the formatter is happy, and you still get masks."
then use 

ssh://unnamedaug15/ihome/jbwang/liy121/ifimage/swint.bash

to get the result 


# 1) Environment / session
```bash
# GPU interactive session on CRC (preempt queue)
salloc --cluster=gpu --partition=preempt --gres=gpu:1 --mem=32G --time=03:00:00
# Activate env + minimal deps
conda activate openmmlab
```
---

# 2) Core inference (MMDetection → segm.json)
## 2.1 Variables

```bash
cd /ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection
CONFIG=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py   # (confirmed present)
CKPT=/ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection/latest_swin-t.pth
DATA=/ihome/jbwang/liy121/ifimage/processed_dataset
OUTP=/ihome/jbwang/liy121/infer/swin_t/swin_t                # -> writes swin_t.segm.json
mkdir -p /ihome/jbwang/liy121/infer/swin_t
```

## 2.2 MMDet 2.x style “format-only” (your repo style)

```bash
python tools/test.py "$CONFIG" "$CKPT" \
  --format-only \
  --options "jsonfile_prefix=$OUTP" \
  --cfg-options \
    data.test.type=CocoDataset \
    data.test.data_root="$DATA/" \
    data.test.ann_file=annotations_images_COCO80.json \
    data.test.img_prefix=images/
```

**What/How:** Runs inference only and dumps predictions to `$OUTP.segm.json`. We used a dummy **COCO-80 categories** test JSON to avoid class-mismatch errors.

---

# 3) Helper JSON generators to make inference robust

## 3.1 Images-only JSON (remove missing images)

```bash
python - <<'PY'
import json, os, glob
from PIL import Image
root="/ihome/jbwang/liy121/ifimage/processed_dataset"; img=os.path.join(root,"images")
ims=[]
for i,p in enumerate(sorted(sum([glob.glob(os.path.join(img,f"*.{e}")) for e in("tiff","tif","png","jpg","jpeg")],[])),1):
    try: w,h=Image.open(p).size; ims.append({"id":i,"file_name":os.path.basename(p),"width":w,"height":h})
    except: pass
cats=[{"id":1,"name":"cell"}]
out=os.path.join(root,"annotations_images_only.json")
json.dump({"images":ims,"annotations":[],"categories":cats},open(out,"w"))
print("[OK]",out,len(ims))
PY
```

**Why/How:** Keeps only images that physically exist to avoid `FileNotFoundError`.

## 3.2 Images + COCO-80 dummy categories

```bash
python - <<'PY'
import os,glob,json
from PIL import Image
root="/ihome/jbwang/liy121/ifimage/processed_dataset"; img=os.path.join(root,"images")
ims=[]
for i,p in enumerate(sorted(glob.glob(os.path.join(img,"*"))),1):
    try: w,h=Image.open(p).size; ims.append({"id":i,"file_name":os.path.basename(p),"width":w,"height":h})
    except: pass
c80=["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",
"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
"backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
"baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
"banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
"potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
"oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
cats=[{"id":i+1,"name":n} for i,n in enumerate(c80)]
out=os.path.join(root,"annotations_images_COCO80.json")
json.dump({"images":ims,"annotations":[],"categories":cats},open(out,"w"))
print("[OK]",out,len(ims))
PY
```

**Why/How:** Avoids `IndexError: list index out of range` when your pretrained Swin-T predicts 80 classes.

---

# 4) Swin-T convenience script

## 4.1 `swint.bash` (inference + mask conversion)

**Path:** `/ihome/jbwang/liy121/ifimage/swint.bash`
**What:** One-shot inference to `.segm.json`, then convert to masks.

```bash
#!/usr/bin/env bash
set -euo pipefail
conda activate openmmlab
cd /ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection
pip install -q pycocotools tifffile pillow

CONFIG=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py
CKPT=/ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection/latest_swin-t.pth
DATA=/ihome/jbwang/liy121/ifimage/processed_dataset
OUTDIR=/ihome/jbwang/liy121/infer/swin_t
OUTP=$OUTDIR/swin_t
mkdir -p "$OUTDIR"

python tools/test.py "$CONFIG" "$CKPT" \
  --format-only \
  --options "jsonfile_prefix=$OUTP" \
  --cfg-options \
    data.test.type=CocoDataset \
    data.test.data_root="$DATA/" \
    data.test.ann_file=annotations_images_COCO80.json \
    data.test.img_prefix=images/

# requires converter below in tools/
python tools/coco_json_to_instance_masks.py \
  --coco-ann "$DATA/annotations_images_COCO80.json" \
  --coco-res "$OUTP.segm.json" \
  --out-dir  "$OUTDIR/masks"
```

**How to use:**
`bash /ihome/jbwang/liy121/ifimage/swint.bash`
**Output:** `swin_t.segm.json` + PNG masks under `/infer/swin_t/`.

---

# 5) Result → mask converters & sanity visualization

## 5.1 `tools/coco_json_to_instance_masks.py`

**What:** Minimal RLE decode → per-image **instance-ID** PNG (uint16) or per-instance binary masks.

```python
# usage
python tools/coco_json_to_instance_masks.py \
  --coco-ann /.../annotations_images_COCO80.json \
  --coco-res /.../swin_t.segm.json \
  --out-dir  /.../masks
# add --per-instance for one 0/1 PNG per instance
```

**Note:** ID masks are uint16; many viewers show them “black” unless you stretch. (See v2 below for colored/overlay.)

## 5.2 `tools/coco_json_to_instance_masks_v2.py`

**What:** Enhanced converter with filtering + colored labels + overlay previews + 8-bit ID export.

```bash
python tools/coco_json_to_instance_masks_v2.py \
  --coco-ann /.../annotations_images_COCO80.json \
  --coco-res /.../swin_t.segm.json \
  --img-root /.../images \
  --out-dir  /.../masks_v2 \
  --score-thr 0.30 --min-area 50 \
  --idmask-uint8 --save-colored --save-overlay
```

**Tip:** Lower `--score-thr` (e.g., 0.10) and `--min-area` if predictions are sparse/low-score.

## 5.3 `tools/coco_pred_sanity_vis.py`

**What:** Visual sanity check—draws decoded masks + bboxes + scores on original images (random samples).

```bash
python tools/coco_pred_sanity_vis.py \
  --coco-ann /.../annotations_images_COCO80.json \
  --coco-res /.../swin_t.segm.json \
  --img-root /.../images \
  --out-dir  /.../vis_pred \
  --num 12 --score-thr 0.30 --max-inst 100
```

## 5.4 Quick mask check / conversions (one-liners)

* **Inspect a mask quickly**

```bash
python - <<'PY'
import numpy as np, imageio.v2 as iio
p="/ihome/jbwang/liy121/infer/swin_t/masks/gfap_3527_instId.png"
im=iio.imread(p); print(im.dtype, im.shape, im.min(), im.max(), np.unique(im)[:12])
PY
```

* **Batch convert uint16 ID → 8-bit ID for viewing**

```bash
python - <<'PY'
import os,glob,imageio.v2 as iio,numpy as np
src="/ihome/jbwang/liy121/infer/swin_t/masks"; dst=src+"_8bit"; os.makedirs(dst,exist_ok=True)
for p in glob.glob(os.path.join(src,"*_instId.png")):
    im=iio.imread(p); iio.imwrite(os.path.join(dst,os.path.basename(p).replace("_instId","_instId_8bit")),
                                  np.clip(im,0,255).astype('uint8'))
PY
```

* **Binary FG masks from ID maps**

```bash
python - <<'PY'
import os,glob,imageio.v2 as iio,numpy as np
src="/ihome/jbwang/liy121/infer/swin_t/masks"; dst=src+"_binary"; os.makedirs(dst,exist_ok=True)
for p in glob.glob(os.path.join(src,"*_instId.png")):
    im=iio.imread(p); iio.imwrite(os.path.join(dst,os.path.basename(p).replace("_instId","_fg")),
                                  (im>0).astype('uint8')*255)
PY
```

---

# 6) Optional: intensity & channel quick-fix pipeline (to improve quality fast)

## 6.1 Preprocess to **8-bit 3-channel** PNG mirror

```bash
python - <<'PY'
import os,glob,numpy as np
from pathlib import Path
import tifffile as tiff
from PIL import Image
SRC="/ihome/jbwang/liy121/ifimage/processed_dataset/images"
DST="/ihome/jbwang/liy121/ifimage/processed_dataset/images_rgb8bit"
os.makedirs(DST,exist_ok=True)
def to8(x):
    x=x.astype(np.float32); lo,hi=np.percentile(x,1),np.percentile(x,99); hi=hi if hi>lo else lo+1
    return (np.clip((x-lo)/(hi-lo),0,1)*255).astype(np.uint8)
for p in sorted(glob.glob(os.path.join(SRC,"*"))):
    stem=Path(p).stem
    try: arr=tiff.imread(p)
    except: 
        try: arr=np.array(Image.open(p))
        except: continue
    if arr.ndim==3: arr=arr.mean(axis=-1)
    rgb=np.stack([to8(arr)]*3,axis=-1)
    Image.fromarray(rgb).save(os.path.join(DST,stem+".png"))
PY
```

**Why/How:** Normalizes 16-bit to 8-bit and replicates channels to fit 3-ch models.

## 6.2 Make JSON for that mirror & run inference (bigger test scale)

```bash
python - <<'PY'
import os,glob,json
from PIL import Image
IMG="/ihome/jbwang/liy121/ifimage/processed_dataset/images_rgb8bit"
OUT="/ihome/jbwang/liy121/ifimage/processed_dataset/ann_imagesRGB8_COCO80.json"
ims=[{"id":i,"file_name":os.path.basename(p),"width":Image.open(p).size[0],"height":Image.open(p).size[1]}
     for i,p in enumerate(sorted(glob.glob(os.path.join(IMG,"*.png"))),1)]
c80=[...]; cats=[{"id":i+1,"name":n} for i,n in enumerate(c80)]  # same as above
json.dump({"images":ims,"annotations":[],"categories":cats},open(OUT,"w")); print("[OK]",OUT,len(ims))
PY

CONFIG=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py
CKPT=/path/to/matching_3ch_coco_weights.pth   # recommended
DATA=/ihome/jbwang/liy121/ifimage/processed_dataset
OUTP=/ihome/jbwang/liy121/infer/swin_t_rgb8/swin_t
mkdir -p /ihome/jbwang/liy121/infer/swin_t_rgb8

python tools/test.py "$CONFIG" "$CKPT" \
  --format-only \
  --options "jsonfile_prefix=$OUTP" \
  --cfg-options \
    data.test.type=CocoDataset \
    data.test.data_root="$DATA/" \
    data.test.ann_file=ann_imagesRGB8_COCO80.json \
    data.test.img_prefix=images_rgb8bit/ \
    data.test.pipeline.1.img_scale="(1388,1040)" \
    data.test.pipeline.1.keep_ratio=False
```

**Why/How:** This “quick-fix” often improves results immediately by fixing intensity/3-ch input and scale.

---

# 7) Key takeaways (Swin-T today)

* **Use a test JSON that matches the model’s classes.** Your Swin-T ckpt is COCO-80; using `annotations_images_COCO80.json` avoids `IndexError`.
* **Filter out missing images.** `annotations_images_only.json` keeps DataLoader from crashing on absent files.
* **2-ch vs 3-ch matters.** Your log shows `[96,2,4,4]` vs `[96,3,4,4]`. If you keep a 2-ch ckpt with a 3-ch config, first conv is random → poor quality. Either switch to a matching 3-ch ckpt or use a 2-ch config.
* **16-bit intensity vs ImageNet norm.** Convert or normalize appropriately; the 8-bit 3-ch mirror is a practical quick win.
* **Visualization pitfalls.** Instance-ID PNGs are `uint16`; many viewers show them “black.” Use v2 converter’s `--idmask-uint8/--save-colored/--save-overlay` or convert to binary/8-bit for viewing.

That’s the complete Swin-T trail for today—what we wrote, why, and how to run it.



## AUG 16
Set up 3 envs