# (re-enter a GPU session if the previous one expired)
salloc --cluster=gpu --partition=preempt --gres=gpu:1 --mem=32G --time=06:00:00
conda activate openmmlab
pip install -q pycocotools tifffile pillow

python - <<'PY'
import os, glob, json
from PIL import Image

DATA = "/ihome/jbwang/liy121/ifimage/processed_dataset"
IMG = os.path.join(DATA, "images")
out = os.path.join(DATA, "annotations_images_COCO80.json")

# 1) list real images and their sizes
images = []
for i, p in enumerate(sorted(sum([glob.glob(os.path.join(IMG, f"*.{e}"))
                                   for e in ("tiff","tif","png","jpg","jpeg")], [])), 1):
    try:
        with Image.open(p) as im:
            w, h = im.size
        images.append({"id": i, "file_name": os.path.basename(p), "width": w, "height": h})
    except Exception as e:
        print("[WARN] skip:", p, e)

# 2) dummy COCO-80 categories (ids 1..80; names are standard but ids simplified)
coco80 = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog",
"horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
"baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
"knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
"pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv",
"laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
"refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
categories = [{"id": i+1, "name": n} for i, n in enumerate(coco80)]

coco = {"images": images, "annotations": [], "categories": categories}
with open(out, "w") as f: json.dump(coco, f)
print(f"[OK] wrote {out} with {len(images)} images and {len(categories)} categories")
PY
