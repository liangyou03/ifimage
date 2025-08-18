


# CONFIG=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py
# CKPT=/ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection/latest_swin-t.pth
# DATA=/ihome/jbwang/liy121/ifimage/processed_dataset
# OUTP=/ihome/jbwang/liy121/infer/swin_t/swin_t
# mkdir -p /ihome/jbwang/liy121/infer/swin_t

# # Your repo looks like MMDet 2.x, so use the 2.x style:
# python tools/test.py "$CONFIG" "$CKPT" \
#   --format-only \
#   --options "jsonfile_prefix=$OUTP" \
#   --cfg-options \
#     data.test.type=CocoDataset \
#     data.test.data_root="$DATA/" \
#     data.test.ann_file=annotations_images_COCO80.json \
#     data.test.img_prefix=images/

python tools/coco_json_to_instance_masks.py \
  --coco-ann /ihome/jbwang/liy121/ifimage/processed_dataset/annotations_images_COCO80.json \
  --coco-res /ihome/jbwang/liy121/infer/swin_t/swin_t.segm.json \
  --out-dir  /ihome/jbwang/liy121/infer/swin_t/masks
# Want one binary PNG per instance? add:
#   --per-instance




# ARchive
# # 先在 mmdetection 根目录下
# cd /ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection
# # conda activate openmmlab
# pip install -q pycocotools tifffile pillow

# CONFIG=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py
# CKPT=/ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection/latest_swin-t.pth
# DATA=/ihome/jbwang/liy121/ifimage/processed_dataset
# OUTP=/ihome/jbwang/liy121/infer/swin_t/swin_t
# mkdir -p /ihome/jbwang/liy121/infer/swin_t

# # 这是 MMDet 2.x 的写法（你的配置命名风格就是 2.x）
# python tools/test.py "$CONFIG" "$CKPT" \
#   --format-only \
#   --options "jsonfile_prefix=$OUTP" \
#   --cfg-options \
#     data.test.type=CocoDataset \
#     data.test.data_root="$DATA/" \
#     data.test.ann_file=annotations_cell.json \
#     data.test.img_prefix=images/

# cd /ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection

# CONFIG=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py   # 你本机确实有这个
# CKPT=/ihome/jbwang/liy121/ifimage/Cell-Segmentation-Benchmark/mmdet_BM/mmdetection/latest_swin-t.pth
# DATA=/ihome/jbwang/liy121/ifimage/processed_dataset
# OUTP=/ihome/jbwang/liy121/infer/swin_t/swin_t
# mkdir -p /ihome/jbwang/liy121/infer/swin_t

# # MMDet 2.x 的写法（你的配置风格正是 2.x）
# python tools/test.py "$CONFIG" "$CKPT" \
#   --format-only \
#   --options "jsonfile_prefix=$OUTP" \
#   --cfg-options \
#     data.test.type=CocoDataset \
#     data.test.data_root="$DATA/" \
#     data.test.ann_file=annotations_images_only.json \
#     data.test.img_prefix=images/