# import json
# import os
# from collections import defaultdict

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# from pycocotools import mask as maskUtils


# def rle_to_mask(segmentation, h=None, w=None):
#     """
#     segmentation: either {"size":[h,w],"counts":...} or polygon etc.
#     here we assume COCO RLE dict.
#     """
#     rle = segmentation
#     # pycocotools expects counts to be bytes in some cases; it can also handle str.
#     m = maskUtils.decode(rle)  # (h,w,1) or (h,w)
#     if m.ndim == 3:
#         m = m[:, :, 0]
#     return m.astype(np.uint8)


# def overlay_instances(img_bgr, instances, alpha=0.45):
#     """
#     instances: list of dicts with keys: mask (H,W uint8 0/1), score (float)
#     """
#     out = img_bgr.copy()
#     H, W = out.shape[:2]

#     # 给每个 instance 一个颜色（随机但稳定一点）
#     rng = np.random.default_rng(12345)

#     for inst in instances:
#         m = inst["mask"]
#         if m.shape != (H, W):
#             # 如果尺寸不一致，强制 resize（一般不应该发生，除非 size 写错或图被 resize 过）
#             m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

#         color = rng.integers(0, 255, size=3, dtype=np.uint8).tolist()  # BGR
#         colored = np.zeros_like(out, dtype=np.uint8)
#         colored[m > 0] = color

#         out = cv2.addWeighted(out, 1.0, colored, alpha, 0)

#         # 画轮廓
#         contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(out, contours, -1, color, 2)

#     return out


# def visualize_dump_on_val(
#     ann_json_path,
#     img_root,
#     pred_dump_path,
#     out_dir,
#     score_thr=0.5,
#     topk_per_image=100,
#     max_images=50
# ):
#     os.makedirs(out_dir, exist_ok=True)

#     coco = COCO(ann_json_path)
#     with open(pred_dump_path, "r", encoding="utf-8") as f:
#         preds = json.load(f)

#     preds_by_img = defaultdict(list)
#     for p in preds:
#         preds_by_img[p["image_id"]].append(p)

#     img_ids = list(preds_by_img.keys())
#     img_ids = img_ids[:max_images]

#     for img_id in img_ids:
#         img_info = coco.loadImgs([img_id])[0]
#         file_name = img_info["file_name"]
#         img_path = os.path.join(img_root, file_name)

#         img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         if img_bgr is None:
#             print(f"[WARN] cannot read {img_path}")
#             continue

#         # 过滤 + 排序
#         ps = [p for p in preds_by_img[img_id] if p.get("score", 1.0) >= score_thr]
#         ps.sort(key=lambda x: x.get("score", 1.0), reverse=True)
#         ps = ps[:topk_per_image]

#         instances = []
#         for p in ps:
#             seg = p["segmentation"]
#             m = rle_to_mask(seg)
#             instances.append({"mask": m, "score": p.get("score", 1.0)})

#         vis = overlay_instances(img_bgr, instances, alpha=0.45)

#         # 右上角写一下分数
#         y = 25
#         for i, p in enumerate(ps):
#             cv2.putText(
#                 vis,
#                 f"#{i} score={p.get('score', 0):.3f}",
#                 (10, y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2,
#                 cv2.LINE_AA,
#             )
#             y += 25

#         out_path = os.path.join(out_dir, f"{img_id}_{os.path.basename(file_name)}")
#         cv2.imwrite(out_path, vis)
#         print("[OK]", out_path)


# # ====== 用法示例（把路径改成你自己的）======
# if __name__ == "__main__":
#     ann_json_path = "../generated_coco/GlaS/val_glas_instance.coco.json"      # COCO val annotation
#     img_root = "./Glas_new/val/images"                   # val images folder
#     pred_dump_path = "../experiments/glas_seg_instance/dumps/glas/coco_predictions_segm.json"         # 你的 dump（list of dict）
#     out_dir = "./vis_out"

#     visualize_dump_on_val(
#         ann_json_path=ann_json_path,
#         img_root=img_root,
#         pred_dump_path=pred_dump_path,
#         out_dir=out_dir,
#         score_thr=0.5,
#         topk_per_image=100,
#         max_images=500
#     )





import os
import json
import numpy as np
import cv2
from collections import defaultdict

# ========= 你需要改的 =========
IMG_ROOT  = r"./Glas_new/val/images"         # 里面是 .bmp
GT_JSON   = r"../generated_coco/GlaS/val_glas_instance.coco.json"  # COCO GT
PRED_JSON = r"../experiments/glas_seg_instance/dumps/glas/coco_predictions_segm.json"    # 你的 dump（COCO detection/seg style）
OUT_DIR   = r"./side_by_side"
SCORE_TH  = 0.5
MAX_SHOW  = 100  # 最多可视化多少张，避免一次出太多
# ============================

os.makedirs(OUT_DIR, exist_ok=True)

def poly_to_mask(polys, H, W):
    mask = np.zeros((H, W), np.uint8)
    if not isinstance(polys, list):
        return mask
    # polys: [ [x1,y1,x2,y2,...], [..], ... ]
    for p in polys:
        if len(p) < 6:
            continue
        pts = np.array(p, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask

def rle_to_mask(rle, H, W):
    # 支持两种常见形式：
    # 1) {"counts": [...], "size":[H,W]} (uncompressed)
    # 2) {"counts": "encoded_str", "size":[H,W]} (pycocotools compressed) -> 需要 pycocotools
    counts = rle.get("counts", None)
    size = rle.get("size", [H, W])
    if isinstance(counts, list):
        # uncompressed RLE
        h, w = size
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0
        for c in counts:
            if c > 0:
                flat[idx:idx+c] = val
            idx += c
            val = 1 - val
        mask = flat.reshape((w, h)).T  # COCO RLE 是按列展开
        return mask.astype(np.uint8)
    elif isinstance(counts, str):
        # compressed RLE -> pycocotools
        try:
            from pycocotools import mask as maskUtils
            return maskUtils.decode(rle).astype(np.uint8)
        except Exception as e:
            raise RuntimeError("你的 RLE 是压缩格式，需要安装 pycocotools：pip install pycocotools") from e
    else:
        return np.zeros((H, W), np.uint8)

def seg_to_mask(seg, H, W):
    if seg is None:
        return np.zeros((H, W), np.uint8)
    if isinstance(seg, list):
        return poly_to_mask(seg, H, W)
    if isinstance(seg, dict):
        return rle_to_mask(seg, H, W)
    return np.zeros((H, W), np.uint8)

def overlay(img_bgr, mask, color=(0, 255, 0), alpha=0.45):
    out = img_bgr.copy()
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask3 = np.stack([mask]*3, axis=-1)
    color_img = np.zeros_like(out)
    color_img[:] = color
    out = np.where(mask3 > 0,
                   (out * (1 - alpha) + color_img * alpha).astype(np.uint8),
                   out)
    # 画一下轮廓更清楚
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    return out

# ---- load COCO gt ----
gt = json.load(open(GT_JSON, "r", encoding="utf-8"))
imgs = {im["id"]: im for im in gt["images"]}
gt_anns_by_img = defaultdict(list)
for a in gt["annotations"]:
    gt_anns_by_img[a["image_id"]].append(a)

# ---- load preds ----
preds = json.load(open(PRED_JSON, "r", encoding="utf-8"))
preds_by_img = defaultdict(list)
for p in preds:
    preds_by_img[p["image_id"]].append(p)

# ---- main ----
count = 0
for image_id, im in imgs.items():
    file_name = im["file_name"]
    img_path = os.path.join(IMG_ROOT, file_name)
    if not os.path.exists(img_path):
        # 如果 COCO file_name 不是 bmp，可尝试强制改后缀
        alt = os.path.splitext(img_path)[0] + ".bmp"
        if os.path.exists(alt):
            img_path = alt
        else:
            continue

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        continue
    H, W = img.shape[:2]

    # build GT union mask
    gt_union = np.zeros((H, W), np.uint8)
    for a in gt_anns_by_img.get(image_id, []):
        m = seg_to_mask(a.get("segmentation"), H, W)
        gt_union = np.maximum(gt_union, m)

    # build Pred union mask (filter by score)
    pred_union = np.zeros((H, W), np.uint8)
    for p in preds_by_img.get(image_id, []):
        if "score" in p and p["score"] < SCORE_TH:
            continue
        m = seg_to_mask(p.get("segmentation"), H, W)
        pred_union = np.maximum(pred_union, m)

    left  = overlay(img, gt_union,  color=(0, 255, 0), alpha=0.45)   # GT green
    right = overlay(img, pred_union, color=(0, 0, 255), alpha=0.45)   # Pred red

    # title
    cv2.putText(left,  "GT",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(right, "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    merged = np.concatenate([left, right], axis=1)
    out_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(file_name))[0] + "_gt_pred.png")
    cv2.imwrite(out_path, merged)

    count += 1
    if count >= MAX_SHOW:
        break

print(f"Saved {count} side-by-side images to: {OUT_DIR}")
