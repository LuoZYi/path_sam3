# glas_to_coco.py

import json
from pathlib import Path

import numpy as np
from PIL import Image

from GlaS_dataset import build_glas_samples  # 按你实际模块名来
from pycocotools import mask as mask_utils

def instances_from_bmp(mask_path: Path):
    """
    返回 list[dict]: 每个实例一个 {rle, bbox, area, instance_id}
    """
    mask = np.array(Image.open(mask_path).convert("L"))  # HxW, uint8
    inst_ids = np.unique(mask)
    inst_ids = inst_ids[inst_ids > 0]  # 去掉背景 0

    out = []
    for inst_id in inst_ids:
        binary = (mask == inst_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        rle = mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("ascii")

        bbox = mask_utils.toBbox(rle).tolist()   # [x, y, w, h]
        area = float(mask_utils.area(rle))

        out.append({
            "instance_id": int(inst_id),
            "rle": rle,
            "bbox": [float(b) for b in bbox],
            "area": area,
        })
    return out

def mask_to_rle_from_bmp(mask_path: Path):
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img)
    binary = (mask > 0).astype(np.uint8)

    if binary.sum() == 0:
        return None

    rle = mask_utils.encode(np.asfortranarray(binary))  # 必须 Fortran order
    rle["counts"] = rle["counts"].decode("ascii")       # json 不能存 bytes
    return rle

def mask_to_bbox_from_bmp(mask_path: Path):
    """
    从 GlaS 的 _anno.bmp 里读 mask，计算前景区域的一个整体 bbox
    返回 [x, y, w, h] 或 None（如果没有前景）
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img)

    pos = mask_arr > 0
    if not np.any(pos):
        return None

    ys, xs = np.where(pos)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    return [int(x_min), int(y_min), int(w), int(h)]


def build_coco_from_samples(samples):
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    for s in samples:
        img_path = s.image_path
        seg_mask_path = s.target_paths.get("seg_mask")
        if seg_mask_path is None:
            raise ValueError(f"Sample {s.id} has no seg_mask path")

        #bbox = mask_to_bbox_from_bmp(seg_mask_path)
        # rle = mask_to_rle_from_bmp(seg_mask_path)
        # if rle is not None:
        #     bbox = mask_utils.toBbox(rle).tolist()   # [x, y, w, h]
        #     area = float(mask_utils.area(rle))       # mask 真面积（像素数）

        #     annotations.append({
        #         "id": ann_id,
        #         "image_id": img_id,
        #         "category_id": 1,
        #         "bbox": [float(b) for b in bbox],
        #         "area": area,
        #         "segmentation": rle,   # ✅ RLE segmentation
        #         "iscrowd": 1,          # ✅ RLE 按 COCO 规范通常设 1
        #     })
        #     ann_id += 1
        insts = instances_from_bmp(seg_mask_path)
        for inst in insts:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": inst["bbox"],
                "area": inst["area"],
                "segmentation": inst["rle"],
                "iscrowd": 0,  # 单个实例通常 0
                # 可选：方便 debug/追踪
                "instance_id": inst["instance_id"],
            })
            ann_id += 1

        with Image.open(img_path) as im:
            width, height = im.size

        grade_glas = s.labels.get("grade_glas")
        grade_sirin = s.labels.get("grade_sirin")

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
                "dataset": s.dataset,
                "split": s.split,
                "grade_glas": int(grade_glas) if grade_glas is not None else -1,
                "grade_sirin": int(grade_sirin) if grade_sirin is not None else -1,
            }
        )
        
        # if bbox is not None:
        #     x, y, w, h = bbox
        #     annotations.append(
        #         {
        #             "id": ann_id,
        #             "image_id": img_id,
        #             "category_id": 1,
        #             "bbox": [x, y, w, h],
        #             "area": int(w * h),
        #             "iscrowd": 0,
        #         }
        #     )
        #     ann_id += 1

        img_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "gland"},
        ],
    }
    return coco_dict


def main():
    # 只读数据根目录：继续用 data_links
    images_root = Path("../data_links/GlaS/GlaS").resolve()
    print("GlaS images root:", images_root)

    # ✅ 可写输出目录：放在你自己的 home/pipeline 下面
    out_root = Path("../generated_coco/GlaS").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print("COCO output dir:", out_root)

    # 1) 读取所有样本（只读盘）
    all_samples = build_glas_samples(images_root)
    print("Total samples:", len(all_samples))

    train_samples = [s for s in all_samples if (s.split == "train" or s.split == "testA")]
    val_samples = [s for s in all_samples if s.split == "testB"]
    
    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))

    # 2) 生成 COCO
    train_coco = build_coco_from_samples(train_samples)
    val_coco = build_coco_from_samples(val_samples)

    # 3) 写到可写目录中
    train_out = out_root / "train_glas_seg.coco.json"
    val_out = out_root / "val_glas_seg.coco.json"

    train_out.write_text(json.dumps(train_coco))
    val_out.write_text(json.dumps(val_coco))

    print(
        f"Wrote {train_out} | images={len(train_coco['images'])}, anns={len(train_coco['annotations'])}"
    )
    print(
        f"Wrote {val_out} | images={len(val_coco['images'])}, anns={len(val_coco['annotations'])}"
    )


if __name__ == "__main__":
    main()
