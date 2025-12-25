# breast_to_coco.py
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from PIL import Image

from breast_dataset import build_breast_cancer_samples  # 你的文件名/函数名按这里改

try:
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError(
        "需要安装 pycocotools 才能输出 COCO RLE segmentation：\n"
        "pip install pycocotools\n"
    ) from e


# ----------------------------
# 配置
# ----------------------------
MODE = "semantic"  # "semantic" or "instance"
VAL_RATIO = 0.2    # 20% 做 val（因为你现在 build_breast_cancer_samples 全部 split=train）
MIN_AREA = 10      # 过滤太小的前景（像素数）

DATA_LINKS_ROOT = Path("../data_links").resolve()

CATEGORIES = [
    {"id": 1, "name": "cell"},  # 你也可以改成 "tumor_cell" / "nucleus" 等
]


def _to_rel_file(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(DATA_LINKS_ROOT))
    except Exception:
        return p.name


def _encode_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    binary_mask: (H,W) uint8 0/1
    return: COCO RLE dict
    """
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _load_mask(mask_path: Path) -> np.ndarray:
    """
    读 TIF mask -> (H,W) int
    强制转 L，避免 RGB / palette 坑
    """
    m = Image.open(mask_path).convert("L")
    arr = np.array(m)
    return arr


def _deterministic_is_val(sample_id: str, val_ratio: float) -> bool:
    """
    用 id hash 做确定性划分：同一份数据每次划分一致
    """
    h = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    x = int(h[:8], 16) % 10000
    return x < int(val_ratio * 10000)


def _mask_to_instances(mask_arr: np.ndarray) -> np.ndarray:
    """
    尝试把 mask 变成 inst_map：
      - 如果 mask 本身就有多种取值（>2 且 max>1），认为它是实例/标签图，直接用
      - 否则把二值前景做连通域 -> inst_map
    """
    uniq = np.unique(mask_arr)

    # 典型二值：{0,255} or {0,1}
    if uniq.size > 2 and mask_arr.max() > 1:
        # 可能已经是 instance id / multi-label
        inst_map = mask_arr.astype(np.int32)
        # 把非零映射成连续 id（可选）
        ids = np.unique(inst_map)
        ids = ids[ids > 0]
        remap = {old: i + 1 for i, old in enumerate(ids)}
        out = np.zeros_like(inst_map, dtype=np.int32)
        for old, new in remap.items():
            out[inst_map == old] = new
        return out

    # 否则做连通域（需要 opencv）
    binary = (mask_arr > 0).astype(np.uint8)
    try:
        import cv2
        num, labels = cv2.connectedComponents(binary, connectivity=8)
        # labels: 0..num-1
        return labels.astype(np.int32)
    except Exception:
        # 没 cv2 或失败：退化成一个实例（union）
        return binary.astype(np.int32)


def build_coco(samples: List, mode: str) -> Dict[str, Any]:
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    for s in samples:
        img_path = s.image_path
        mask_path = s.target_paths.get("seg_mask")
        if mask_path is None or not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for sample {s.id}: {mask_path}")

        with Image.open(img_path) as im:
            width, height = im.size

        images.append(
            {
                "id": img_id,
                "file_name": _to_rel_file(img_path),
                "width": int(width),
                "height": int(height),

                # 额外 meta：不会影响 COCO 标准字段
                "dataset": s.dataset,
                "split": s.split,
                "status": int(s.labels.get("status")) if s.labels.get("status") is not None else -1,
                "status_str": s.extras.get("status_str"),
            }
        )

        mask_arr = _load_mask(mask_path)

        if mode == "semantic":
            binary = (mask_arr > 0).astype(np.uint8)
            if int(binary.sum()) < MIN_AREA:
                img_id += 1
                continue

            rle = _encode_rle(binary)
            bbox = mask_utils.toBbox(rle).tolist()  # [x,y,w,h]
            area = float(mask_utils.area(rle))

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x) for x in bbox],
                    "area": area,
                    "segmentation": rle,
                    "iscrowd": 1,
                    # 方便回溯
                    "mask_file": _to_rel_file(mask_path),
                }
            )
            ann_id += 1

        elif mode == "instance":
            inst_map = _mask_to_instances(mask_arr)
            inst_ids = np.unique(inst_map)
            inst_ids = inst_ids[inst_ids > 0]

            for iid in inst_ids:
                binary = (inst_map == iid).astype(np.uint8)
                if int(binary.sum()) < MIN_AREA:
                    continue

                rle = _encode_rle(binary)
                bbox = mask_utils.toBbox(rle).tolist()
                area = float(mask_utils.area(rle))

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x) for x in bbox],
                        "area": area,
                        "segmentation": rle,
                        "iscrowd": 1,
                        "instance_id": int(iid),
                        "mask_file": _to_rel_file(mask_path),
                    }
                )
                ann_id += 1

        else:
            raise ValueError(f"Unknown MODE: {mode}")

        img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def main():
    root = Path("../data_links/BreastCancer").resolve()
    out_root = Path("../generated_coco/BreastCancer").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("BreastCancer root:", root)
    print("COCO out:", out_root)
    print("MODE:", MODE, "| VAL_RATIO:", VAL_RATIO)

    all_samples = build_breast_cancer_samples(root)
    print("Total samples:", len(all_samples))

    # 你现在 samples 里 split 全是 "train"，所以这里做一个确定性 train/val 划分
    train_samples = [s for s in all_samples if not _deterministic_is_val(s.id, VAL_RATIO)]
    val_samples = [s for s in all_samples if _deterministic_is_val(s.id, VAL_RATIO)]

    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))

    train_coco = build_coco(train_samples, MODE)
    val_coco = build_coco(val_samples, MODE)

    train_out = out_root / f"train_breast_{MODE}.coco.json"
    val_out = out_root / f"val_breast_{MODE}.coco.json"

    train_out.write_text(json.dumps(train_coco, ensure_ascii=False))
    val_out.write_text(json.dumps(val_coco, ensure_ascii=False))

    print(f"Wrote {train_out} | images={len(train_coco['images'])}, anns={len(train_coco['annotations'])}")
    print(f"Wrote {val_out} | images={len(val_coco['images'])}, anns={len(val_coco['annotations'])}")


if __name__ == "__main__":
    main()
