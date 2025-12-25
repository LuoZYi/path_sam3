# pannuke_to_coco.py
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image

# ✅ 按你的文件名来：如果你的文件就是 pannuke_dataset.py
from pannuke_dataset import build_pannuke_samples


# ----------------------------
# 配置区
# ----------------------------
# 输出两种模式：
#   semantic: 每张图每个类别一个 union mask（推荐，JSON 不会爆炸）
#   instance: 每个 nucleus 一个 annotation（非常大，慎用）
MODE = "semantic"  # or "instance"

# folds 划分
TRAIN_FOLDS = ("fold0", "fold1")
VAL_FOLDS = ("fold2",)

# data_links 根（用于写相对路径，避免多数据集 file_name 撞名）
DATA_LINKS_ROOT = Path("../data_links").resolve()

# 细胞类别（PanNuke常见5类）
# 注意：你自己的 type_map 如果是 0-4 或 1-5，这里下面会自动适配
NUCLEI_CATEGORIES = [
    (1, "Neoplastic"),
    (2, "Inflammatory"),
    (3, "Connective"),
    (4, "Dead"),
    (5, "Epithelial"),
]


# ----------------------------
# COCO RLE 需要 pycocotools
# ----------------------------
try:
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError(
        "需要安装 pycocotools 才能输出 COCO RLE segmentation：\n"
        "pip install pycocotools\n"
    ) from e


def _to_rel_file(p: Path) -> str:
    """把绝对路径转成相对 data_links 的路径，避免合并多数据集时 file_name 冲突"""
    p = p.resolve()
    try:
        return str(p.relative_to(DATA_LINKS_ROOT))
    except Exception:
        # 兜底：如果不在 data_links 下，就用文件名
        return p.name


def _encode_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """binary_mask: (H,W) uint8 0/1 -> COCO RLE dict"""
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    # pycocotools 会返回 bytes counts，json 不能直接存
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _safe_mode_label(vals: np.ndarray) -> int:
    """返回 vals 的众数（用于 instance -> 类别推断）"""
    if vals.size == 0:
        return 0
    vals = vals.astype(np.int64)
    # 过滤负值（如果有）
    vals = vals[vals >= 0]
    if vals.size == 0:
        return 0
    uniq, cnt = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(cnt)])


def load_inst_and_type_map(npy_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    尽可能从你的 .npy 里解析出：
      inst_map: (H,W) 每个 nucleus 一个 id（0 背景）
      type_map: (H,W) 每个像素一个类别（0 背景，其它为细胞类别）
    你的数据格式可能不同：dict / ndarray / object array，都做了兼容。
    """
    raw = np.load(npy_path, allow_pickle=True)

    obj = raw
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.size == 1:
            obj = obj.item()
        else:
            # 极少见：object array 多元素，直接变 list
            obj = list(obj)

    inst_map = None
    type_map = None

    # 情况 1：dict
    if isinstance(obj, dict):
        # 常见 key
        for k in ("inst_map", "instance_map", "inst", "instances"):
            if k in obj:
                inst_map = np.asarray(obj[k])
                break

        for k in ("type_map", "class_map", "types", "type", "classes"):
            if k in obj:
                type_map = np.asarray(obj[k])
                break

        # 有些人把 mask 直接放在 "mask" 里
        if inst_map is None and "mask" in obj:
            arr = np.asarray(obj["mask"])
            if arr.ndim == 2:
                # 猜测：如果最大值大于 20，基本就是 instance id；否则可能是 type map
                if arr.max() > 20:
                    inst_map = arr
                else:
                    type_map = arr
            elif arr.ndim == 3:
                # 猜测：channel-last 或 channel-first
                if arr.shape[-1] >= 2:
                    inst_map = arr[..., 0]
                    type_map = arr[..., 1]
                elif arr.shape[0] >= 2:
                    inst_map = arr[0]
                    type_map = arr[1]

    # 情况 2：数值 ndarray
    elif isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.ndim == 2:
            # 猜测：max 很大通常是 instance id；max 小通常是 type
            if arr.max() > 20:
                inst_map = arr
            else:
                type_map = arr
        elif arr.ndim == 3:
            # 猜测：channel-last 或 channel-first
            if arr.shape[-1] >= 2:
                inst_map = arr[..., 0]
                type_map = arr[..., 1]
            elif arr.shape[0] >= 2:
                inst_map = arr[0]
                type_map = arr[1]

    # squeeze 兜底
    if inst_map is not None and inst_map.ndim > 2:
        inst_map = np.squeeze(inst_map)
    if type_map is not None and type_map.ndim > 2:
        type_map = np.squeeze(type_map)

    return inst_map, type_map


def normalize_type_id(t: int) -> int:
    """
    把 type_map 的类别值规范到 1..5（背景0）
    兼容两种常见标注：
      - 0..4（0背景？或0第一类） -> 转成 1..5（保留0背景）
      - 1..5 直接用
    """
    if t <= 0:
        return 0
    # 如果最大只到4，很可能是 0..4 体系 -> +1
    # 这里用一个简单规则：t in [1..4] 并且没有 5 时也可能是 0..4，
    # 但我们在 build 时会用全图 max 来判断一次更稳（见下面）。
    return t


def build_coco_from_samples(samples, mode: str = "semantic") -> Dict[str, Any]:
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    # categories
    categories = [{"id": cid, "name": name} for cid, name in NUCLEI_CATEGORIES]
    # 如果你想做单类（所有核都当 1 类），可以改成：
    # categories = [{"id": 1, "name": "nucleus"}]

    for s in samples:
        img_path = s.image_path
        npy_path = s.target_paths.get("instance_mask_npy")
        if npy_path is None:
            raise ValueError(f"Sample {s.id} missing instance_mask_npy")

        with Image.open(img_path) as im:
            width, height = im.size

        # image-level meta
        tissue_type = s.labels.get("tissue_type", None)
        proxy_malignancy = s.labels.get("proxy_malignancy", None)
        cell_counts = s.extras.get("cell_counts", None)
        if isinstance(cell_counts, np.ndarray):
            cell_counts = cell_counts.astype(int).tolist()

        images.append(
            {
                "id": img_id,
                "file_name": _to_rel_file(img_path),
                "width": int(width),
                "height": int(height),
                "dataset": s.dataset,
                "split": s.split,
                "tissue_type": int(tissue_type) if tissue_type is not None else -1,
                "proxy_malignancy": int(proxy_malignancy) if proxy_malignancy is not None else -1,
                "cell_counts": cell_counts if cell_counts is not None else None,
                "tissue_type_str": s.extras.get("tissue_type_str"),
            }
        )

        inst_map, type_map = load_inst_and_type_map(npy_path)

        # 先用 type_map 的全图 max 推断是否 0..4 体系
        type_plus_one = False
        if type_map is not None:
            tmax = int(np.max(type_map))
            # 常见：0..4 或 0..5（背景0）
            if tmax == 4:
                type_plus_one = True

        # ----------------------------
        # semantic 模式（推荐）
        # ----------------------------
        if mode == "semantic":
            # 优先用 type_map 做 5 类 union mask；没有 type_map 就用 inst_map union
            if type_map is not None:
                # 遍历 1..5 类（兼容 0..4 -> +1）
                for cid, _name in NUCLEI_CATEGORIES:
                    src_c = cid - 1 if type_plus_one else cid  # 如果 type_map 是 0..4，就用 cid-1
                    if src_c <= 0:
                        continue
                    binary = (type_map == src_c).astype(np.uint8)
                    if binary.sum() == 0:
                        continue

                    rle = _encode_rle(binary)
                    bbox = mask_utils.toBbox(rle).tolist()  # [x,y,w,h]
                    area = float(mask_utils.area(rle))

                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(cid),
                            "bbox": [float(x) for x in bbox],
                            "area": area,
                            "segmentation": rle,
                            "iscrowd": 1,
                        }
                    )
                    ann_id += 1

            elif inst_map is not None:
                # 没 type_map：把所有 instance union 当成 1 类（这里用 category 1）
                binary = (inst_map > 0).astype(np.uint8)
                if binary.sum() > 0:
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
                        }
                    )
                    ann_id += 1

            else:
                raise ValueError(f"Cannot parse mask from {npy_path} (no inst_map/type_map)")

        # ----------------------------
        # instance 模式（非常大，慎用）
        # ----------------------------
        elif mode == "instance":
            if inst_map is None:
                raise ValueError(f"Instance mode requires inst_map, but got None for {npy_path}")

            inst_ids = np.unique(inst_map)
            inst_ids = inst_ids[inst_ids > 0]

            for iid in inst_ids:
                binary = (inst_map == iid).astype(np.uint8)
                if binary.sum() == 0:
                    continue

                # 推断类别：用 type_map 在该 instance 内的众数
                cid = 1
                if type_map is not None:
                    vals = type_map[binary.astype(bool)]
                    t = _safe_mode_label(vals)
                    if type_plus_one:
                        t = t + 1
                    # 限制到 1..5
                    if 1 <= t <= 5:
                        cid = int(t)

                rle = _encode_rle(binary)
                bbox = mask_utils.toBbox(rle).tolist()
                area = float(mask_utils.area(rle))

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": [float(x) for x in bbox],
                        "area": area,
                        "segmentation": rle,
                        "iscrowd": 1,
                        # 可选：保留 instance id 方便 debug
                        "instance_id": int(iid),
                    }
                )
                ann_id += 1

        else:
            raise ValueError(f"Unknown mode: {mode}")

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco


def main():
    root = Path("../data_links/PanNuke").resolve()
    out_root = Path("../generated_coco/PanNuke").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("PanNuke root:", root)
    print("COCO out:", out_root)
    print("MODE:", MODE)

    # 读取样本
    all_train = build_pannuke_samples(root, folds=TRAIN_FOLDS)
    all_val = build_pannuke_samples(root, folds=VAL_FOLDS)

    print("Train samples:", len(all_train), "folds:", TRAIN_FOLDS)
    print("Val samples:", len(all_val), "folds:", VAL_FOLDS)

    train_coco = build_coco_from_samples(all_train, mode=MODE)
    val_coco = build_coco_from_samples(all_val, mode=MODE)

    train_out = out_root / f"train_pannuke_{MODE}.coco.json"
    val_out = out_root / f"val_pannuke_{MODE}.coco.json"

    train_out.write_text(json.dumps(train_coco, ensure_ascii=False))
    val_out.write_text(json.dumps(val_coco, ensure_ascii=False))

    print(f"Wrote {train_out} | images={len(train_coco['images'])}, anns={len(train_coco['annotations'])}")
    print(f"Wrote {val_out} | images={len(val_coco['images'])}, anns={len(val_coco['annotations'])}")


if __name__ == "__main__":
    main()
