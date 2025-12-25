# glas_to_coco.py
# Generate COCO instance-seg JSONs for GlaS:
# - train -> train_coco.json
# - testA -> val_coco.json
# - testB -> test_coco.json

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from GlaS_dataset import build_glas_samples  # adjust if your module name differs


# -------------------------
# Instance extraction
# -------------------------
def instances_from_bmp(mask_path: Path):
    """
    Treat non-zero pixel values as instance IDs (0=background).
    Return list of dicts, each containing rle/bbox/area for one instance.
    """
    mask = np.array(Image.open(mask_path).convert("L"))  # HxW uint8
    inst_ids = np.unique(mask)
    inst_ids = inst_ids[inst_ids > 0]  # exclude background

    out = []
    for inst_id in inst_ids:
        binary = (mask == inst_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        rle = mask_utils.encode(np.asfortranarray(binary))
        # pycocotools returns counts as bytes; JSON needs str
        rle["counts"] = rle["counts"].decode("ascii")

        bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]
        area = float(mask_utils.area(rle))

        out.append(
            {
                "rle": rle,
                "bbox": [float(b) for b in bbox],
                "area": area,
            }
        )
    return out


# -------------------------
# COCO builder
# -------------------------
def build_coco_from_samples(samples, images_root: Path, *, iscrowd: int = 0):
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    for s in samples:
        img_path: Path = s.image_path
        seg_mask_path: Path | None = s.target_paths.get("seg_mask")

        if seg_mask_path is None:
            raise ValueError(f"Sample {s.id} has no seg_mask path")
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not seg_mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {seg_mask_path}")

        # Instances -> annotations
        insts = instances_from_bmp(seg_mask_path)
        for inst in insts:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": inst["bbox"],
                    "area": inst["area"],
                    "segmentation": inst["rle"],  # RLE dict
                    "iscrowd": int(iscrowd),
                }
            )
            ann_id += 1

        # Image metadata
        with Image.open(img_path) as im:
            width, height = im.size

        # Safer than img_path.name: preserve subfolders relative to images_root
        try:
            file_name = str(img_path.relative_to(images_root))
        except ValueError:
            # If for some reason img_path isn't under images_root, fall back
            file_name = img_path.name

        grade_glas = s.labels.get("grade_glas")
        grade_sirin = s.labels.get("grade_sirin")

        images.append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "dataset": s.dataset,
                "split": s.split,
                "grade_glas": int(grade_glas) if grade_glas is not None else -1,
                "grade_sirin": int(grade_sirin) if grade_sirin is not None else -1,
            }
        )

        img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "gland"}],
    }


def main():
    # Read-only dataset root (your symlinked read-only data)
    images_root = Path("../data_links/GlaS/GlaS").resolve()
    print("GlaS images root:", images_root)

    # Writable output dir
    out_root = Path("../generated_coco/GlaS").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print("COCO output dir:", out_root)

    # Load all samples
    all_samples = build_glas_samples(images_root)
    print("Total samples:", len(all_samples))

    # Split exactly as you requested
    train_samples = [s for s in all_samples if s.split == "train"]
    val_samples = [s for s in all_samples if s.split == "testA"]
    test_samples = [s for s in all_samples if s.split == "testB"]

    print("Train samples (train):", len(train_samples))
    print("Val samples (testA):", len(val_samples))
    print("Test samples (testB):", len(test_samples))

    # Build COCO dicts
    # NOTE: Some loaders expect RLE -> iscrowd=1. If you see an error about segmentation type,
    # change iscrowd to 1 here.
    train_coco = build_coco_from_samples(train_samples, images_root, iscrowd=0)
    val_coco = build_coco_from_samples(val_samples, images_root, iscrowd=0)
    test_coco = build_coco_from_samples(test_samples, images_root, iscrowd=0)

    # Write JSONs
    train_out = out_root / "train_glas_instance.coco.json"
    val_out = out_root / "val_glas_instance.coco.json"     # from testA
    test_out = out_root / "test_glas_instance.coco.json"   # from testB

    train_out.write_text(json.dumps(train_coco, ensure_ascii=False))
    val_out.write_text(json.dumps(val_coco, ensure_ascii=False))
    test_out.write_text(json.dumps(test_coco, ensure_ascii=False))

    print(f"Wrote {train_out} | images={len(train_coco['images'])}, anns={len(train_coco['annotations'])}")
    print(f"Wrote {val_out}   | images={len(val_coco['images'])}, anns={len(val_coco['annotations'])}")
    print(f"Wrote {test_out}  | images={len(test_coco['images'])}, anns={len(test_coco['annotations'])}")

    # Quick sanity check: show first sample instance count vs unique mask ids
    if len(val_samples) > 0:
        s = val_samples[0]
        mask = np.array(Image.open(s.target_paths["seg_mask"]).convert("L"))
        gt_instances = len(np.unique(mask[mask > 0]))
        json_instances = len(instances_from_bmp(s.target_paths["seg_mask"]))
        print(f"[Sanity] example={s.id} split={s.split} gt_instances={gt_instances} json_instances={json_instances}")


if __name__ == "__main__":
    main()
