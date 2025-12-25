# import os
# import yaml
# from torch.utils.data import Dataset
# from PIL import Image

# class SimpleImageDataset(Dataset):
#     def __init__(self, dataset_name, split="train"):
#         cfg = yaml.safe_load(open("configs/datasets.yaml"))
#         root = cfg["datasets"][dataset_name]["root"]
#         self.image_dir = os.path.join(root, "images") 
#         self.images = sorted(os.listdir(self.image_dir))

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.images[idx])
#         img = Image.open(img_path).convert("RGB")
#         return img







# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Optional, Dict, Any

# @dataclass
# class SampleEntry:
#     # Core identity
#     id: str                      # e.g. "ytma10_010704_benign1_ccd", "testA_1", "0_0"
#     dataset: str                 # e.g. "BreastCancer", "GlaS", "PanNuke"
#     split: Optional[str] = None  # e.g. "train", "val", "testA", "fold1"

#     # Files on disk
#     image_path: Path = Path()
#     target_paths: Dict[str, Path] = field(default_factory=dict)
#     # e.g. {
#     #   "seg_mask": Path(...),
#     #   "npy_mask": Path(...),
#     #   "grade_mask": Path(...),
#     # }

#     meta_paths: Dict[str, Path] = field(default_factory=dict)
#     # e.g. {
#     #   "image_xml": Path(...),
#     #   "mask_xml": Path(...),
#     # }

#     # Already-parsed / numeric stuff (optional, can be filled by indexer)
#     labels: Dict[str, Any] = field(default_factory=dict)
#     # e.g. {
#     #   "status": 0,                    # benign/malignant
#     #   "grade_glas": 1,                # benign/malignant
#     #   "tissue_type": 3,               # Breast, Colon, etc.
#     # }

#     extras: Dict[str, Any] = field(default_factory=dict)
#     # e.g. {
#     #   "patient_id": 4,
#     #   "cell_counts": np.array([...]), # per-class counts
#     # }

# @dataclass
# class DatasetContext:
#     name: str
#     root: Path
#     label_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
#     # e.g. {
#     #   "tissue_types": {"Breast": 3, "Colon": 5, ...},
#     #   "nuclei_types": {"Neoplastic": 1, ...},
#     #   "grade_glas": {"benign": 0, "malignant": 1},
#     # }

#     class_weights: Dict[str, Dict[int, float]] = field(default_factory=dict)
#     # e.g. {"tissue_types": {0: 1.3, 1: 0.7, ...}}

#     # any dataset-wide dataframes/CSVs you want to reuse
#     tables: Dict[str, Any] = field(default_factory=dict)
#     # e.g. {"grades": grades_df, "cell_counts": cell_counts_df}



# SampleEntry(
#     id="ytma10_010704_benign1_ccd",
#     dataset="BreastCancer",
#     split="train",  # or None if you don't have split

#     image_path=Path("../data_links/BreastCancer/BreastCancerCells/Images/ytma10_010704_benign1_ccd.tif"),
#     target_paths={
#         "seg_mask": Path("../data_links/BreastCancer/BreastCancerCells/Masks/ytma10_010704_benign1_ccd_mask.TIF")
#     },
#     meta_paths={
#         "image_xml": Path("../data_links/BreastCancer/BreastCancerCells/Images/ytma10_010704_benign1_ccd.tif.xml"),
#         "mask_xml": Path("../data_links/BreastCancer/BreastCancerCells/Masks/ytma10_010704_benign1_ccd_mask.TIF.xml"),
#     },
#     labels={
#         "status": 0,  # from status field in XML, mapped via context.label_maps["status"]
#     },
#     extras={}
# )


# breast_dataset.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

import xml.etree.ElementTree as ET

@dataclass
class SampleEntry:
    id: str
    dataset: str
    split: str

    image_path: Path
    target_paths: Dict[str, Path] = field(default_factory=dict)
    meta_paths: Dict[str, Path] = field(default_factory=dict)

    labels: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


def parse_status_from_xml(xml_path: Path) -> str | None:
    """
    Parse <tag name="status" value="..."/> from the image XML.
    Returns the raw string (e.g. 'benign', 'malignant') or None.
    """
    if not xml_path.exists():
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    status_tag = root.find(".//tag[@name='status']")
    if status_tag is None:
        return None

    return status_tag.attrib.get("value")


def build_breast_cancer_samples(root: Path) -> List[SampleEntry]:
    """
    root: ../data_links/BreastCancer

    This will look into:
      root / 'BreastCancerCells' / 'Images'
      root / 'BreastCancerCells' / 'Masks'
    and build a list of SampleEntry objects.
    """
    cells_root = root / "BreastCancerCells"
    images_dir = cells_root / "Images"
    masks_dir = cells_root / "Masks"

    samples: List[SampleEntry] = []

    # Map string status to int label (you can adjust later)
    status_map = {
        "benign": 0,
        "malignant": 1,
    }

    # Iterate all .tif images (ignore .tif.xml)
    for img_path in sorted(images_dir.glob("*.tif")):
        # Skip the xml files themselves, in case glob picks them
        if img_path.name.lower().endswith(".tif.xml"):
            continue
        
        # e.g. ytma10_010704_benign1_ccd.tif -> ytma10_010704_benign1_ccd
        stem = img_path.stem
        real_stem = stem[:-4] if stem.endswith("_ccd") else stem
        # Build expected paths
        image_xml_path = img_path.with_suffix(img_path.suffix + ".xml")

        # Mask name: stem + "_mask.TIF" (case-sensitive; adjust if needed)
        mask_path = masks_dir / f"{real_stem}.TIF"
        # Some datasets might use .tif instead of .TIF; add a fallback:
        if not mask_path.exists():
            mask_path = masks_dir / f"{stem}_mask.tif"

        mask_xml_path = mask_path.with_suffix(mask_path.suffix + ".xml")

        # Extract id and status
        sample_id = stem  # or you can strip further if you like
        status_str = parse_status_from_xml(image_xml_path)
        status_label = None
        if status_str is not None:
            status_label = status_map.get(status_str.lower(), None)

        sample = SampleEntry(
            id=sample_id,
            dataset="BreastCancer",
            split="train",  # for now, just call everything train

            image_path=img_path,
            target_paths={
                "seg_mask": mask_path,
            },
            meta_paths={
                "image_xml": image_xml_path,
                "mask_xml": mask_xml_path,
            },
            labels={
                "status": status_label,
            },
            extras={
                "status_str": status_str,
            },
        )

        samples.append(sample)

    return samples







from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class BreastCancerDataset(Dataset):
    def __init__(self, samples: List[SampleEntry], transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # --- load image ---
        img = Image.open(s.image_path)

        # --- load mask ---
        mask = None
        seg_mask_path = s.target_paths.get("seg_mask")
        if seg_mask_path is not None and seg_mask_path.exists():
            mask = Image.open(seg_mask_path)
        else:
            # Optional: raise if you expect every sample to have mask
            raise FileNotFoundError(f"Mask not found for sample {s.id}: {seg_mask_path}")
            mask = None

        # --- apply transforms / convert to tensors ---
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            arr = np.array(img).astype("float32")
            if arr.ndim == 2:  # grayscale
                arr = arr[None, ...]
            else:
                arr = arr.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(arr) / 255.0

        mask_tensor = None
        if mask is not None:
            if self.target_transform is not None:
                mask_tensor = self.target_transform(mask)
            else:
                mask_tensor = torch.from_numpy(np.array(mask).astype("int64"))

        # --- status label: make sure it's not None ---
        status_label = s.labels.get("status")
        if status_label is None:
            raise ValueError(f"Missing status label for sample {s.id} (xml: {s.meta_paths.get('image_xml')})")

        status_tensor = torch.tensor(status_label, dtype=torch.long)

        return {
            "image": img_tensor,
            "mask": mask_tensor,       # mask_tensor is either a tensor or None
            "status": status_tensor,   # <--- tensor, safe to collate
            "status_str": s.extras.get("status_str"),
            "id": s.id,
            "paths": {
                "image": str(s.image_path),
                "mask": str(seg_mask_path) if seg_mask_path is not None else None,
            },
        }






if __name__ == "__main__":
    root = Path("../data_links/BreastCancer")
    samples = build_breast_cancer_samples(root)
    print("Number of samples:", len(samples))
    print("First sample:", samples[0])

    ds = BreastCancerDataset(samples)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    batch = next(iter(dl))

    print("image shape:", batch["image"].shape)    # -> [B, C, H, W]
    print("mask shape:", None if batch["mask"][0] is None else batch["mask"].shape)
    print("status:", batch["status"])              # tensor of 0/1 or list
    print("status_str:", batch["status_str"])
    print("ids:", batch["id"])
    print("paths[0]:", batch["paths"]["image"][0])


