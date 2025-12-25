# pannuke_dataset.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Iterable

import numpy as np
import pandas as pd
import yaml

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ======================
# 通用样本结构 SampleEntry
# ======================

@dataclass
class SampleEntry:
    id: str
    dataset: str
    split: str  # 这里我们直接用 "fold1"/"fold2"/"fold3"

    image_path: Path
    target_paths: Dict[str, Path] = field(default_factory=dict)
    meta_paths: Dict[str, Path] = field(default_factory=dict)

    labels: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


# ==========================
# PanNuke 样本构建函数
# ==========================

def load_tissue_type_mapping(config_path: Path) -> Dict[str, int]:
    """
    从 dataset_config.yaml 读取 tissue_types 映射：
    tissue_types:
      "Adrenal_gland": 0
      "Bile-duct": 1
      ...
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tissue_map = cfg.get("tissue_types", {})
    # 确保 value 是 int
    return {k: int(v) for k, v in tissue_map.items()}


def build_pannuke_samples(
    root: Path,
    folds: Iterable[str] = ("fold0", "fold1", "fold2"),
) -> List[SampleEntry]:
    """
    root: ../data_links/PanNuke

    假设结构是：
      root / foldX / images / *.png
      root / foldX / masks  / *.npy
      root / foldX / cell_count.csv
      root / foldX / types.csv
      root / dataset_config.yaml
    """
    samples: List[SampleEntry] = []

    # tissue 类型从 dataset_config.yaml 里读
    dataset_cfg_path = root / "dataset_config.yaml"
    if dataset_cfg_path.exists():
        tissue_map = load_tissue_type_mapping(dataset_cfg_path)  # str -> int
    else:
        tissue_map = {}
        print(f"[Warning] dataset_config.yaml not found at {dataset_cfg_path}, tissue_type 将保持字符串。")

    for fold_name in folds:
        fold_dir = root / fold_name

        images_dir = fold_dir / "images"   # 如果不是这个目录名，在这里改
        masks_dir  = fold_dir / "labels"    # 如果是 "labels"，改成 fold_dir / "labels"

        cell_csv   = fold_dir / "cell_count.csv"
        types_csv  = fold_dir / "types.csv"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks dir not found: {masks_dir}")
        if not cell_csv.exists():
            raise FileNotFoundError(f"cell_count.csv not found: {cell_csv}")
        if not types_csv.exists():
            raise FileNotFoundError(f"types.csv not found: {types_csv}")

        # 读 CSV
        cell_df  = pd.read_csv(cell_csv)   # columns: Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
        types_df = pd.read_csv(types_csv)  # columns: img,type

        # 建 index，方便查找
        cell_by_name  = {row["Image"]: row for _, row in cell_df.iterrows()}
        types_by_name = {row["img"]: row for _, row in types_df.iterrows()}

        for img_path in sorted(images_dir.glob("*.png")):
            img_name = img_path.name    # e.g. "0_0.png"
            stem = img_path.stem        # e.g. "0_0"

            # mask: 同名 .npy
            mask_path = masks_dir / f"{stem}.npy"
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for {img_name}: {mask_path}")

            # tissue type：字符串
            t_row = types_by_name.get(img_name, None)
            tissue_str = t_row["type"] if t_row is not None else None

            # 映射成 int（如果 mapping 存在）
            tissue_id = None
            if tissue_str is not None and tissue_map:
                tissue_id = tissue_map.get(tissue_str, None)

            # cell counts：Neoplastic, Inflammatory, Connective, Dead, Epithelial
            c_row = cell_by_name.get(img_name, None)
            cell_counts = None
            neoplastic_count = 0
            if c_row is not None:
                neoplastic_count = int(c_row["Neoplastic"])
                cell_counts = np.array([
                    c_row["Neoplastic"],
                    c_row["Inflammatory"],
                    c_row["Connective"],
                    c_row["Dead"],
                    c_row["Epithelial"],
                ], dtype=int)

            # proxy_malignancy：如果 neoplastic 细胞 > 0，就记作 1，否则 0
            proxy_malignancy = int(neoplastic_count > 0)

            # 构造 SampleEntry
            sample = SampleEntry(
                id=f"{fold_name}_{stem}",   # 全局唯一 id
                dataset="PanNuke",
                split=fold_name,            # 直接用 fold 信息
                image_path=img_path,
                target_paths={
                    "instance_mask_npy": mask_path,  # nuclei instance mask
                },
                meta_paths={
                    "cell_count_csv": cell_csv,
                    "types_csv": types_csv,
                },
                labels={
                    "slide_status": None,           # PanNuke 没有真正的良恶性 label
                    "proxy_malignancy": proxy_malignancy,  # Neoplastic>0 的 proxy
                    "tissue_type": tissue_id,       # int 或 None
                },
                extras={
                    "fold": fold_name,
                    "tissue_type_str": tissue_str,
                    "cell_counts": cell_counts,     # np.ndarray 或 None
                },
            )

            samples.append(sample)

    return samples


# ==========================
# PanNuke Dataset 类
# ==========================

# class PannukeDataset(Dataset):
#     def __init__(self, samples: List[SampleEntry], transform=None, target_transform=None):
#         """
#         samples: build_pannuke_samples(root) 生成的 SampleEntry 列表
#         """
#         self.samples = samples
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx: int):
#         s = self.samples[idx]

#         # --- 图像 ---
#         img = Image.open(s.image_path).convert("RGB")

#         if self.transform is not None:
#             img_tensor = self.transform(img)
#         else:
#             arr = np.array(img).astype("float32")  # (H,W,C)
#             arr = arr.transpose(2, 0, 1)          # (C,H,W)
#             img_tensor = torch.from_numpy(arr) / 255.0

#         # --- mask: .npy (instance 或 semantic，看你数据) ---
#         mask_tensor = None
#         npy_path = s.target_paths.get("instance_mask_npy")
#         if npy_path is not None and npy_path.exists():
#             #mask = np.load(npy_path)  # 可能是 (H,W) 或 (H,W,K)
#             mask = np.load(npy_path, allow_pickle=True)

#             # 这里先假设是 (H,W)，如果是别的形状你可以改一改
#             mask_tensor = torch.from_numpy(mask.astype("int64"))
#         else:
#             raise FileNotFoundError(f"Mask npy not found for sample {s.id}: {npy_path}")

#         # --- label: proxy_malignancy & tissue_type ---
#         proxy_malignancy = s.labels.get("proxy_malignancy", None)
#         if proxy_malignancy is None:
#             proxy_malignancy = -1
#         proxy_malignancy = torch.tensor(proxy_malignancy, dtype=torch.long)

#         tissue_type = s.labels.get("tissue_type", None)
#         if tissue_type is None:
#             tissue_type = -1
#         tissue_type = torch.tensor(tissue_type, dtype=torch.long)

#         cell_counts = s.extras.get("cell_counts", None)
#         if cell_counts is not None:
#             cell_counts_tensor = torch.from_numpy(cell_counts.astype("int64"))
#         else:
#             cell_counts_tensor = None

#         return {
#             "image": img_tensor,
#             "mask": mask_tensor,
#             "proxy_malignancy": proxy_malignancy,   # 0/1/-1
#             "tissue_type": tissue_type,             # int id 或 -1
#             "cell_counts": cell_counts_tensor,      # [5] 或 None
#             "tissue_type_str": s.extras.get("tissue_type_str"),
#             "fold": s.extras.get("fold"),
#             "id": s.id,
#             "dataset": s.dataset,
#             "split": s.split,                       # "fold1"/"fold2"/"fold3"
#             "paths": {
#                 "image": str(s.image_path),
#                 "mask_npy": str(npy_path),
#             },
#         }

class PannukeDataset(Dataset):
    def __init__(self, samples: List[SampleEntry], transform=None, target_transform=None):
        """
        samples: build_pannuke_samples(root) 生成的 SampleEntry 列表
        transform: 对 image 的变换
        target_transform: 对 mask 的变换（如果你以后想加）
        """
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # -------- 图像 --------
        img = Image.open(s.image_path).convert("RGB")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            arr = np.array(img).astype("float32")  # (H, W, C)
            arr = arr.transpose(2, 0, 1)          # (C, H, W)
            img_tensor = torch.from_numpy(arr) / 255.0

        # -------- mask: .npy 里可能是 ndarray / ndarray(object) / dict --------
        mask_tensor = None
        npy_path = s.target_paths.get("instance_mask_npy")

        if npy_path is None or not npy_path.exists():
            raise FileNotFoundError(f"Mask npy not found for sample {s.id}: {npy_path}")

        raw = np.load(npy_path, allow_pickle=True)

        # 1) 如果是 ndarray(object) 且只有一个元素，通常是包了一层 dict
        obj = raw
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            if obj.size == 1:
                obj = obj.item()   # 取出里面那个 dict
            else:
                # 多元素 object array，很少见，先直接转 list 看看
                obj = list(obj)

        # 2) 如果是 dict，优先从里面拿 type_map / class_map / inst_map
        if isinstance(obj, dict):
            # 按优先级挑一个 key
            chosen_arr = None
            for key in ("type_map", "class_map", "types", "mask", "inst_map"):
                if key in obj:
                    chosen_arr = obj[key]
                    break
            if chosen_arr is None:
                raise ValueError(
                    f"Unknown mask dict structure for sample {s.id}: keys = {list(obj.keys())}"
                )
            mask_arr = np.asarray(chosen_arr)

        # 3) 如果还是 ndarray（数值型），就直接当成 mask 用
        elif isinstance(obj, np.ndarray):
            mask_arr = np.asarray(obj)

        else:
            raise TypeError(
                f"Unexpected mask object type for sample {s.id}: {type(obj)} "
                f"(raw type: {type(raw)}, raw dtype: {getattr(raw, 'dtype', None)})"
            )

        # 如果是 (H, W, 1) 或 (1, H, W) 这种，简单 squeeze 一下
        if mask_arr.ndim > 2:
            mask_arr = np.squeeze(mask_arr)

        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_arr)
        else:
            mask_tensor = torch.from_numpy(mask_arr.astype("int64"))

        # -------- labels: proxy_malignancy & tissue_type --------
        proxy_malignancy = s.labels.get("proxy_malignancy", None)
        if proxy_malignancy is None:
            proxy_malignancy = -1
        proxy_malignancy = torch.tensor(proxy_malignancy, dtype=torch.long)

        tissue_type = s.labels.get("tissue_type", None)
        if tissue_type is None:
            tissue_type = -1
        tissue_type = torch.tensor(tissue_type, dtype=torch.long)

        # -------- cell counts --------
        cell_counts = s.extras.get("cell_counts", None)
        if cell_counts is not None:
            cell_counts_tensor = torch.from_numpy(
                np.asarray(cell_counts).astype("int64")
            )
        else:
            # 理论上 PanNuke 每张都有 cell_counts，这里只是兜底
            cell_counts_tensor = torch.zeros(5, dtype=torch.long)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "proxy_malignancy": proxy_malignancy,   # 0/1/-1
            "tissue_type": tissue_type,             # int id 或 -1
            "cell_counts": cell_counts_tensor,      # [5]
            "tissue_type_str": s.extras.get("tissue_type_str"),
            "fold": s.extras.get("fold"),
            "id": s.id,
            "dataset": s.dataset,
            "split": s.split,                       # "fold0"/"fold1"/"fold2"
            "paths": {
                "image": str(s.image_path),
                "mask_npy": str(npy_path),
            },
        }



# ==========================
# 简单测试入口
# ==========================

if __name__ == "__main__":
    root = Path("../data_links/PanNuke")  # 按你的实际路径改

    samples = build_pannuke_samples(root, folds=("fold0", "fold1", "fold2"))
    print("Number of PanNuke samples:", len(samples))
    print("First SampleEntry:", samples[0])

    ds = PannukeDataset(samples)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    batch = next(iter(dl))
    print("image shape:", batch["image"].shape)        # [B, C, H, W]
    print("mask shape:", batch["mask"].shape)          # [B, H, W] 或 [B, H, W, ...] 视你的 npy 而定
    print("proxy_malignancy:", batch["proxy_malignancy"])
    print("tissue_type:", batch["tissue_type"])
    print("cell_counts shape:", None if batch["cell_counts"][0] is None else batch["cell_counts"].shape)
    print("folds:", batch["fold"])
    print("ids:", batch["id"])
    print("paths[0]:", batch["paths"]["image"][0])
