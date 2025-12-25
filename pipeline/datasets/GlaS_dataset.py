from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


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



def build_glas_samples(root: Path) -> List[SampleEntry]:
    """
    root: ../data_links/GlaS

    需要有一个 grades.csv，格式类似：
    name,patient ID, grade (GlaS), grade (Sirinukunwattana et al. 2015)
    testA_1,4, benign, adenomatous
    ...
    """
    data_dir = root  # 或者 root / "images_and_masks"，按你真实结构改
    grades_csv = root / "Grade.csv"

    df = pd.read_csv(grades_csv)

    # 建一个 name -> row 的索引
    rows_by_name = {row["name"]: row for _, row in df.iterrows()}

    # 定义 grade 映射
    glas_map = {
        "benign": 0,
        "malignant": 1,
    }

    sirin_map = {
        "healthy": 0,
        "adenomatous": 1,
        "moderately differentiated": 2,
        "poorly differentiated": 3,
        "moderately-to-poorly differentated": 4,
    }

    samples: List[SampleEntry] = []

    # 遍历 image 文件
    for img_path in sorted(data_dir.glob("*.bmp")):
        name = img_path.stem          # e.g. "testA_1"
        if name.endswith("_anno"):
            # 跳过mask文件本体，只用 image 来驱动 sample
            continue

        # 根据文件名决定 split
        if name.startswith("train_"):
            split = "train"
        elif name.startswith("testA_"):
            split = "testA"
        elif name.startswith("testB_"):
            split = "testB"
        else:
            split = "unknown"

        # 读取对应的 grade 行（有可能有些名字不在 csv 里）
        row = rows_by_name.get(name)
        grade_glas_label = None
        grade_sirin_label = None
        patient_id = None

        if row is not None:
            # 注意列名要和你真正 csv 的 header 对上
            grade_glas_str = str(row[" grade (GlaS)"]).strip().lower()
            grade_sirin_str = str(row[" grade (Sirinukunwattana et al. 2015)"]).strip().lower()

            grade_glas_label = glas_map.get(grade_glas_str)
            grade_sirin_label = sirin_map.get(grade_sirin_str)
            patient_id = row["patient ID"]

        # mask 命名规则：这里假设是 name + "_mask.bmp"
        mask_path = data_dir / f"{name}_anno.bmp"
        if not mask_path.exists():
            # 如果你真实数据 mask 命名是别的规则（例如 name + "_anno.bmp"）
            # 在这里改一下
            mask_path = None

        target_paths = {}
        if mask_path is not None:
            target_paths["seg_mask"] = mask_path

        sample = SampleEntry(
            id=name,
            dataset="GlaS",
            split=split,
            image_path=img_path,
            target_paths=target_paths,
            meta_paths={},    # 没有 xml 就先空着
            labels={
                "grade_glas": grade_glas_label,
                "grade_sirin": grade_sirin_label,
            },
            extras={
                "patient_id": patient_id,
            },
        )
        samples.append(sample)

    return samples





from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader



class GlasDataset(Dataset):
    def __init__(self, samples: List[SampleEntry], transform=None, target_transform=None):
        """
        samples: build_glas_samples(root) 返回的 SampleEntry 列表
        transform: 对 image 的变换（如 torchvision.transforms）
        target_transform: 对 mask 的变换
        """
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # --- load image ---
        img = Image.open(s.image_path).convert("RGB")

        # --- load mask (segmentation) ---
        mask = None
        seg_mask_path = s.target_paths.get("seg_mask")
        if seg_mask_path is not None and seg_mask_path.exists():
            mask = Image.open(seg_mask_path)
        else:
            # GlaS 一般每张都有 mask，如果不存在我们直接报错帮助排查命名问题
            raise FileNotFoundError(f"Segmentation mask not found for sample {s.id}: {seg_mask_path}")

        # --- apply transforms / convert to tensors ---
        # image
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            arr = np.array(img).astype("float32")
            # (H, W, C) -> (C, H, W)
            arr = arr.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(arr) / 255.0

        # mask
        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask)
        else:
            # GlaS 的 mask 通常是单通道整数标签
            mask_tensor = torch.from_numpy(np.array(mask).astype("int64"))

        # --- grades: 映射成 tensor，若缺失则用 -1 占位 ---
        grade_glas = s.labels.get("grade_glas")
        grade_sirin = s.labels.get("grade_sirin")

        if grade_glas is None:
            grade_glas = -1
        if grade_sirin is None:
            grade_sirin = -1

        grade_glas = torch.tensor(grade_glas, dtype=torch.long)
        grade_sirin = torch.tensor(grade_sirin, dtype=torch.long)

        patient_id = s.extras.get("patient_id", None)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "grade_glas": grade_glas,       # benign/malignant -> 0/1/-1
            "grade_sirin": grade_sirin,     # 多级别 -> 0..N / -1
            "patient_id": patient_id,
            "id": s.id,
            "split": s.split,
            "paths": {
                "image": str(s.image_path),
                "mask": str(seg_mask_path),
            },
        }


if __name__ == "__main__":
    from GlaS_dataset import build_glas_samples  # 如果在同一文件里可以删掉这行

    root = Path("../data_links/GlaS/GlaS")
    samples = build_glas_samples(root)
    print("Number of GlaS samples:", len(samples))
    print("First SampleEntry:", samples[0])

    ds = GlasDataset(samples)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    batch = next(iter(dl))
    print("image shape:", batch["image"].shape)       # [B, C, H, W]
    print("mask shape:", batch["mask"].shape)         # [B, H, W]
    print("grade_glas:", batch["grade_glas"])         # tensor([0/1/...])
    print("grade_sirin:", batch["grade_sirin"])       # tensor([...])
    print("patient_id:", batch["patient_id"])
    print("ids:", batch["id"])
    print("paths[0]:", batch["paths"]["image"][0])