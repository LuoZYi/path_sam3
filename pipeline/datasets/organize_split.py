#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

# --------- EDIT THESE TWO ---------
SRC = Path("../data_links/GlaS/GlaS")   # <- datalink root
OUT = Path("./Glas_new")      # <- your own folder
# ----------------------------------

# Split rule: (train + testA) => train, testB => val
TRAIN_PREFIXES = {"train"}
VAL_PREFIXES = {"testA"}
TEST_PREFIXES = {"testB"}

# Matches: <prefix>_<id>.bmp  OR  <prefix>_<id>_anno.bmp
pat = re.compile(r"^(?P<prefix>train|testA|testB)_(?P<id>\d+)(?P<anno>_anno)?\.bmp$")

def ensure_dirs(root: Path):
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "annos").mkdir(parents=True, exist_ok=True)

def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()  # overwrite if rerun
    dst.symlink_to(src.resolve())


def main():
    ensure_dirs(OUT)

    # Track which images/annos exist per (prefix,id)
    seen = {}  # (prefix, id) -> {"img": Path|None, "anno": Path|None}

    for p in SRC.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue

        prefix = m.group("prefix")
        idx = m.group("id")
        is_anno = m.group("anno") is not None

        key = (prefix, idx)
        if key not in seen:
            seen[key] = {"img": None, "anno": None}

        if is_anno:
            seen[key]["anno"] = p
        else:
            seen[key]["img"] = p

    # Copy into split folders
    for (prefix, idx), d in sorted(seen.items()):
        if prefix in TRAIN_PREFIXES:
            split = "train"
        elif prefix in VAL_PREFIXES:
            split = "val"
        elif prefix in TEST_PREFIXES:
            split = "test"
        else:
            continue

        img = d["img"]
        anno = d["anno"]

        if img is None or anno is None:
            print(f"[WARN] missing pair for {prefix}_{idx}: img={img is not None}, anno={anno is not None}")
            continue

        copy(img, OUT / split / "images" / img.name)
        copy(anno, OUT / split / "annos" / anno.name)

    print(f"Done. Output at: {OUT}")

if __name__ == "__main__":
    main()
