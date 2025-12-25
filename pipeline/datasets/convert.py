import argparse
import os
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)
    args = ap.parse_args()

    in_ckpt = os.path.abspath(args.in_ckpt)
    out_ckpt = os.path.abspath(args.out_ckpt)

    ckpt = torch.load(in_ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # 去掉常见前缀（以防万一）
    cleaned = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        cleaned[k2] = v

    # 加 detector. 前缀，让 builder 能吃进去
    det_state = {f"detector.{k}": v for k, v in cleaned.items()}

    # 存成一个最小 ckpt：只放 model 就够了
    torch.save({"model": det_state}, out_ckpt)

    print(f"[OK] wrote converted ckpt to: {out_ckpt}")
    print(f"  original keys: {len(cleaned)}")
    print(f"  detector keys: {len(det_state)}")
    print("  sample:", list(det_state.keys())[:5])

if __name__ == "__main__":
    main()
