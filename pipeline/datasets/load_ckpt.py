# check_ckpt_load.py
import argparse
import os
from collections import Counter

import torch


def torch_load_compat(path: str):
    """Torch load that works across torch versions (weights_only is not always supported)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(ckpt_obj):
    """
    Try to extract a "state_dict-like" mapping from various checkpoint formats.
    Returns: (state_dict, where_from)
    """
    if isinstance(ckpt_obj, dict):
        # common patterns
        for k in ["model", "state_dict", "model_state", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k], f"ckpt['{k}']"
        # sometimes the whole dict IS the state dict (key->tensor)
        if all(isinstance(v, (torch.Tensor,)) for v in ckpt_obj.values()):
            return ckpt_obj, "ckpt (already state_dict)"
    return None, "unknown"


def summarize_keys(keys, topk=30):
    cnt = Counter()
    for k in keys:
        if k.startswith("module."):
            cnt["prefix:module."] += 1
        if k.startswith("model."):
            cnt["prefix:model."] += 1
        if "detector" in k:
            cnt["contains:detector"] += 1
        if "tracker" in k:
            cnt["contains:tracker"] += 1
        if "segmentation_head" in k:
            cnt["contains:segmentation_head"] += 1
        if "backbone" in k:
            cnt["contains:backbone"] += 1
        if "transformer" in k:
            cnt["contains:transformer"] += 1
    print("\n[Key pattern summary]")
    for kk, vv in cnt.most_common():
        print(f"  {kk:28s} = {vv}")

    print(f"\n[First {topk} keys]")
    for i, k in enumerate(list(keys)[:topk]):
        print(f"  {i:02d}: {k}")


def compare_models(base_model, loaded_model, max_print=20):
    """
    Compare parameters between two models built with same seed.
    If loading worked, many params should differ.
    """
    base_sd = base_model.state_dict()
    load_sd = loaded_model.state_dict()

    diffs = []
    same_shape = 0
    for k, v in load_sd.items():
        if k in base_sd and base_sd[k].shape == v.shape:
            same_shape += 1
            d = (v.float() - base_sd[k].float()).abs().mean().item()
            diffs.append((d, k))

    diffs.sort(reverse=True)  # biggest difference first
    changed = sum(1 for d, _ in diffs if d > 1e-7)
    print("\n[Model param diff check]")
    print(f"  comparable params (same key+shape): {same_shape}")
    print(f"  params with mean(|diff|) > 1e-7:   {changed}")
    if diffs:
        print(f"\n  Top {max_print} largest diffs:")
        for d, k in diffs[:max_print]:
            print(f"    {d:.3e}  {k}")

        print(f"\n  Top {max_print} smallest diffs:")
        for d, k in diffs[-max_print:]:
            print(f"    {d:.3e}  {k}")

    # quick heuristic
    if changed < max(10, int(0.01 * max(1, same_shape))):
        print("\n[Heuristic] ⚠️ 看起来“加载后”和“未加载”几乎一样：很可能没有真正加载到你的 finetune 权重。")
    else:
        print("\n[Heuristic] ✅ 加载后参数变化明显：大概率 checkpoint 真的被加载进模型了。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt/.ckpt checkpoint")
    parser.add_argument("--bpe", default=None, help="Path to bpe_simple_vocab_16e6.txt.gz (optional)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to build model")
    parser.add_argument("--enable_segmentation", action="store_true", help="Enable segmentation head")
    parser.add_argument("--no_build", action="store_true", help="Only inspect ckpt; do not build model")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    print(f"[1] Loading checkpoint: {ckpt_path}")
    ckpt_obj = torch_load_compat(ckpt_path)
    print(f"  ckpt type: {type(ckpt_obj)}")
    if isinstance(ckpt_obj, dict):
        print(f"  top-level keys: {list(ckpt_obj.keys())[:30]}{' ...' if len(ckpt_obj) > 30 else ''}")

    state, where = extract_state_dict(ckpt_obj)
    if state is None:
        print("\n[ERROR] Cannot find state_dict-like mapping in this checkpoint.")
        print("        你需要自己确认 ckpt 结构，或者告诉我 ckpt 的 top-level keys 我帮你适配。")
        return

    keys = list(state.keys())
    print(f"\n[2] Extracted state_dict from: {where}")
    print(f"  num params in extracted state_dict: {len(keys)}")
    summarize_keys(keys, topk=30)

    if args.no_build:
        return

    print("\n[3] Importing model builder and testing load...")
    # IMPORTANT: adjust import path if your repo structure differs
    # If your builder is at sam3/model_builder.py, this should work:
    from sam3.model_builder import build_sam3_image_model

    # Build a baseline model (same seed), without loading ckpt
    torch.manual_seed(0)
    base_model = build_sam3_image_model(
        bpe_path=args.bpe,
        device=args.device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=args.enable_segmentation,
        enable_inst_interactivity=False,
        compile=False,
    )

    # Build a loaded model (same seed), with your ckpt
    torch.manual_seed(0)
    loaded_model = build_sam3_image_model(
        bpe_path=args.bpe,
        device=args.device,
        eval_mode=True,
        checkpoint_path=ckpt_path,
        load_from_HF=False,
        enable_segmentation=args.enable_segmentation,
        enable_inst_interactivity=False,
        compile=False,
    )

    # Compare weights to verify if loading actually changed parameters
    compare_models(base_model, loaded_model, max_print=20)


if __name__ == "__main__":
    main()
