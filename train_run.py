#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, subprocess
from pathlib import Path

def safe_find(p: Path, candidates):
    for name in candidates:
        f = p / name
        if f.exists():
            return f
    return None

def main():
    ap = argparse.ArgumentParser("Stable Audio Open fine-tune launcher (doc-aligned)")
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--hf-model-dir", type=Path, required=True,
                    help="Local folder that contains HF files (model.ckpt or model.safetensors, model_config.json)")
    ap.add_argument("--dataset-config", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum-batches", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--precision", choices=["16","bf16","32"], default="16")
    ap.add_argument("--extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_dir   = args.repo_dir.resolve()
    defaults   = repo_dir / "defaults.ini"
    train_py   = repo_dir / "train.py"
    if not train_py.exists():
        sys.exit(f"[train] FATAL: {train_py} not found")

    # HF model bits
    hf_dir     = args.hf_model_dir.resolve()
    model_cfg  = safe_find(hf_dir, ["model_config.json"])
    ckpt       = safe_find(hf_dir, ["model.safetensors", "model.ckpt"])
    if not model_cfg:
        sys.exit(f"[train] FATAL: model_config.json not found in {hf_dir}")
    if not ckpt:
        sys.exit(f"[train] FATAL: model.safetensors/.ckpt not found in {hf_dir}")

    # dataset config
    dataset_cfg = args.dataset_config.resolve()
    if not dataset_cfg.exists():
        sys.exit(f"[train] FATAL: dataset config not found: {dataset_cfg}")

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stable Audio Fine-tune ===")
    print(f"repo_dir:   {repo_dir}")
    print(f"defaults:   {defaults}")
    print(f"model_cfg:  {model_cfg}")
    print(f"ckpt:       {ckpt}")
    print(f"dataset:    {dataset_cfg}")
    print(f"save_dir:   {save_dir}")
    print(f"run_name:   {args.run_name}")
    print(f"precision:  {args.precision}")
    print(f"workers:    {args.num_workers}  batch: {args.batch_size}  accum: {args.accum_batches}")
    print(f"extra:      {(args.extra or '(none)')}")
    print("==============================")

    env = os.environ.copy()
    # You can keep WANDB offline if you prefer; the README assumes wandb login.
    env.setdefault("WANDB_MODE", "offline")

    cmd = [
        sys.executable, "-u", "-m", "train",
        "--config-file", str(defaults),
        "--dataset-config", str(dataset_cfg),
        "--model-config", str(model_cfg),
        "--pretrained-ckpt-path", str(ckpt),   # doc: fine-tune from unwrapped HF checkpoint
        "--save-dir", str(save_dir),
        "--name", str(args.run_name),
        "--precision", str(args.precision),
        "--num-workers", str(args.num_workers),
        "--batch-size", str(args.batch_size),
        "--accum-batches", str(args.accum_batches),
    ]
    if args.extra:
        # allow passing e.g. --checkpoint-every 2000 --seed 42
        cmd += (args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra)

    subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    print("\n✅ Training launched via official flags.")

if __name__ == "__main__":
    main()

