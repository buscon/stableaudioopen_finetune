#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List

def find_checkpoint(model_dir: Path) -> Path:
    exts = (".ckpt", ".safetensors", ".pt")
    candidates: List[Path] = []
    for ext in exts:
        candidates += sorted(model_dir.rglob(f"*{ext}"))
    if not candidates:
        sys.exit(f"[train] FATAL: no checkpoint found under: {model_dir}")
    preferred = [p for p in candidates if p.name == "model.ckpt"]
    return preferred[0] if preferred else candidates[0]

def rewrite_dataset_config_to_file_path(dataset_cfg_path: Path, adapter_py: Path) -> None:
    """Force custom_metadata_module to be the ABSOLUTE FILE PATH to metadata_adapter.py,
    both at the top-level and inside each dataset entry if present."""
    if not dataset_cfg_path.exists():
        sys.exit(f"[train] FATAL: dataset config not found: {dataset_cfg_path}")
    try:
        data = json.loads(dataset_cfg_path.read_text())
    except Exception as e:
        sys.exit(f"[train] FATAL: cannot read JSON {dataset_cfg_path}: {e}")

    adapter_abs = str(adapter_py.resolve())
    changed = False

    # top-level key
    if isinstance(data, dict) and data.get("custom_metadata_module") != adapter_abs:
        data["custom_metadata_module"] = adapter_abs
        changed = True

    # per-dataset key(s)
    if isinstance(data, dict) and isinstance(data.get("datasets"), list):
        for ds in data["datasets"]:
            if isinstance(ds, dict) and ds.get("custom_metadata_module") != adapter_abs:
                ds["custom_metadata_module"] = adapter_abs
                changed = True

    if changed:
        dataset_cfg_path.write_text(json.dumps(data, indent=2))
        print(f"[train] Patched {dataset_cfg_path.name} custom_metadata_module -> {adapter_abs}")
    else:
        print(f"[train] {dataset_cfg_path.name} already points to {adapter_abs}")

def main():
    ap = argparse.ArgumentParser(description="Stable Audio Open – Fine-tune launcher")
    ap.add_argument("--repo-dir", required=True, type=Path, help="Path to stable-audio-tools repo")
    ap.add_argument("--hf-model-dir", required=True, type=Path, help="Directory with the downloaded checkpoint(s)")
    ap.add_argument("--data-root", required=True, type=Path, help="Root where dataset builder put output_dataset/")
    ap.add_argument("--save-dir", required=True, type=Path, help="Directory to store checkpoints/logs")
    ap.add_argument("--run-name", required=True, help="Run/experiment name")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum-batches", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--precision", default="16", choices=["16", "bf16", "32"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Anything extra to pass to train.py after '--'")
    args = ap.parse_args()

    repo_dir: Path = args.repo_dir.resolve()
    if not (repo_dir.exists() and (repo_dir / "train.py").exists()):
        sys.exit(f"[train] FATAL: repo_dir not valid or train.py missing: {repo_dir}")

    dataset_cfg = repo_dir / "configs" / "train" / "dataset_config.json"
    model_cfg   = repo_dir / "configs" / "train" / "model_config.json"
    adapter_py  = repo_dir / "configs" / "train" / "metadata_adapter.py"
    for pth in (dataset_cfg, model_cfg, adapter_py):
        if not pth.exists():
            sys.exit(f"[train] FATAL: missing required file: {pth}")

    ds_root       = args.data_root.resolve()
    aug_audio_dir = ds_root / "dataset" / "output_dataset" / "audio"
    aug_csv       = ds_root / "dataset" / "output_dataset" / "metadata.csv"

    if not aug_audio_dir.exists():
        sys.exit(f"[train] FATAL: dataset audio not found: {aug_audio_dir}\n"
                 f"  (Run your setup script to build the augmented dataset.)")
    if not aug_csv.exists():
        sys.exit(f"[train] FATAL: dataset metadata.csv not found: {aug_csv}\n"
                 f"  (Run your setup script to build the augmented dataset.)")

    ckpt = find_checkpoint(args.hf_model_dir.resolve())

    # Crucial fix: train.py loads the module *by file path* → rewrite to absolute path
    rewrite_dataset_config_to_file_path(dataset_cfg, adapter_py)

    # Environment for the subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir)
    env["MPLBACKEND"] = "Agg"
    env.setdefault("WANDB_MODE", "offline")
    env["SA_METADATA_CSV"] = str(aug_csv)

    # Optional quick probe (kept for sanity, not required for train.py)
    try:
        import importlib
        sys.path.insert(0, str(adapter_py.parent))
        os.environ["SA_METADATA_CSV"] = str(aug_csv)
        importlib.invalidate_caches()
        importlib.import_module("metadata_adapter")
        print("[train] metadata_adapter import OK")
    except Exception as e:
        print(f"[train][WARN] metadata_adapter probe import failed (train.py will still use file path): {e}")
    finally:
        if str(adapter_py.parent) in sys.path:
            sys.path.remove(str(adapter_py.parent))

    print("=== Stable Audio Fine-tune ===")
    print(f"repo_dir:   {repo_dir}")
    print(f"ckpt:       {ckpt}")
    print(f"dataset:    {aug_audio_dir}")
    print(f"csv:        {aug_csv}")
    print(f"save_dir:   {args.save_dir.resolve()}")
    print(f"run_name:   {args.run_name}")
    print(f"precision:  {args.precision}")
    print(f"workers:    {args.num_workers}  batch: {args.batch_size}  accum: {args.accum_batches}")
    extra_display = ' '.join(args.extra) if args.extra else "(none)"
    print(f"extra:      {extra_display}")
    print("==============================", flush=True)

    cmd = [
        sys.executable, "-u", "-m", "train",
        "--dataset-config", str(dataset_cfg),
        "--model-config",   str(model_cfg),
        "--pretrained-ckpt-path", str(ckpt),
        "--save-dir", str(args.save_dir),
        "--name", args.run_name,
        "--precision", args.precision,
        "--num-workers", str(args.num_workers),
        "--batch-size", str(args.batch_size),
        "--accum-batches", str(args.accum_batches),
    ]
    if args.max_epochs is not None:
        cmd += ["--max-epochs", str(args.max_epochs)]
    if args.extra:
        cmd += args.extra

    try:
        subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[train] Training failed with exit code {e.returncode}")

    print("\n✅ Training finished.")
    print("Outputs:")
    print(f"  Checkpoints & logs: {args.save_dir.resolve()}")
    print(f"  Augmented dataset:  {ds_root / 'dataset' / 'output_dataset'}")
    print(f"    Audio:            {aug_audio_dir}")
    print(f"    CSV:              {aug_csv}")

if __name__ == "__main__":
    main()

