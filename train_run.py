#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List


def find_checkpoint(model_dir: Path) -> Path:
    """
    Return an existing checkpoint path under --hf-model-dir.
    """
    exts = (".ckpt", ".safetensors", ".pt")
    for ext in exts:
        for p in sorted(model_dir.rglob(f"*{ext}")):
            return p
    sys.exit(f"[train] FATAL: no checkpoint found under: {model_dir}")


def abs_from_cwd(p: Path | str, provided_by_user: bool) -> Path:
    """
    Make a path absolute. If the user passed a relative path, resolve against CWD.
    If the path is a default inside the repo, resolve normally (later we pass cwd=repo_dir).
    """
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve() if provided_by_user else p.resolve()


def patch_dataset_config_adapter(dataset_cfg_path: Path, adapter_py_path: Path) -> None:
    """
    Ensure dataset_config.json points at the absolute metadata_adapter.py so the
    import works no matter where train.py runs from.
    """
    adapter_abs = str(adapter_py_path.resolve())
    try:
        data = json.loads(dataset_cfg_path.read_text())
    except FileNotFoundError:
        sys.exit(f"[train] FATAL: missing required file: {dataset_cfg_path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[train] FATAL: invalid JSON in {dataset_cfg_path}: {e}")

    changed = False

    # common schema:
    if isinstance(data, dict):
        # top-level (rare)
        if data.get("custom_metadata_module") != adapter_abs:
            if "custom_metadata_module" in data:
                data["custom_metadata_module"] = adapter_abs
                changed = True

        # per-dataset entries
        ds = data.get("datasets")
        if isinstance(ds, list):
            for entry in ds:
                if isinstance(entry, dict) and entry.get("custom_metadata_module") != adapter_abs:
                    entry["custom_metadata_module"] = adapter_abs
                    changed = True

    if changed:
        dataset_cfg_path.write_text(json.dumps(data, indent=2))
        print(f"[train] Patched {dataset_cfg_path.name} custom_metadata_module -> {adapter_abs}")
    else:
        print(f"[train] {dataset_cfg_path.name} already points to {adapter_abs}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stable Audio Open fine-tune launcher (no edits to stable-audio-tools)."
    )
    ap.add_argument("--repo-dir", type=Path, required=True,
                    help="Path to stable-audio-tools repository.")
    ap.add_argument("--hf-model-dir", type=Path, required=True,
                    help="Directory containing model checkpoint(s).")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Project data root (informational).")
    ap.add_argument("--save-dir", type=Path, required=True,
                    help="Where to store checkpoints/logs.")
    ap.add_argument("--run-name", required=True,
                    help="Run name for logging/checkpoints.")

    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum-batches", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--precision", choices=["16", "bf16", "32"], default="16")
    ap.add_argument("--max-epochs", type=int, default=None)

    # NEW: local override paths (minimal changes you asked for)
    ap.add_argument("--dataset-config", type=Path, default=None,
                    help="Path to a custom dataset_config.json (recommended).")
    ap.add_argument("--model-config", type=Path, default=None,
                    help="Path to a custom model_config.json (recommended).")

    # passthrough
    ap.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Anything extra to pass to train.py after '--'")

    args = ap.parse_args()

    repo_dir: Path = args.repo_dir.resolve()
    hf_model_dir: Path = args.hf_model_dir.resolve()
    save_dir: Path = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        sys.exit(f"[train] FATAL: --repo-dir not found: {repo_dir}")

    ckpt_path = find_checkpoint(hf_model_dir)

    # Defaults if user did NOT pass overrides (some repos have no configs/ folder).
    default_dataset_cfg = repo_dir / "configs" / "train" / "dataset_config.json"
    default_model_cfg = repo_dir / "configs" / "train" / "model_config.json"

    # Prefer user-provided override files
    dataset_cfg = args.dataset_config or default_dataset_cfg
    model_cfg = args.model_config or default_model_cfg

    # Make both absolute — if provided by user and relative, resolve against CWD,
    # because we'll run train.py with cwd=repo_dir.
    dataset_cfg = abs_from_cwd(dataset_cfg, provided_by_user=args.dataset_config is not None)
    model_cfg = abs_from_cwd(model_cfg, provided_by_user=args.model_config is not None)

    # Fail fast if the files don't exist
    if not dataset_cfg.exists():
        sys.exit(f"[train] FATAL: missing required file: {dataset_cfg}")
    if not model_cfg.exists():
        sys.exit(f"[train] FATAL: missing required file: {model_cfg}")

    # Ensure adapter path inside dataset_config is absolute
    adapter_py = repo_dir / "configs" / "train" / "metadata_adapter.py"
    if adapter_py.exists():
        patch_dataset_config_adapter(dataset_cfg, adapter_py)
    else:
        # If your adapter lives elsewhere (and printed 'Loaded 1200 rows' before),
        # just ensure your dataset_config already points to its absolute path.
        print(f"[train] Note: adapter not found at {adapter_py}. Skipping auto-patch.")

    # Derive some convenience paths for banner (optional)
    ds_root = args.data_root.resolve()
    aug_dir = ds_root / "dataset" / "output_dataset"
    aug_audio_dir = aug_dir / "audio"
    aug_csv = aug_dir / "metadata.csv"

    print("=== Stable Audio Fine-tune ===")
    print(f"repo_dir:   {repo_dir}")
    print(f"ckpt:       {ckpt_path}")
    print(f"dataset:    {aug_audio_dir}")
    print(f"csv:        {aug_csv}")
    print(f"save_dir:   {save_dir}")
    print(f"run_name:   {args.run_name}")
    print(f"precision:  {args.precision}")
    print(f"workers:    {args.num_workers}  batch: {args.batch_size}  accum: {args.accum_batches}")
    print(f"extra:      {(args.extra or '(none)')}")
    print("==============================")

    # Environment: let the user control WANDB via env; default to offline if unset
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    # Make the repo importable for `-m train`
    env["PYTHONPATH"] = str(repo_dir)

    # Build train command
    cmd = [
        sys.executable, "-u", "-m", "train",
        "--dataset-config", str(dataset_cfg),
        "--model-config",   str(model_cfg),
        "--pretrained-ckpt-path", str(ckpt_path),
        "--save-dir",       str(save_dir),
        "--name",           str(args.run_name),
        "--precision",      str(args.precision),
        "--num-workers",    str(args.num_workers),
        "--batch-size",     str(args.batch_size),
        "--accum-batches",  str(args.accum_batches),
    ]
    if args.max_epochs is not None:
        cmd += ["--max-epochs", str(args.max_epochs)]
    if args.extra:
        # allow passing raw train.py flags after a lone "--"
        if args.extra and args.extra[0] == "--":
            cmd += args.extra[1:]
        else:
            cmd += args.extra

    try:
        subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[train] Training failed with exit code {e.returncode}")

    print("\n✅ Training finished.")
    print("Outputs:")
    print(f"  Checkpoints & logs: {save_dir}")
    print(f"  Augmented dataset:  {aug_dir}")
    print(f"    Audio:            {aug_audio_dir}")
    print(f"    CSV:              {aug_csv}")


if __name__ == "__main__":
    main()

