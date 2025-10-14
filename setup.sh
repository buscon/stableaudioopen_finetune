#!/usr/bin/env bash
set -euo pipefail

# ---------- CONFIG (override via env if needed) ----------
: "${PRJ_DIR:=$(cd "$(dirname "$0")" && pwd)}"
: "${REPO_DIR:=${PRJ_DIR}/../stable-audio-tools}"
: "${HF_MODEL_DIR:=${PRJ_DIR}/../stableaudio/models/stabilityai__stable-audio-open-1.0}"
: "${DATA_ROOT:=${PRJ_DIR}/data}"
: "${VENV_DIR:=${PRJ_DIR}/.venv}"

: "${RAW_ZIP_URL:=https://zenodo.org/records/15630417/files/DCASE-TASK7-2024-Open-Source.zip}"
RAW_DIR="${DATA_ROOT}/raw"
RAW_ZIP="${RAW_DIR}/DCASE-TASK7-2024-Open-Source.zip"
UNZIP_DIR="${RAW_DIR}/DCASE"

AUG_ROOT="${DATA_ROOT}/dataset/output_dataset"
AUG_AUDIO="${AUG_ROOT}/audio"
AUG_CSV="${AUG_ROOT}/metadata.csv"

BUILDER_LOCAL="${PRJ_DIR}/build_dataset_auto.py"
: "${BUILDER_URL:=https://raw.githubusercontent.com/inkuele/stableaudio/main/training_scripts/build_dataset_auto.py}"

: "${TARGET_MINUTES:=120}"
: "${MAX_VARIANTS:=19}"

# ---------- OS packages ----------
if command -v apt-get >/dev/null 2>&1; then
  echo "[setup] apt-get install python3.10, sox, unzip, wget, git"
  sudo apt-get update -y
  sudo apt-get install -y python3.10 python3.10-venv sox libsox-fmt-all unzip wget git
else
  echo "[setup][warn] Not Debian/Ubuntu; ensure python3.10, sox, unzip, wget, git are installed."
fi

# ---------- venv ----------
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating venv at $VENV_DIR"
  python3.10 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -V

# ---------- stable-audio-tools ----------
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[setup] Cloning stable-audio-tools -> $REPO_DIR"
  git clone https://github.com/Stability-AI/stable-audio-tools.git "$REPO_DIR"
else
  echo "[setup] Found repo at $REPO_DIR"
fi

python -m pip install -U pip setuptools wheel
python -m pip install -e "$REPO_DIR"

# Runtime deps sometimes missing in envs — ensure them:
python - <<'PY'
import sys, subprocess
pkgs = [
  "webdataset", "k-diffusion", "alias-free-torch", "auraloss",
  "transformers==4.43.4", "sentencepiece", "pytorch-lightning==2.2.5",
  "accelerate", "safetensors", "einops", "omegaconf", "torchmetrics",
  "pandas", "tqdm", "matplotlib", "wandb", "prefigure"
]
def ensure(p):
    try:
        __import__(p.split("==")[0].replace("-", "_"))
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p])
for p in pkgs: ensure(p)
print("[setup] Python deps OK.")
PY

# ---------- make configs.train a package + adapter ----------
mkdir -p "${REPO_DIR}/configs/train"
[ -f "${REPO_DIR}/configs/train/__init__.py" ] || echo "# pkg" > "${REPO_DIR}/configs/train/__init__.py"

cat > "${REPO_DIR}/configs/train/metadata_adapter.py" <<'PY'
import csv, os
print("[metadata_adapter] >>> MODULE IMPORTED <<<")
CSV_PATH = os.environ.get("SA_METADATA_CSV") or "metadata.csv"
_rows_by_base, _printed = {}, 0
def _load_csv():
    if not os.path.exists(CSV_PATH):
        print(f"[metadata_adapter] CSV NOT FOUND: {CSV_PATH}"); return 0
    n=0
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            import os as _os
            base = _os.path.basename((row.get("file") or "").strip())
            if base: _rows_by_base[base] = row; n += 1
    print(f"[metadata_adapter] Loaded {n} rows."); return n
_loaded = _load_csv()
def get_custom_metadata(info, audio):
    global _printed
    import os as _os
    base = _os.path.basename(info.get("relpath",""))
    row = _rows_by_base.get(base)
    prompt = (row.get("prompt").strip() if row and row.get("prompt") else "")
    if not prompt:
        stem, _ = _os.path.splitext(base); prompt = stem.replace("_"," ").replace("-"," ")
    if _printed < 6:
        print(f"[metadata_adapter] base={base!r} matched={'yes' if row else 'no'} prompt='{prompt[:60]}'"); _printed += 1
    return {"prompt": prompt}
PY

# ---------- dataset: download, unzip, augment ----------
mkdir -p "$RAW_DIR"
if [ -d "${UNZIP_DIR}/dev/audio" ] && [ -f "${UNZIP_DIR}/dev/caption.csv" ]; then
  echo "[setup] DCASE already unzipped at ${UNZIP_DIR}"
else
  [ -f "$RAW_ZIP" ] || wget -O "$RAW_ZIP" "$RAW_ZIP_URL"
  mkdir -p "$UNZIP_DIR"
  unzip -o -q "$RAW_ZIP" -d "$UNZIP_DIR"
fi

if [ -f "$BUILDER_LOCAL" ]; then
  echo "[setup] Using local builder $BUILDER_LOCAL"
else
  wget -O "$BUILDER_LOCAL" "$BUILDER_URL"
fi

if [ -f "$AUG_CSV" ] && [ -d "$AUG_AUDIO" ]; then
  echo "[setup] Augmented dataset exists at $AUG_ROOT"
else
  echo "[setup] Building augmented dataset..."
  mkdir -p "$AUG_ROOT"
  python "$BUILDER_LOCAL" \
    --raw_dir       "${UNZIP_DIR}/dev/audio" \
    --caption_csv   "${UNZIP_DIR}/dev/caption.csv" \
    --out_root      "${DATA_ROOT}/dataset/output_dataset" \
    --target_minutes "${TARGET_MINUTES}" \
    --max_variants_per_file "${MAX_VARIANTS}"
fi

# ---------- dataset & model configs (abs paths) ----------
cat > "${REPO_DIR}/configs/train/dataset_config.json" <<JSON
{
  "dataset_type": "audio_dir",
  "datasets": [
    {
      "id": "ambient_dataset",
      "path": "${AUG_AUDIO}",
      "custom_metadata_module": "configs.train.metadata_adapter"
    }
  ],
  "random_crop": true
}
JSON

if [ ! -f "${REPO_DIR}/configs/train/model_config.json" ]; then
cat > "${REPO_DIR}/configs/train/model_config.json" <<'JSON'
{
  "model_type": "diffusion_cond",
  "sample_rate": 44100,
  "audio_channels": 2,
  "sample_size": 1323000,
  "model": {
    "pretransform": {
      "type": "autoencoder",
      "iterate_batch": true,
      "config": {
        "encoder": {"type":"oobleck","requires_grad":false,"config":{"in_channels":2,"channels":128,"c_mults":[1,2,4,8,16],"strides":[2,4,4,8,8],"latent_dim":128,"use_snake":true}},
        "decoder": {"type":"oobleck","config":{"out_channels":2,"channels":128,"c_mults":[1,2,4,8,16],"strides":[2,4,4,8,8],"latent_dim":64,"use_snake":true,"final_tanh":false}},
        "bottleneck": {"type":"vae"},
        "latent_dim": 64,
        "downsampling_ratio": 2048,
        "io_channels": 2
      }
    },
    "conditioning": {
      "configs": [
        { "id": "prompt", "type": "t5", "config": { "t5_model_name": "t5-base", "max_length": 128 } }
      ],
      "cond_dim": 768
    },
    "diffusion": {
      "cross_attention_cond_ids": ["prompt"],
      "global_cond_ids": [],
      "type": "dit",
      "config": {
        "io_channels": 64,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "cond_token_dim": 768,
        "global_cond_dim": 768,
        "project_cond_tokens": false,
        "transformer_type": "continuous_transformer"
      }
    },
    "io_channels": 64
  },
  "training": {
    "use_ema": true,
    "log_loss_info": false,
    "optimizer_configs": {
      "diffusion": {
        "optimizer": { "type": "AdamW", "config": { "lr": 5e-5, "betas": [0.9, 0.999], "weight_decay": 1e-3 } },
        "scheduler": { "type": "CosineAnnealingLR", "config": { "T_max": 100000, "eta_min": 1e-6 } }
      }
    }
  }
}
JSON
fi

echo
echo "✅ Setup complete."
echo "   Repo:              ${REPO_DIR}"
echo "   Venv:              ${VENV_DIR}"
echo "   Model dir:         ${HF_MODEL_DIR}"
echo "   Raw unzip:         ${UNZIP_DIR}"
echo "   Aug dataset:       ${AUG_ROOT}"
echo "     audio:           ${AUG_AUDIO}"
echo "     csv:             ${AUG_CSV}"
echo
echo "Next: run training with your Python script: .venv/bin/python train_local.py"

