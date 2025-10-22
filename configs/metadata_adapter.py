# ./configs/metadata_adapter.py
import os, csv
from pathlib import Path
from typing import Dict

# Cache: {stem -> {"prompt": "...", "negative_prompt": "..."}}
_META: Dict[str, Dict[str, str]] = {}
_LOADED = False

def _load():
    global _LOADED
    if _LOADED:
        return
    csv_path = Path(os.environ.get(
        "SA_METADATA_CSV",
        "./data/dataset/output_dataset/metadata.csv"
    )).expanduser()
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in reader.fieldnames or []}
            for r in reader:
                stem = Path(r.get("filepath","")).stem
                if not stem:
                    continue
                # Accept either 'prompt' or 'text' (fall back to filename)
                prompt = r.get(cols.get("prompt") or "prompt") \
                      or r.get(cols.get("text") or "text") \
                      or stem.replace("_"," ").replace("-"," ")
                neg = r.get(cols.get("negative_prompt") or "negative_prompt") or ""
                _META[stem] = {"prompt": prompt, "negative_prompt": neg}
    _LOADED = True

def _meta_for(sample_path) -> Dict[str, str]:
    _load()
    stem = Path(str(sample_path)).stem
    return _META.get(stem, {"prompt": "", "negative_prompt": ""})

# -------- PATH-style entry points (must return a string) --------
def get_audio_path(sample_path, *_, **__) -> str: return str(sample_path)
def get_path(sample_path, *_, **__) -> str:       return str(sample_path)
def resolve_path(sample_path, *_, **__) -> str:    return str(sample_path)

# -------- DOCS-style / METADATA-style (must return a dict) --------
# Official docs: get_custom_metadata(info, audio) -> dict
def get_custom_metadata(info, audio=None):
    # prefer path fields if present; fall back to audio or info itself
    path = (
        (isinstance(info, dict) and (info.get("path") or info.get("relpath") or info.get("filepath")))
        or audio
        or info
    )
    return _meta_for(path)

# Some branches call this name instead
def get_metadata(*args, **kwargs):
    try:
        return get_custom_metadata(*args, **kwargs)
    except TypeError:
        # If only a bare path is passed
        return _meta_for(args[0] if args else "")

