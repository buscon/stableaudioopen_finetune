# from your finetune project root
mkdir -p overrides

# 1) Copy the repo config once
cp ../stable-audio-tools/configs/train/model_config.json overrides/model_config.demo.json

# 2) Add the missing demo keys (safe idempotent patch)
python - <<'PY'
import json, pathlib
p = pathlib.Path("overrides/model_config.demo.json")
data = json.loads(p.read_text())
data.setdefault("demo_every", 2000)
data.setdefault("demo_steps", 250)
data.setdefault("num_demos", 0)          # 0 => disables demo renders
data.setdefault("demo_cond", [])
data.setdefault("demo_cfg_scales", [5,7])
p.write_text(json.dumps(data, indent=2))
print("Patched overrides/model_config.demo.json")
PY

