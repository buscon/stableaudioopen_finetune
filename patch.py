#!/usr/bin/env python3

import json, pathlib
p = pathlib.Path("overrides/model_config.demo.json")
data = json.loads(p.read_text())

# The values; num_demos=0 disables demo renders (safe default)
demo_defaults = {
    "num_demos": 0,
    "demo_every": 2000,
    "demo_steps": 250,
    "demo_cond": [],
    "demo_cfg_scales": [5, 7],
}

# 1) Ensure nested demo block exists and has the keys
demo = data.get("demo", {})
for k, v in demo_defaults.items():
    demo.setdefault(k, v)
data["demo"] = demo

# 2) Also add flat keys (for configs that read from top-level)
for k, v in demo_defaults.items():
    data.setdefault(k, v)

p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
print("Patched", p)
print("Result demo section:", json.dumps(data["demo"], indent=2))
