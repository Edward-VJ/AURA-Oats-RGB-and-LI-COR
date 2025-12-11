#!/usr/bin/env python
import os
import json
import pandas as pd
from pathlib import Path
import re

# Root directory where all the paper_2_experiment_* folders are
ROOT = Path(".")

# Expected splits, modes, and models
SPLITS = [
    "Split_A", "Split_B", "Split_C", "Split_D",
    "Split_A_Barley", "Split_B_Barley", "Split_C_Barley", "Split_D_Barley",
    "Split_A_Oat", "Split_B_Oat", "Split_C_Oat", "Split_D_Oat"
]
MODES  = ["RGB", "MS"]
MODELS = ["swin", "resnet"]
TARGETS = ["gsw", "VPleaf", "VPDleaf", "Fs", "Fm'", "yield", "biomass"]


rows = []

def safe_json_load(path):
    """Load JSON file even if it contains NaN or Infinity."""
    text = path.read_text()
    # Replace invalid JSON constants (NaN, Infinity, -Infinity) with null
    text = re.sub(r'\bNaN\b', 'null', text)
    text = re.sub(r'\bInfinity\b', 'null', text)
    text = re.sub(r'\b-Infinity\b', 'null', text)
    return json.loads(text)


for target in TARGETS:
    for split in SPLITS:
        for mode in MODES:
            for model in MODELS:
                exp_dir = ROOT / f"paper_2_SINGLEVAR_384_GB_{split}_{mode}_{target}" / f"{split}__{mode}__{model}__{target}"
                metrics_path = exp_dir / "metrics.json"

                if metrics_path.exists():
                    try:
                        metrics = safe_json_load(metrics_path)
                        row = {"dataset": split, "mode": mode, "model": model}
                        row.update(metrics)
                        rows.append(row)
                    except Exception as e:
                        print(f"[WARN] Could not load {metrics_path}: {e}")
                else:
                    print(f"[MISS] No metrics.json for {metrics_path}")

if not rows:
    print("No valid metrics files found — check your paths or JSON contents.")
else:
    df = pd.DataFrame(rows)
    front_cols = [c for c in ["dataset", "mode", "model", "lr", "val_rmse", "avg_r2"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]
    out_path = "combined_metrics_singlevariable_384_A-D-species.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path} with shape {df.shape}")
