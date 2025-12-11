#!/usr/bin/env python
import os
import json
import pandas as pd
from pathlib import Path
import re

ROOT = Path(".")

SPLITS = [
    "Split_A", "Split_B", "Split_C", "Split_D",
    "Split_A_Barley", "Split_B_Barley", "Split_C_Barley", "Split_D_Barley",
    "Split_A_Oat", "Split_B_Oat", "Split_C_Oat", "Split_D_Oat"
]
MODES  = ["RGB", "MS"]
MODELS = ["swin", "resnet"]
TARGETS = ["gsw", "VPleaf", "VPDleaf", "Fs", "Fm'", "yield", "biomass"]

def safe_json_load(path):
    """Load JSON file even if it contains NaN or Infinity."""
    text = path.read_text()
    text = re.sub(r'\bNaN\b', 'null', text)
    text = re.sub(r'\bInfinity\b', 'null', text)
    text = re.sub(r'\b-Infinity\b', 'null', text)
    return json.loads(text)

rows = []

# --- Load all metric.json files ---
for target in TARGETS:
    for split in SPLITS:
        for mode in MODES:
            for model in MODELS:
                exp_dir = (
                    ROOT
                    / f"paper_2_SINGLEVAR_384_GB_{split}_{mode}_{target}"
                    / f"{split}__{mode}__{model}__{target}"
                )
                metrics_path = exp_dir / "metrics.json"

                if metrics_path.exists():
                    try:
                        metrics = safe_json_load(metrics_path)
                        row = {
                            "dataset": split,
                            "mode": mode,
                            "model": model,
                        }
                        row.update(metrics)
                        rows.append(row)
                    except Exception as e:
                        print(f"[WARN] Could not load {metrics_path}: {e}")
                else:
                    print(f"[MISS] No metrics.json for {metrics_path}")

if not rows:
    print("No valid metrics files found — check your paths or JSON contents.")
    raise SystemExit(1)

# --- Combine into one DataFrame ---
df = pd.DataFrame(rows)

# Drop redundant columns
drop_cols = [c for c in ["target", "avg_r2", "val_rmse"] if c in df.columns]
df = df.drop(columns=drop_cols, errors="ignore")

# --- Merge by (dataset, mode, model) ---
# This groups all targets into one combined row per experiment
agg_df = (
    df.groupby(["dataset", "mode", "model"], dropna=False)
      .first()   # since each metric file is unique per target, we can just merge columns
      .reset_index()
)

# --- Now, pivot so that each metric (like gsw_rmse, gsw_r2) stays as its own column ---
# Any duplicated columns will be automatically handled
out_path = "combined_metrics_singlevariable_384_A-D-species-merged.csv"
agg_df.to_csv(out_path, index=False)

print(f"✅ Saved {out_path} with shape {agg_df.shape}")
