#!/usr/bin/env python
# infer_predict_export_od.py
# ---------------------------------------------------------
# Usage examples:
"""
for d in \
  experiments_od_fm \
  experiments_od_fs \
  experiments_od_gsw \
  experiments_od_h2oleaf \
  experiments_od_multiclass \
  experiments_od_vpdleaf \
  experiments_od_vpleaf
do
  echo "=============================================="
  echo "Running inference + plots for $d"
  echo "=============================================="

  python infer_predict_export_od.py \
    --results-dir "$d" \
    --dataset-root "/run/user/1000/gvfs/smb-share:server=149.157.140.139,share=public/OD1/DATASET4" \

    --runs ALL
done

for d in \
  experiments_od1-dataset4-multiclass
do
  echo "=============================================="
  echo "Running inference + plots for $d"
  echo "=============================================="

  python infer_predict_export_od.py \
    --results-dir "$d" \
    --dataset-root "/run/user/1000/gvfs/smb-share:server=149.157.140.139,share=public/OD1/DATASET4" \

    --runs ALL
done

"""
# This script:
# - Grabs candidate targets from the latest scoreboard in --results-dir
# - Filters them to columns that actually exist in the train/test CSVs
# - Recreates preprocessing/normalization exactly like training
# - Loads best.pt for each run and exports per-image GT + predictions
# - Saves scatter plots + example images for each model/target
# - Prints mean/std/variance for GT and predictions per target

import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models
import timm
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ---------- Constants (mirrors training script) ------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 16
NUM_WORKERS = 4
CROP_SIZE   = 420
IMAGE_SIZE  = 384
IMAGE_SUBDIR_DEFAULT = "images"

# ---------- Helpers ---------------------------------------
def np_slice_to_3(arr):
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] >= 3:
        return arr[..., :3]
    raise ValueError(f"Unexpected channel count {arr.shape}")

def show_examples(df, img_dir, targets, n=5, figsize_per_image=4):
    """
    Original interactive version (kept for completeness, not used now).
    """
    samples = df.sample(n).reset_index(drop=True)

    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per_image * n, figsize_per_image),
        squeeze=False
    )

    for i, row in samples.iterrows():
        ax = axes[0, i]
        img = Image.open(img_dir / row.image_file).convert("RGB")
        ax.imshow(img)

        title = "\n".join(
            f"{t}: GT={row[t+'_gt']:.3f}, Pred={row[t+'_pred']:.3f}"
            for t in targets
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    # No plt.show(); this function is not called in the automated pipeline.
    plt.close(fig)


def save_examples(df, img_dir, targets, out_path, n=5, figsize_per_image=4):
    """
    New non-interactive version: saves a single figure with n images side by side.
    """
    samples = df.sample(n).reset_index(drop=True)

    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per_image * n, figsize_per_image),
        squeeze=False
    )

    for i, row in samples.iterrows():
        ax = axes[0, i]
        img = Image.open(img_dir / row.image_file).convert("RGB")
        ax.imshow(img)

        title = "\n".join(
            f"{t}: GT={row[t+'_gt']:.3f}, Pred={row[t+'_pred']:.3f}"
            for t in targets
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def tensor_slice_to_3(t):
    if t.size(0) >= 3:
        return t[:3]
    if t.size(0) == 1:
        return t.expand(3, -1, -1)
    raise RuntimeError(f"Unexpected tensor shape {t.shape}")

def compute_image_stats(image_dir: Path):
    rgb_vals = []
    for f in image_dir.iterdir():
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        arr = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
        h, w, _ = arr.shape
        top = (h - CROP_SIZE) // 2
        left = (w - CROP_SIZE) // 2
        arr = arr[top:top+CROP_SIZE, left:left+CROP_SIZE]

        arr = np_slice_to_3(arr)
        pix = arr.reshape(-1, 3)
        non_black = pix[~np.all(pix == 0, axis=1)]
        if non_black.size:
            rgb_vals.append(non_black)

    if not rgb_vals:
        raise RuntimeError(f"No valid images in {image_dir}")

    rgb = np.concatenate(rgb_vals, 0)
    return rgb.mean(0).tolist(), rgb.std(0).tolist()


class ImageTabularDataset(Dataset):
    def __init__(self, df, img_dir, tfms, μy, σy, target_cols):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.μy = μy
        self.σy = σy
        self.target_cols = target_cols

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row.image_file).convert("RGB")
        img = self.tfms(img)
        y   = row[self.target_cols].values.astype(np.float32)
        y   = (y - self.μy) / self.σy
        return img, torch.from_numpy(y), row.image_file

def build_model(name: str, out_dim: int) -> nn.Module:
    if name == "resnet50":
        net = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, out_dim)
        return net
    if name.startswith("efficientnet"):
        net = timm.create_model(name, pretrained=True)
        if hasattr(net, "classifier") and isinstance(net.classifier, nn.Module):
            in_feats = net.classifier.in_features
            net.classifier = nn.Linear(in_feats, out_dim)
            return net
        if hasattr(net, "get_classifier"):
            in_feats = net.get_classifier().in_features
            net.reset_classifier(out_dim)
            return net
        raise ValueError(f"Unexpected EfficientNet head for {name}")
    if name.startswith("swin"):
        return timm.create_model(name, pretrained=True, num_classes=out_dim)
    raise ValueError(f"Unknown model name: {name}")

def load_state_flex(model: nn.Module, ckpt_path: Path):
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # Strip possible "module." prefixes (DataParallel)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)

def parse_targets_from_scoreboard_columns(df_scoreboard: pd.DataFrame) -> List[str]:
    """Pull candidate target names from scoreboard columns like '<target>_rmse' or '<target>_r2' in column order."""
    candidates: List[str] = []
    for c in df_scoreboard.columns:
        if c.endswith("_rmse"):
            t = c[:-5]
        elif c.endswith("_r2"):
            t = c[:-3]
        else:
            continue
        if t not in candidates:
            candidates.append(t)
    return candidates

def find_latest_scoreboard(results_dir: Path) -> Path:
    boards = sorted(results_dir.glob("scoreboard_*.csv"))
    if not boards:
        raise FileNotFoundError(f"No scoreboard_*.csv found in {results_dir}")
    return boards[-1]

def find_runs(results_dir: Path) -> List[Path]:
    return [p for p in results_dir.iterdir() if p.is_dir() and (p/"best.pt").exists()]

def parse_run_tag(run_dir_name: str):
    """
    Training saved runs as: f"{dataset_name}__{model_name}__lr{LR}"
    Example: "od1.csv__swin_large_patch4_window12_384__lr1e-05"
    """
    parts = run_dir_name.split("__")
    if len(parts) < 2:
        raise ValueError(f"Run directory name does not match expected pattern: {run_dir_name}")
    dataset_name = parts[0]
    model_name   = parts[1]
    return dataset_name, model_name

def common_numeric_columns(df_tr: pd.DataFrame, df_ts: pd.DataFrame) -> List[str]:
    common = [c for c in df_tr.columns if c in df_ts.columns]
    # Keep only numeric columns; typical non-numeric: image_file
    common_numeric = [c for c in common if is_numeric_dtype(df_tr[c])]
    # Explicitly drop known non-targets if they slipped in
    drop = {"index", "id"}
    return [c for c in common_numeric if c not in drop]

def smart_best_run_names(results_dir: Path) -> List[str]:
    """
    From the latest scoreboard in results_dir, pick the best avg_r2 (OD)
    or overall_r2 (GH) per model and build run names of the form:
        dataset__model__lr{lr}
    where lr is formatted in scientific notation as in training (e.g. 1e-05, 5e-05).
    """
    scoreboard_csv = find_latest_scoreboard(results_dir)
    df = pd.read_csv(scoreboard_csv)
    metric_col = "avg_r2" if "avg_r2" in df.columns else "overall_r2"
    if metric_col not in df.columns:
        raise RuntimeError(
            f"Metric column '{metric_col}' not found in {scoreboard_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # For each model, pick row with max metric_col
    idx = df.groupby("model")[metric_col].idxmax()
    best = df.loc[idx]

    run_names = []
    for _, row in best.iterrows():
        # lr was formatted with {:.0e} when training
        lr_str = f"{row['lr']:.0e}"
        run_name = f"{row['dataset']}__{row['model']}__lr{lr_str}"
        run_names.append(run_name)

    return run_names

def resolve_csv_paths(dataset_root: Path, dataset_name: str) -> Tuple[Path, Path]:
    """
    Robustly resolve train/test CSV paths for a given dataset_name.
    - First try the 'canonical' paths used in training:
        training/train_{dataset_name}
        test/test_{dataset_name}
    - If missing, search for any CSV in training/ and test/ that contains
      the dataset_name stem (without extension) in the filename.
    """
    train_dir = dataset_root / "training"
    test_dir  = dataset_root / "test"

    # Expected, training-style filenames
    expected_train = train_dir / f"train_{dataset_name}"
    expected_test  = test_dir  / f"test_{dataset_name}"

    if expected_train.exists() and expected_test.exists():
        return expected_train, expected_test

    # Fallback: search by stem
    stem = Path(dataset_name).stem  # e.g. "od1" from "od1.csv"
    train_candidates = list(train_dir.glob(f"*{stem}.csv"))
    test_candidates  = list(test_dir.glob(f"*{stem}.csv"))

    if len(train_candidates) == 1 and len(test_candidates) == 1:
        print(f"  ! Canonical CSVs not found; using discovered CSVs:")
        print(f"    train: {train_candidates[0]}")
        print(f"    test : {test_candidates[0]}")
        return train_candidates[0], test_candidates[0]

    msg_lines = [
        f"Could not resolve train/test CSVs for dataset_name='{dataset_name}'.",
        f"Expected either:",
        f"  {expected_train}",
        f"  {expected_test}",
        f"or exactly one matching CSV in:",
        f"  {train_dir}/*{stem}.csv",
        f"  {test_dir}/*{stem}.csv",
        f"Found {len(train_candidates)} train candidates and {len(test_candidates)} test candidates.",
    ]
    if train_candidates:
        msg_lines.append(f"Train candidates: {[str(p) for p in train_candidates]}")
    if test_candidates:
        msg_lines.append(f"Test candidates : {[str(p) for p in test_candidates]}")

    raise FileNotFoundError("\n".join(msg_lines))

def main():
    ap = argparse.ArgumentParser(description="Run inference and export predictions/GT per target.")
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="Directory containing run subfolders and a scoreboard_*.csv")
    ap.add_argument("--dataset-root", required=True, type=Path,
                    help="Root of the dataset (contains training/, validation/, test/ subfolders)")
    ap.add_argument("--image-subdir", default=IMAGE_SUBDIR_DEFAULT,
                    help=f"Name of image subdir under each split (default: {IMAGE_SUBDIR_DEFAULT})")
    ap.add_argument("--runs", nargs="+", default=["ALL"],
                    help='Run folder names to evaluate (e.g. "X.csv__resnet50__lr1e-04"). '
                         'Use ALL to auto-select best run per model from scoreboard.')
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    dataset_root: Path = args.dataset_root
    image_subdir: str = args.image_subdir
    batch_size: int = args.batch_size

    # Root plot directory
    plots_root = results_dir / "plots"
    plots_root.mkdir(exist_ok=True)

    # --------- Load latest scoreboard (for candidate target names) -----
    scoreboard_csv = find_latest_scoreboard(results_dir)
    df_score = pd.read_csv(scoreboard_csv)
    score_candidates = parse_targets_from_scoreboard_columns(df_score)
    print(f"Found scoreboard: {scoreboard_csv.name}")
    print(f"Scoreboard candidates (unfiltered): {score_candidates}")

    # --------- Discover runs to evaluate --------------------
    if len(args.runs) == 1 and args.runs[0].upper() == "ALL":
        run_names = smart_best_run_names(results_dir)
        run_dirs = [results_dir / rn for rn in run_names]
        print("Auto-selected best runs per model from scoreboard:")
        for rn in run_names:
            print(f"  - {rn}")
    else:
        run_dirs = [results_dir / r for r in args.runs]
        for rd in run_dirs:
            if not (rd/"best.pt").exists():
                raise FileNotFoundError(f"{rd}/best.pt not found")

    # --------- Evaluate each run ----------------------------
    for run_dir in run_dirs:
        run_name = run_dir.name
        dataset_name, model_name = parse_run_tag(run_name)
        print(f"\n▶ Evaluating run: {run_name}")
        print(f"   dataset_name={dataset_name}  model_name={model_name}")

        # Per-model plot directory
        model_plot_dir = plots_root / model_name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        # CSV paths following / robust to the training script convention
        csv_train, csv_test = resolve_csv_paths(dataset_root, dataset_name)
        print(f"   Using train CSV: {csv_train}")
        print(f"   Using test  CSV: {csv_test}")

        # Load CSVs
        df_tr = pd.read_csv(csv_train)
        df_ts = pd.read_csv(csv_test)

        # Determine allowed targets from dataset columns (numeric & shared)
        allowed_from_csv = common_numeric_columns(df_tr, df_ts)

        # Final targets = intersection of scoreboard candidates and allowed CSV columns
        targets = [t for t in score_candidates if t in allowed_from_csv]

        # If intersection is empty, fall back to all allowed CSV numeric columns
        if not targets:
            print("  ! No overlap between scoreboard candidates and CSV columns.")
            print(f"    Falling back to numeric columns common to train/test: {allowed_from_csv}")
            targets = allowed_from_csv

        print(f"   Using targets: {targets}")

        # Standardization params from TRAIN
        μy = df_tr[targets].mean(0).values.astype(np.float32)
        σy = (df_tr[targets].std(0).values + 1e-8).astype(np.float32)

        # Image normalization stats from TRAIN images
        img_dir_tr = dataset_root / "training" / image_subdir
        μrgb, σrgb = compute_image_stats(img_dir_tr)

        tf_eval = T.Compose([
            T.CenterCrop(CROP_SIZE),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Lambda(tensor_slice_to_3),
            T.Normalize(μrgb, σrgb),
        ])

        # Test image dir
        img_dir_ts = dataset_root / "test" / image_subdir

        # Dataset & loader
        ds_test = ImageTabularDataset(df_ts, img_dir_ts, tf_eval, μy, σy, targets)
        dl_test = DataLoader(ds_test, batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

        # Build & load model
        out_dim = len(targets)
        model = build_model(model_name, out_dim).to(DEVICE)
        load_state_flex(model, run_dir / "best.pt")
        model.eval()

        # Inference
        all_preds = []
        all_targs = []
        all_files = []
        with torch.no_grad():
            for xb, yb, files in dl_test:
                xb = xb.to(DEVICE)
                out = model(xb).cpu().numpy()
                yb  = yb.cpu().numpy()
                all_preds.append(out)
                all_targs.append(yb)
                all_files.extend(list(files))

        z_pred = np.vstack(all_preds)
        z_true = np.vstack(all_targs)

        # De-standardize to original target scales
        preds = z_pred * σy[None, :] + μy[None, :]
        trues = z_true * σy[None, :] + μy[None, :]

        # Build output DataFrame: image_file + GT/Pred columns per target
        out = pd.DataFrame({"image_file": all_files})
        for i, t in enumerate(targets):
            out[f"{t}_gt"]   = trues[:, i]
            out[f"{t}_pred"] = preds[:, i]

        # Save CSV next to run dir (per model_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = results_dir / f"predictions__{model_name}.csv"
        out.to_csv(out_csv, index=False)
        print(f"  ✔ Saved predictions to: {out_csv.resolve()}")

        # Scatter plots (one per target), saved to disk
        for t in targets:
            gt = out[f"{t}_gt"].values
            pred = out[f"{t}_pred"].values
            r, p = pearsonr(gt, pred)
            r2_test = r2_score(gt, pred)
            print(f"{t}: Pearson r = {r:.3f}, r² = {r**2:.3f}, p = {p:.3e}")
            print(f"{t}: test R² (r2_score) = {r2_test:.3f}")

            fig, ax = plt.subplots(figsize=(4,4))
            ax.scatter(gt, pred, alpha=0.6)
            lims = [
                min(gt.min(), pred.min()),
                max(gt.max(), pred.max())
            ]
            ax.plot(lims, lims, 'r--')
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.set_title(f"{model_name} — {t}\nr={r:.3f}, R²={r2_test:.3f}", fontsize=10)
            plt.tight_layout()

            scatter_path = model_plot_dir / f"{model_name}_{t}_scatter.jpg"
            fig.savefig(scatter_path, dpi=200)
            plt.close(fig)

        # Example images panel saved to disk
        examples_path = model_plot_dir / f"{model_name}_examples.jpg"
        save_examples(out, img_dir_ts, targets, examples_path, n=5)

        # Print summary stats (mean, std, var) for GT and Pred per target
        print("  ── Summary stats (GT vs Pred) ──────────────────────────────")
        rows = []
        for t in targets:
            gt_vals   = out[f"{t}_gt"].values
            pred_vals = out[f"{t}_pred"].values
            rows.append({
                "target": t,
                "gt_mean":   float(np.mean(gt_vals)),
                "gt_std":    float(np.std(gt_vals, ddof=0)),
                "gt_var":    float(np.var(gt_vals, ddof=0)),
                "pred_mean": float(np.mean(pred_vals)),
                "pred_std":  float(np.std(pred_vals, ddof=0)),
                "pred_var":  float(np.var(pred_vals, ddof=0)),
            })
        stats_df = pd.DataFrame(rows, columns=[
            "target","gt_mean","gt_std","gt_var","pred_mean","pred_std","pred_var"
        ])
        with pd.option_context("display.max_rows", None,
                               "display.width", 140,
                               "display.precision", 6):
            print(stats_df.to_string(index=False))
        print("  ────────────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
