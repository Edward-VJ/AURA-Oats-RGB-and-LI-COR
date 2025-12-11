#!/usr/bin/env python
# train_compare.py  (RMSE version)
# ---------------------------------------------------------
# Unified training / validation / testing for ResNet‑50,
# EfficientNet‑B3, and Swin‑Large – using RMSE loss only.
# ---------------------------------------------------------
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as tv_models
import timm

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ========= USER CONFIG –‑‑‑ EDIT ME ======================

def make_dataset(name, root, weight_csv=None):
    return dict(
        name=name,
        root=root,
        csv_train=f"training/train_{name}",
        csv_val=f"validation/validation_{name}",
        csv_test=f"test/test_{name}",
        image_subdir="images",
        targets=["Fm'"],#, "gtw", "VPleaf", "VPDleaf", "H2O_leaf", "Fs", "Fm'"],
        weight_csv=weight_csv  # optional CSV with 'sigma' column
    )
DATASETS = [
    make_dataset(
        name="resnet-lam-top-only-all-days.csv",
        root="/media/edward/HDD/Workspace/Resnet/WHOLE/lam"
    ),
]

RESULTS_DIR  = Path("experiments_gh_fm")
LEARNING_RATES = [5e-3, 1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

MODELS = [
    "resnet50",
    "efficientnet_b5",
    "swin_large_patch4_window12_384",
]
# =========== END OF USER CONFIG ==========================

# ---------- Hyper‑parameters -----------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 8
NUM_EPOCHS   = 100
NUM_WORKERS  = 4
IMAGE_SIZE   = 384

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Utility: compute RGB μ/σ ignoring black pixels
# ---------------------------------------------------------
def compute_image_stats(image_dir: Path):
    rgb_vals = []
    for f in image_dir.iterdir():
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        arr = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
        pix = arr.reshape(-1, 3)
        non_black = pix[~np.all(pix == 0, axis=1)]
        if non_black.size:
            rgb_vals.append(non_black)
    rgb = np.concatenate(rgb_vals, 0)
    return rgb.mean(0).tolist(), rgb.std(0).tolist()

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class ImageTabularDataset(Dataset):
    def __init__(self, df, img_dir, tfms, label_means, label_stds, target_cols):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.label_means = label_means
        self.label_stds = label_stds
        self.target_cols = target_cols

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row.image_file).convert("RGB")
        img = self.tfms(img)
        y = row[self.target_cols].values.astype(np.float32)
        y = (y - self.label_means) / self.label_stds
        return img, torch.from_numpy(y)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
def build_model(name, out_dim):
    if name == "resnet50":
        net = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, out_dim)
        return net
    if name.startswith("efficientnet"):
        net = timm.create_model(name, pretrained=True)
        net.classifier = nn.Linear(net.classifier.in_features, out_dim)
        return net
    if name.startswith("swin"):
        return timm.create_model(name, pretrained=True, num_classes=out_dim)
    raise ValueError(name)

# ---------------------------------------------------------
# Loss – weighted RMSE (weights optional)
# ---------------------------------------------------------
class WeightedRMSE(nn.Module):
    def __init__(self, weights=None, eps=1e-8):
        super().__init__()
        self.register_buffer("w", None if weights is None
                                   else torch.as_tensor(weights, dtype=torch.float32))
        self.eps = eps
    def forward(self, pred, target):
        err2 = (pred - target)**2
        if self.w is not None:
            err2 = err2 * self.w
        return torch.sqrt(err2.mean() + self.eps)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def rmse(pred, targ): return float(np.sqrt(((pred - targ)**2).mean()))
def r2(pred, targ):
    ss_res = np.sum((targ - pred)**2)
    ss_tot = np.sum((targ - targ.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot else float("nan")

# ---------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------
def run_epoch(loader, model, loss_fn, opt=None):
    model.train(opt is not None)
    running = 0.0
    preds, targs = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.set_grad_enabled(opt is not None):
            out = model(xb)
            loss = loss_fn(out, yb)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
        running += loss.item() * xb.size(0)
        preds.append(out.detach().cpu().numpy())
        targs.append(yb.cpu().numpy())
    n = len(loader.dataset)
    return running / n, np.concatenate(preds), np.concatenate(targs)

# ---------------------------------------------------------
# Main driver
# ---------------------------------------------------------
def main():
    scoreboard = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for cfg in DATASETS:
        root = Path(cfg["root"])
        df_tr  = pd.read_csv(root / cfg["csv_train"])
        df_val = pd.read_csv(root / cfg["csv_val"])
        df_ts  = pd.read_csv(root / cfg["csv_test"])

        targets = cfg["targets"]

        mu_y = df_tr[targets].mean(0).values
        sd_y = df_tr[targets].std(0).values + 1e-8

        if cfg.get("weight_csv"):
            sigma = pd.read_csv(root / cfg["weight_csv"], index_col=0)["sigma"].values
            weights = 1.0 / (sigma + 1e-8)
        else:
            weights = None

        img_dir_tr = root / Path(cfg["csv_train"]).parent / cfg["image_subdir"]
        mu_rgb, sd_rgb = compute_image_stats(img_dir_tr)

        tf_train = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mu_rgb, sd_rgb)
        ])
        tf_eval = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mu_rgb, sd_rgb)
        ])

        ds_train = ImageTabularDataset(df_tr, img_dir_tr, tf_train, mu_y, sd_y, target_cols=targets)
        ds_val   = ImageTabularDataset(df_val,
                       root / Path(cfg["csv_val"]).parent / cfg["image_subdir"],
                       tf_eval, mu_y, sd_y, target_cols=targets)
        ds_test  = ImageTabularDataset(df_ts,
                       root / Path(cfg["csv_test"]).parent / cfg["image_subdir"],
                       tf_eval, mu_y, sd_y, target_cols=targets)

        dl_train = DataLoader(ds_train, BATCH_SIZE, True,  num_workers=NUM_WORKERS, pin_memory=True)
        dl_val   = DataLoader(ds_val,   BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)
        dl_test  = DataLoader(ds_test,  BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)

        for mname in MODELS:
            for lr in LEARNING_RATES:
                tag = f"{cfg['name']}__{mname}__lr{lr:.0e}"
                print(f"\n▶ {tag}")
                run_dir = RESULTS_DIR / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                writer = SummaryWriter(run_dir/"tb")

                model = build_model(mname, len(targets)).to(DEVICE)
                opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                loss_fn = WeightedRMSE(weights)

                best, best_path = np.inf, run_dir/"best.pt"
                for ep in range(1, NUM_EPOCHS+1):
                    tr_l, _, _ = run_epoch(dl_train, model, loss_fn, opt)
                    val_l, _, _ = run_epoch(dl_val,   model, loss_fn)
                    writer.add_scalars("loss", {"train": tr_l, "val": val_l}, ep)
                    if val_l < best:
                        best = val_l
                        torch.save(model.state_dict(), best_path)
                    if ep % 10 == 0 or ep == NUM_EPOCHS:
                        print(f"  {ep:03d}/{NUM_EPOCHS}  train {tr_l:.4f}  val {val_l:.4f}")

                # ---- Test phase -------------------------------------------------
                model.load_state_dict(torch.load(best_path))
                _, preds_z, targs_z = run_epoch(dl_test, model, loss_fn)
                preds = preds_z * sd_y + mu_y
                targs = targs_z * sd_y + mu_y

                row = dict(dataset=cfg["name"], model=mname, lr=lr,
                           val_rmse=best,
                           overall_rmse=rmse(preds, targs),
                           overall_r2=r2(preds.ravel(), targs.ravel()))
                for i, col in enumerate(targets):
                    row[f"{col}_rmse"] = rmse(preds[:, i], targs[:, i])
                    row[f"{col}_r2"]   = r2(preds[:, i], targs[:, i])
                scoreboard.append(row)

                with open(run_dir/"metrics.json", "w") as fp:
                    json.dump(row, fp, indent=2)
                for k,v in row.items():
                    if isinstance(v,(int,float)):
                        writer.add_scalar(f"test/{k}", v)
                writer.close()
                print(f"  ✔ test RMSE={row['overall_rmse']:.3f}  R²={row['overall_r2']:.3f}")

    out_csv = RESULTS_DIR / f"scoreboard_{stamp}.csv"
    pd.DataFrame(scoreboard).to_csv(out_csv, index=False)
    print(f"\n🏁 Done – scoreboard saved to {out_csv.resolve()}")

if __name__ == "__main__":
    main()
