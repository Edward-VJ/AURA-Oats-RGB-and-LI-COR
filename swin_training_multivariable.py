import os
import pandas as pd
import numpy as np
import torch
import timm                # pip install timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ----------------------------------------
# 0) Weighted RMSE criterion
# ----------------------------------------
def weighted_rmse_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
    """
    preds, targets: (B, V)
    weights:       (V,) inverse‐std per variable
    Returns scalar WRMSE = sqrt(mean((w_i*(pred_i - targ_i))^2)))
    """
    diff = (preds - targets) * weights.unsqueeze(0)   # (B, V)
    mse  = (diff**2).mean()                            # mean over B×V
    return torch.sqrt(mse)

# -------------------------
# 1) Config
# -------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
IMAGE_SIZE = 384
NUM_EPOCHS = 30
LEARNING_RATE = 3e-5
NUM_WORKERS = 4

BACKBONE_NAME = "swin_large_patch4_window12_384"
TARGET_COLS   = ["gsw","gtw","VPleaf","VPDleaf","H2O_leaf","Fs","Fm'"]
NUM_VARS      = len(TARGET_COLS)

TRAIN_CSV = "/media/edward/HDD/Workspace/Resnet/WHOLE/lam/training/train_resnet-lam-top-and-angled.csv"
VAL_CSV   = "/media/edward/HDD/Workspace/Resnet/WHOLE/lam/validation/validation_resnet-lam-top-and-angled.csv"
TRAIN_DIR = "/media/edward/HDD/Workspace/Resnet/WHOLE/lam/training/images"
VAL_DIR   = "/media/edward/HDD/Workspace/Resnet/WHOLE/lam/validation/images"

# -------------------------
# 2) Compute image stats
# -------------------------
def compute_image_stats(image_dir):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(("png","jpg","jpeg"))]
    acc   = []
    for f in files:
        arr = np.array(Image.open(os.path.join(image_dir,f)).convert("RGB"), np.float32)/255.0
        pix = arr.reshape(-1,3)
        non_black = pix[~np.all(pix==0,axis=1)]
        if non_black.size: acc.append(non_black)
    pix_all = np.concatenate(acc,axis=0)
    return pix_all.mean(axis=0).tolist(), pix_all.std(axis=0).tolist()

# -------------------------
# 3) Dataset
# -------------------------
class MultiTargetLeafDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.df         = df.reset_index(drop=True)
        self.image_dir  = image_dir
        self.transforms = transforms
        self.labels     = df[TARGET_COLS].values.astype(np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img      = Image.open(os.path.join(self.image_dir, row["image_file"])).convert("RGB")
        if self.transforms: img = self.transforms(img)
        label    = torch.from_numpy(self.labels[idx])  # (7,)
        return img, label

# -------------------------
# 4) Model
# -------------------------
class MultiTaskSwin(nn.Module):
    def __init__(self, backbone_name, num_vars):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True, out_indices=[-1]
        )
        in_feats = self.backbone.feature_info.channels()[-1]
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_feats, in_feats//2), nn.GELU(), nn.Linear(in_feats//2, 1))
            for _ in range(num_vars)
        ])

    def forward(self, x):
        feats = self.backbone(x)[0].permute(0,3,1,2)  # (B,C,H,W)
        pooled= feats.mean(dim=[2,3])                # (B,C)
        outs  = [h(pooled) for h in self.heads]       # list of (B,1)
        return torch.cat(outs, dim=1)                 # (B, V)

# -------------------------
# 5) Metrics
# -------------------------
def evaluate_metrics(preds, targs, delta=0.05):
    err   = preds - targs
    mse   = float((err**2).mean())
    rmse  = float(np.sqrt(mse))
    small = np.abs(err) <= delta
    huber = np.where(small, 0.5*err**2, delta*(np.abs(err)-0.5*delta))
    return mse, rmse, float(huber.mean())

# -------------------------
# 6) Training / Validation
# -------------------------
def train_one_epoch(model, loader, optimizer, weights):
    model.train()
    total = 0.0
    pbar  = tqdm(loader, desc="Training", ncols=80)
    for imgs, targs in pbar:
        imgs, targs = imgs.to(DEVICE), targs.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = weighted_rmse_loss(preds, targs, weights)
        loss.backward()
        optimizer.step()
        total += loss.item()
        pbar.set_postfix(train_wrmse=total/(pbar.n+1))
    return total/len(loader)

def validate(model, loader, weights):
    model.eval()
    total = 0.0
    all_p, all_t = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", ncols=80)
        for imgs, targs in pbar:
            imgs, targs = imgs.to(DEVICE), targs.to(DEVICE)
            preds = model(imgs)
            total += weighted_rmse_loss(preds, targs, weights).item()
            all_p.append(preds.cpu().numpy())
            all_t.append(targs.cpu().numpy())
    all_p = np.concatenate(all_p,0); all_t = np.concatenate(all_t,0)
    avg_wrmse = total/len(loader)
    per_var = {}
    for i,var in enumerate(TARGET_COLS):
        mse_i, rmse_i, hub_i = evaluate_metrics(all_p[:,i], all_t[:,i])
        per_var[var] = {"mse":mse_i,"rmse":rmse_i,"huber":hub_i}
    return avg_wrmse, per_var

# -------------------------
# 7) Main
# -------------------------
if __name__ == "__main__":
    writer = SummaryWriter("runs/leaf_swin_wrmse")

    # Load CSVs
    import timm; print("timm v", timm.__version__)
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # --- load per-target σ from your weights CSV
    base, ext = os.path.splitext(TRAIN_CSV)
    stats_csv  = base + "_weights" + ext
    stats      = pd.read_csv(stats_csv, index_col=0)
    sigmas     = torch.tensor(stats["sigma"].values, dtype=torch.float32, device=DEVICE)
    weights    = 1.0 / sigmas

    # Compute and log image normalization
    mu_rgb, std_rgb = compute_image_stats(TRAIN_DIR)
    print("Image mean:", mu_rgb, "std:", std_rgb)

    # Transforms
    train_tf = T.Compose([
        T.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation((0,90)),
        T.ColorJitter(0.1,0.25,0.25,0.04),
        T.ToTensor(),
        T.Normalize(mu_rgb, std_rgb),
    ])
    val_tf = T.Compose([
        T.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mu_rgb, std_rgb),
    ])

    # Datasets & Loaders
    train_ds = MultiTargetLeafDataset(train_df, TRAIN_DIR, transforms=train_tf)
    val_ds   = MultiTargetLeafDataset(val_df, VAL_DIR, transforms=val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model & optimizer
    model     = MultiTaskSwin(BACKBONE_NAME, NUM_VARS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Training loop
    best_val = float("inf")
    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        tr_wrmse = train_one_epoch(model, train_loader, optimizer, weights)
        val_wrmse, var_metrics = validate(model, val_loader, weights)

        writer.add_scalar("Loss/Train_WRMSE", tr_wrmse, epoch)
        writer.add_scalar("Loss/Val_WRMSE",   val_wrmse, epoch)
        for var,m in var_metrics.items():
            writer.add_scalar(f"Val_RMSE/{var}", m["rmse"], epoch)

        print(f"Epoch {epoch} → Train WRMSE: {tr_wrmse:.4f}  |  Val WRMSE: {val_wrmse:.4f}")
        for var, m in var_metrics.items():
            print(f"  {var:8s} → RMSE: {m['rmse']:.3f}")

        if val_wrmse < best_val:
            best_val = val_wrmse
            torch.save(model.state_dict(), "best_multi_task_swin_large.pt")
            print("  → Saved best checkpoint.")

    writer.close()
    print(f"\nTraining complete. Best validation WRMSE: {best_val:.4f}")
