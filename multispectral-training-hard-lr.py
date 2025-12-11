#!/usr/bin/env python
# train_compare.py – 5-channel input (RGB+RE+NIR) + masked per-target weighted RMSE + ETA progress
# ---------------------------------------------------------
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.models as tv_models
import timm

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# ========= USER CONFIG ====================================
def make_dataset(name, root):
    return dict(
        name=name,
        root=root,
        csv_train="train.csv",
        csv_val="val.csv",
        csv_test="test.csv",
        targets=["gsw", "gtw", "VPleaf", "VPDleaf", "H2O_leaf", "Fs", "Fm'", "yield", "biomass"],
    )

# DATASETS = [
#     make_dataset("Split_1",             "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split1"),
#     make_dataset("Split_1_Filtered",    "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split1-filt"),
#     make_dataset("Split_2",             "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split2"),
#     make_dataset("Split_2_Filtered",    "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split2-filt"),
#     make_dataset("Split_3",             "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split3"),
#     make_dataset("Split_3_Filtered",    "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split3-filt"),
#     make_dataset("Split_4",             "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split4"),
#     make_dataset("Split_4_Filtered",    "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final/split4-filt"),
# ]

DATASETS = [
    make_dataset("Split_A", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitA"),
    make_dataset("Split_B", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitB"),
    make_dataset("Split_C", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitC"),
    make_dataset("Split_D", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitD"),

    # --- added species-specific splits ---
    make_dataset("Split_A_Oat", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitA_Oat"),
    make_dataset("Split_B_Oat", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitB_Oat"),
    make_dataset("Split_C_Oat", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitC_Oat"),
    make_dataset("Split_D_Oat", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitD_Oat"),

    make_dataset("Split_A_Barley", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitA_Barley"),
    make_dataset("Split_B_Barley", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitB_Barley"),
    make_dataset("Split_C_Barley", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitC_Barley"),
    make_dataset("Split_D_Barley", "/media/edward/HDD/Workspace/OAT_PAPER_2/licor_reading_interpolation/splits_idw_final_fixed/splitD_Barley"),
]



BASE_DIR = Path("/media/edward/HDD/Workspace/OAT_PAPER_2/DATA")

MODEL_ALIASES = {
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b5",
    "swin": "swin_large_patch4_window12_384",
}
MODELS = list(MODEL_ALIASES.keys())
INPUT_MODES = ["ms"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 75
NUM_WORKERS = 16
CROP_SIZE = 384
IMAGE_SIZE = 384
# ==========================================================

# ---------------- Path helpers ----------------
def rgb_to_re_path(rel_path: str) -> Path:
    return BASE_DIR / Path(rel_path.replace("/rgb", "/reg_right"))

def rgb_to_nir_path(rel_path: str) -> Path:
    return BASE_DIR / Path(rel_path.replace("/rgb", "/reg_left"))

# ---------------- Image stats (5ch) ----------------
def compute_image_stats_5ch(image_paths):
    r_list, g_list, b_list, re_list, nir_list = [], [], [], [], []
    for rel in image_paths:
        rgb_p = BASE_DIR / rel
        re_p  = rgb_to_re_path(rel)
        nir_p = rgb_to_nir_path(rel)
        if not (rgb_p.exists() and re_p.exists() and nir_p.exists()):
            continue

        rgb = np.asarray(Image.open(rgb_p).convert("RGB"), np.float32) / 255.0
        re  = np.asarray(Image.open(re_p).convert("L"),   np.float32) / 255.0
        nir = np.asarray(Image.open(nir_p).convert("L"),  np.float32) / 255.0

        h, w = rgb.shape[:2]
        top  = max((h - CROP_SIZE) // 2, 0)
        left = max((w - CROP_SIZE) // 2, 0)
        rgb = rgb[top:top+CROP_SIZE, left:left+CROP_SIZE]
        re  = re [top:top+CROP_SIZE, left:left+CROP_SIZE]
        nir = nir[top:top+CROP_SIZE, left:left+CROP_SIZE]

        rgb_flat = rgb.reshape(-1, 3)
        mask_rgb = ~np.all(rgb_flat == 0, axis=1)
        rgb_flat = rgb_flat[mask_rgb]
        if rgb_flat.size:
            r_list.append(rgb_flat[:, 0]); g_list.append(rgb_flat[:, 1]); b_list.append(rgb_flat[:, 2])

        re_flat  = re.reshape(-1)
        nir_flat = nir.reshape(-1)
        re_flat  = re_flat [re_flat  > 0]
        nir_flat = nir_flat[nir_flat > 0]
        if re_flat.size:  re_list.append(re_flat)
        if nir_flat.size: nir_list.append(nir_flat)

    def _mstd(vs):
        if not vs: return 0.5, 0.25
        v = np.concatenate(vs, 0)
        return float(v.mean()), float(v.std() + 1e-8)

    r_m, r_s = _mstd(r_list)
    g_m, g_s = _mstd(g_list)
    b_m, b_s = _mstd(b_list)
    re_m, re_s = _mstd(re_list)
    nir_m, nir_s = _mstd(nir_list)

    mean5 = [r_m, g_m, b_m, re_m, nir_m]
    std5  = [r_s, g_s, b_s, re_s, nir_s]
    return mean5, std5

# ---------------- Multi-modal augment ----------------
class MultiModalAugment:
    def __init__(self, train: bool, mean5, std5):
        self.train = train
        self.mean  = torch.tensor(mean5, dtype=torch.float32).view(-1,1,1)
        self.std   = torch.tensor(std5,  dtype=torch.float32).view(-1,1,1)

    def __call__(self, rgb_pil: Image.Image, re_pil: Image.Image, nir_pil: Image.Image):
        w, h = rgb_pil.size
        top  = max((h - CROP_SIZE) // 2, 0)
        left = max((w - CROP_SIZE) // 2, 0)
        rgb = F.crop(rgb_pil, top, left, CROP_SIZE, CROP_SIZE)
        re  = F.crop(re_pil,  top, left, CROP_SIZE, CROP_SIZE)
        nir = F.crop(nir_pil, top, left, CROP_SIZE, CROP_SIZE)

        if self.train:
            if torch.rand(1).item() < 0.5:
                rgb = F.hflip(rgb); re = F.hflip(re); nir = F.hflip(nir)
            if torch.rand(1).item() < 0.5:
                rgb = F.vflip(rgb); re = F.vflip(re); nir = F.vflip(nir)


        angle = (torch.rand(1).item() * 30.0 - 15.0) if self.train else 0.0
        rgb = F.rotate(rgb, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        re  = F.rotate(re,  angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        nir = F.rotate(nir, angle, interpolation=InterpolationMode.BILINEAR, fill=0)

        # size = [IMAGE_SIZE, IMAGE_SIZE]
        # rgb = F.resize(rgb, size, interpolation=InterpolationMode.BILINEAR)
        # re  = F.resize(re,  size, interpolation=InterpolationMode.BILINEAR)
        # nir = F.resize(nir, size, interpolation=InterpolationMode.BILINEAR)

        rgb_t = F.to_tensor(rgb)
        re_t  = F.to_tensor(re)
        nir_t = F.to_tensor(nir)
        x = torch.cat([rgb_t, re_t, nir_t], dim=0)

        x = (x - self.mean) / self.std
        return x

# ---------------- Dataset ----------------
class ImageTabularDataset(Dataset):
    def __init__(self, df, base_dir, tfms, y_mean, y_std, target_cols, mode="ms"):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.tfms = tfms
        self.y_mean = y_mean.astype(np.float32)
        self.y_std  = y_std.astype(np.float32)
        self.target_cols = target_cols
        self.mode = mode

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb_path = self.base_dir / row.image_file

        if self.mode == "ms":
            re_path  = rgb_to_re_path(row.image_file)
            nir_path = rgb_to_nir_path(row.image_file)
            rgb = Image.open(rgb_path).convert("RGB")
            re  = Image.open(re_path).convert("L")
            nir = Image.open(nir_path).convert("L")
            x = self.tfms(rgb, re, nir)
        else:
            rgb = Image.open(rgb_path).convert("RGB")
            x = self.tfms(rgb)

        y = row[self.target_cols].to_numpy(dtype=np.float32)
        m = ~np.isnan(y)
        y[m] = (y[m] - self.y_mean[m]) / self.y_std[m]
        return x, torch.from_numpy(y)

# ---------------- Models ----------------
def build_model(name, out_dim, in_chans=5):
    real_name = MODEL_ALIASES[name]
    if real_name == "resnet50":
        net = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        conv1 = net.conv1
        new_conv = nn.Conv2d(in_chans, conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride, padding=conv1.padding,
                            bias=(conv1.bias is not None))
        with torch.no_grad():
            if in_chans == 3:
                new_conv.weight[:, :3] = conv1.weight
            else:
                new_conv.weight[:, :3] = conv1.weight
                mean_w = conv1.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:in_chans] = mean_w.repeat(1, in_chans-3, 1, 1)
            if conv1.bias is not None:
                new_conv.bias.copy_(conv1.bias)
        net.conv1 = new_conv
        net.fc = nn.Linear(net.fc.in_features, out_dim)
        return net

    if real_name.startswith("efficientnet"):
        net = timm.create_model(real_name, pretrained=True, in_chans=in_chans, num_classes=out_dim)
        return net

    if real_name.startswith("swin"):
        net = timm.create_model(real_name, pretrained=True, in_chans=in_chans, num_classes=out_dim)
        net.set_grad_checkpointing(True)
        return net

    raise ValueError(real_name)

# ---------------- Loss ----------------
class MaskedWeightedRMSE(nn.Module):
    def __init__(self, weights=None, eps=1e-8):
        super().__init__()
        if weights is not None:
            w = torch.as_tensor(weights, dtype=torch.float32)
            self.register_buffer("w", w)
        else:
            self.w = None
        self.eps = eps

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        diff = torch.where(mask, pred - target, 0.0)
        err2 = diff**2
        if getattr(self, "w", None) is not None:
            w = torch.nan_to_num(self.w, nan=0.0).to(pred.device).view(1, -1).expand_as(err2)
            err2 = err2 * w
        valid = mask.sum().clamp_min(1)
        return torch.sqrt(err2.sum() / valid + self.eps)

# ---------------- Metrics ----------------
def nan_rmse(p, t):
    mask = ~np.isnan(t)
    if mask.sum() == 0: return float("nan")
    return float(np.sqrt(((p[mask]-t[mask])**2).mean()))

def nan_r2(p, t):
    mask = ~np.isnan(t)
    if mask.sum() < 2: return float("nan")
    p, t = p[mask], t[mask]
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    return float(1 - ss_res/ss_tot) if ss_tot else float("nan")

# ---------------- Batch size finder ----------------
def find_max_batch_size(model, dataset, loss_fn, start=8, max_bs=512):
    bs = start
    best = start
    sample_loader = DataLoader(dataset, batch_size=start, shuffle=True)
    xb, yb = next(iter(sample_loader))
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)

    while bs <= max_bs:
        try:
            xb_rep = xb.repeat(bs, 1, 1, 1)[:bs]
            yb_rep = yb.repeat(bs, 1)[:bs]

            if DEVICE.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    out = model(xb_rep)
                    loss = loss_fn(out, yb_rep)
            else:
                out = model(xb_rep); loss = loss_fn(out, yb_rep)

            model.zero_grad(set_to_none=True)
            loss.backward()
            model.zero_grad(set_to_none=True)

            best = bs
            bs *= 2

            del xb_rep, yb_rep, out, loss
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[INFO] Max batch size for {type(model).__name__}: {best}")
                return max(int(best*0.5), 1)
            raise
    print(f"[INFO] Max batch size for {type(model).__name__}: {best}")
    return max(int(best*0.5), 1)

# ---------------- Epoch loop ----------------
def run_epoch(loader, model, loss_fn, opt=None):
    model.train(opt is not None)
    total, preds, targs = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.set_grad_enabled(opt is not None):
            out = model(xb)
            loss = loss_fn(out, yb)
            if opt:
                opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * xb.size(0)
        preds.append(out.detach().cpu().numpy())
        targs.append(yb.cpu().numpy())
    n = len(loader.dataset)
    return total/max(n,1), np.concatenate(preds), np.concatenate(targs)

# ---------------- Main ----------------
def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scoreboard = []

    fixed_lrs = {"swin": 2e-5, "resnet": 7e-4, "efficientnet": 3e-3}

    for cfg in DATASETS:
        for mode in INPUT_MODES:
            print(f"\n[INFO] Dataset {cfg['name']} – Mode={mode.upper()}")
            root = Path(cfg["root"])
            df_tr  = pd.read_csv(root / cfg["csv_train"])
            df_val = pd.read_csv(root / cfg["csv_val"])
            df_ts  = pd.read_csv(root / cfg["csv_test"])
            targets = cfg["targets"]

            y_mean = np.nanmean(df_tr[targets].to_numpy(np.float32), axis=0)
            y_std  = np.nanstd(df_tr[targets].to_numpy(np.float32), axis=0) + 1e-8

            sigma = np.nanstd(df_tr[targets].to_numpy(np.float32), axis=0)
            weights = np.where(np.isfinite(sigma), 1.0/(sigma+1e-8), 0.0).astype(np.float32)
            loss_fn_ctor = lambda: MaskedWeightedRMSE(weights)

            if mode == "ms":
                mean5, std5 = [0.5]*5, [0.25]*5
                tf_train = MultiModalAugment(train=True,  mean5=mean5, std5=std5)
                tf_eval  = MultiModalAugment(train=False, mean5=mean5, std5=std5)
            else:
                mean3, std3 = [0.5]*3, [0.25]*3
                def tf_rgb(train):
                    def f(rgb):
                        w, h = rgb.size
                        top  = max((h - CROP_SIZE)//2, 0)
                        left = max((w - CROP_SIZE)//2, 0)
                        rgb_c = F.crop(rgb, top, left, CROP_SIZE, CROP_SIZE)
                        if train and torch.rand(1).item() < 0.5:
                            rgb_c = F.hflip(rgb_c)
                        angle = (torch.rand(1).item()*30.0 - 15.0) if train else 0.0
                        rgb_c = F.rotate(rgb_c, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
                        rgb_c = F.resize(rgb_c, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)
                        t = F.to_tensor(rgb_c)
                        t = (t - torch.tensor(mean3).view(-1,1,1)) / torch.tensor(std3).view(-1,1,1)
                        return t
                    return f
                tf_train = tf_rgb(True)
                tf_eval  = tf_rgb(False)

            ds_train = ImageTabularDataset(df_tr, BASE_DIR, tf_train, y_mean, y_std, targets, mode=mode)
            ds_val   = ImageTabularDataset(df_val, BASE_DIR, tf_eval,  y_mean, y_std, targets, mode=mode)
            ds_test  = ImageTabularDataset(df_ts, BASE_DIR, tf_eval,  y_mean, y_std, targets, mode=mode)

            RESULTS_DIR = Path(f"paper_2_FINAL_MULTIVARIABLE_384_{cfg['name']}_{mode.upper()}")
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            for mname in MODELS:
                try:
                    print(f"\n▶ {cfg['name']}__{mode.upper()}__{mname}")
                    tag = f"{cfg['name']}__{mode.upper()}__{mname}"
                    run_dir = RESULTS_DIR / tag
                    run_dir.mkdir(parents=True, exist_ok=True)
                    writer  = SummaryWriter(run_dir/"tb")

                    in_chans = 5 if mode == "ms" else 3
                    model = build_model(mname, out_dim=len(targets), in_chans=in_chans).to(DEVICE)
                    loss_fn = loss_fn_ctor()

                    BATCH_SIZE = find_max_batch_size(model, ds_train, loss_fn)
                    if mname == "swin":
                        print(f"[INFO] Overriding batch size for swin → 1/2")
                        BATCH_SIZE = int(BATCH_SIZE*0.5)
                    else:
                        print(f"[INFO] Using batch size {BATCH_SIZE} for {mname}")

                    dl_train = DataLoader(ds_train, BATCH_SIZE, True,  num_workers=NUM_WORKERS, pin_memory=True)
                    dl_val   = DataLoader(ds_val,   BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)
                    dl_test  = DataLoader(ds_test,  BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)

                    # ---- Hardcoded LR ----
                    best_lr = fixed_lrs[mname]
                    print(f"[INFO] Using fixed LR={best_lr:.2e} for {mname}")
                    opt = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=1e-4)

                    best_val, best_path = np.inf, run_dir/"best.pt"
                    block_start = time.time()
                    for ep in trange(1, NUM_EPOCHS+1, desc=f"Epochs {mname}", ncols=100):
                        tr_l,_,_ = run_epoch(dl_train, model, loss_fn, opt)
                        val_l,_,_ = run_epoch(dl_val,   model, loss_fn)
                        writer.add_scalars("loss", {"train":tr_l,"val":val_l}, ep)
                        if val_l < best_val:
                            best_val = val_l
                            torch.save(model.state_dict(), best_path)
                        if ep % 10 == 0:
                            block_time = time.time() - block_start
                            print(f"[INFO] Epoch {ep:03d}/{NUM_EPOCHS} | train {tr_l:.4f} val {val_l:.4f} | last 10 epochs took {block_time:.1f}s")
                            block_start = time.time()

                    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
                    _, p_z, t_z = run_epoch(dl_test, model, loss_fn)
                    preds = p_z * y_std + y_mean
                    targs = t_z * y_std + y_mean

                    row = {"dataset":cfg["name"], "mode":mode, "model":mname, "lr":best_lr, "val_rmse":best_val}
                    r2_vals=[]
                    for i,col in enumerate(targets):
                        r   = nan_rmse(preds[:,i], targs[:,i])
                        r2v = nan_r2  (preds[:,i], targs[:,i])
                        row[f"{col}_rmse"] = r
                        row[f"{col}_r2"]   = r2v
                        if not np.isnan(r2v): r2_vals.append(r2v)
                    row["avg_r2"] = float(np.mean(r2_vals)) if r2_vals else float("nan")
                    scoreboard.append(row)

                    with open(run_dir/"metrics.json","w") as fp:
                        json.dump(row, fp, indent=2)
                    for k,v in row.items():
                        if isinstance(v,(int,float)) and np.isfinite(v):
                            writer.add_scalar(f"test/{k}", v)
                    writer.close()
                    print(f"  ✔ avg R²={row['avg_r2']:.3f}")

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[OOM] Skipping {cfg['name']} {mode.upper()} {mname} due to OOM.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

    out_csv = Path(f"scoreboard_{stamp}.csv")
    pd.DataFrame(scoreboard).to_csv(out_csv, index=False)
    print(f"\n🏁 Done – scoreboard saved to {out_csv.resolve()}")

if __name__ == "__main__":
    main()
