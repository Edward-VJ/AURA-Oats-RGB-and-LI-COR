import os
import random
import pandas as pd
import numpy as np
import torch
import timm
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------------------
# 1) Configuration (edit these paths)
# ----------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
IMAGE_SIZE = 224

ARCH = "swin"  
MODEL_PATH = "best_multi_task_swin.pt"

# Path to your training CSV (so we can find its paired weights file)
TRAIN_CSV = "/media/edward/HDD/Workspace/Resnet/WHOLE/lam/training/train_resnet-lam-top-and-angled.csv"
# Weights CSV was saved as TRAIN_CSV minus “.csv” plus “_weights.csv”
WEIGHTS_CSV = TRAIN_CSV.replace(".csv", "_weights.csv")
TEST_CSV = "WHOLE/lam/test/test_resnet-lam-top-and-angled.csv"

# -- Directory containing your test images
# Example: "LEAF/human/test/images"
TEST_DIR = "WHOLE/lam/test/images"

TARGET_COLS = ["gsw", "gtw", "VPleaf", "VPDleaf", "H2O_leaf", "Fs", "Fm'"]

# ----------------------------------------
# 2) Load per-target weights (1/sigma) from the saved CSV
# ----------------------------------------
stats_df = pd.read_csv(WEIGHTS_CSV, index_col=0)
sigmas    = stats_df["sigma"].values.astype(np.float32)    # shape = (7,)
weights_np = 1.0 / sigmas                                  # shape = (7,)
weights_t  = torch.tensor(weights_np, device=DEVICE)       # for torch loss if needed

# ----------------------------------------
# 3) Utility: compute mean/std of test images
# ----------------------------------------
def compute_dataset_stats(image_dir):
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    rgb_acc = []
    for path in image_files:
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        non_black = flat[~np.all(flat == 0, axis=1)]
        if non_black.size > 0:
            rgb_acc.append(non_black)
    all_pix = np.concatenate(rgb_acc, axis=0)
    return all_pix.mean(axis=0).tolist(), all_pix.std(axis=0).tolist()

# ----------------------------------------
# 4) Dataset for inference
# ----------------------------------------
class MultiTargetImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.labels = df[TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_file"])
        pil_img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            normed_tensor = self.transforms(pil_img)
        else:
            normed_tensor = None
        label_tensor = torch.from_numpy(self.labels[idx])  # (7,)
        return pil_img, normed_tensor, label_tensor

# ----------------------------------------
# 5) Swin model definition
# ----------------------------------------
class MultiTaskSwin(nn.Module):
    def __init__(self, backbone_name: str, num_vars: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=[-1]
        )
        in_feats = self.backbone.feature_info.channels()[-1]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_feats, in_feats // 2),
                nn.GELU(),
                nn.Linear(in_feats // 2, 1),
            )
            for _ in range(num_vars)
        ])

    def forward(self, x):
        feats = self.backbone(x)[0].permute(0, 3, 1, 2)  # (B,C,H,W)
        pooled = feats.mean(dim=[2,3])                   # (B,C)
        outs = [h(pooled) for h in self.heads]           # list of (B,1)
        return torch.cat(outs, dim=1)                    # (B,7)

# ----------------------------------------
# 6) Metric functions
# ----------------------------------------
def evaluate_metrics(preds, targets, delta=0.05):
    err  = preds - targets
    mse  = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    small = np.abs(err) <= delta
    huber_elem = np.where(
        small,
        0.5 * err**2,
        delta * (np.abs(err) - 0.5 * delta)
    )
    return mse, rmse, float(np.mean(huber_elem))

# ----------------------------------------
# 7) Build test DataLoader
# ----------------------------------------
mu_rgb, std_rgb = compute_dataset_stats(TEST_DIR)
test_transforms = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=mu_rgb, std=std_rgb)
])

df_test = pd.read_csv(TEST_CSV)
full_test_dataset = MultiTargetImageDataset(df_test, TEST_DIR, transforms=test_transforms)

def collate_fn(batch):
    tensors = [item[1] for item in batch]  # normalized tensors
    labels  = [item[2] for item in batch]  # label tensors
    return torch.stack(tensors), torch.stack(labels)

test_loader = DataLoader(
    full_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# ----------------------------------------
# 8) Load model
# ----------------------------------------
if ARCH.lower() == "swin":
    model = MultiTaskSwin("swin_small_patch4_window7_224", num_vars=len(TARGET_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
else:
    raise ValueError("ARCH must be 'swin'")

# ----------------------------------------
# 9) Inference on test set
# ----------------------------------------
all_preds = []
all_targs = []

with torch.no_grad():
    for imgs, targs in test_loader:
        imgs, targs = imgs.to(DEVICE), targs.to(DEVICE)
        outputs = model(imgs)               # (B,7)
        all_preds.append(outputs.cpu().numpy())
        all_targs.append(targs.cpu().numpy())

all_preds_np = np.concatenate(all_preds, axis=0)  # (N,7)
all_targs_np = np.concatenate(all_targs, axis=0)  # (N,7)

# ----------------------------------------
# 10) Compute & print weighted RMSE
# ----------------------------------------
diff = (all_preds_np - all_targs_np) * weights_np[np.newaxis, :]
weighted_rmse = np.sqrt(np.mean(diff**2))
print(f"\nWeighted RMSE (test set): {weighted_rmse:.4f}")

# ----------------------------------------
# 11) Compute & print per-variable metrics
# ----------------------------------------
print("\n=== Test‐set per-variable Metrics ===")
for i, var in enumerate(TARGET_COLS):
    mse_i, rmse_i, huber_i = evaluate_metrics(all_preds_np[:,i], all_targs_np[:,i])
    print(f"{var:8s} → MSE: {mse_i:.4f}  | RMSE: {rmse_i:.4f}  | Huber: {huber_i:.4f}")

# ----------------------------------------
# 12) Visualize 9 random test images + GT vs Pred
# ----------------------------------------
indices = random.sample(range(len(full_test_dataset)), 9)
inv_norm = T.Normalize(mean=[-m/s for m,s in zip(mu_rgb,std_rgb)], std=[1/s for s in std_rgb])

fig, axes = plt.subplots(3,3,figsize=(12,12))
axes = axes.flatten()
for ax, idx in zip(axes, indices):
    pil_img, norm_tensor, gt = full_test_dataset[idx]
    gt_vals = gt.numpy().tolist()

    # predict single
    pt = norm_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(pt).cpu().squeeze(0).numpy().tolist()

    # display
    disp = inv_norm(norm_tensor).clamp(0,1)
    img_np = disp.permute(1,2,0).numpy()
    ax.imshow(img_np)
    ax.axis("off")

    gt_str   = ", ".join(f"{v:.2f}" for v in gt_vals)
    pred_str = ", ".join(f"{v:.2f}" for v in pred)
    ax.set_title(f"GT: [{gt_str}]\nPred: [{pred_str}]", fontsize=8)

plt.tight_layout()
plt.show()
