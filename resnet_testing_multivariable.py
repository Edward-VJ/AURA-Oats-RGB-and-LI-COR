import os
import json
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from deepml.tasks import ImageRegression
from deepml.train import Learner
from deepml.losses import RMSELoss
import torch.nn as nn

# 1) Subclass ImageRegression so no .item() errors on a 7-vector
class MultiTargetRegression(ImageRegression):
    def transform_target(self, y: torch.Tensor):
        # y is shape (7,)
        return [ round(val.item(), 2) for val in y ]

    def transform_output(self, y: torch.Tensor):
        return [ round(val.item(), 2) for val in y ]


# 2) Custom Dataset that returns (image, 7-vector-of-raw-labels)
class MultiTargetImageDataset(Dataset):
    """
    Expects `df` to have exactly these seven columns (raw units):
      ["gsw", "gtw", "VPleaf", "VPDleaf", "H2O_leaf", "Fs", "Fm'"]
    plus an "image_file" column.  root_dir is the folder containing images.
    transforms should match whatever image normalization you used at train time.
    """
    def __init__(self, df: pd.DataFrame, feature_column: str,
                 label_columns: list, root_dir: str, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_column = feature_column
        self.label_columns = label_columns  # length = 7
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1) load image
        img_name = self.df.loc[idx, self.feature_column]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 2) apply transforms (pixel-wise normalization)
        if self.transforms is not None:
            image = self.transforms(image)

        # 3) gather the 7 raw labels into a torch.float32 tensor
        labels_np = self.df.loc[idx, self.label_columns].values.astype(np.float32)
        labels_tensor = torch.from_numpy(labels_np)  # shape = (7,)
        return image, labels_tensor


# 3) Load a saved ResNet50→7 model
def load_multi7_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model


# 4) Compute mean/std over the test‐set images (unchanged from training)
def compute_dataset_stats(image_dir):
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    rgb_accumulator = []
    for path in image_files:
        arr = np.array(Image.open(path), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        non_black = flat[~np.all(flat == 0, axis=1)]
        if non_black.size > 0:
            rgb_accumulator.append(non_black)
    rgb_all = np.concatenate(rgb_accumulator, axis=0)
    return rgb_all.mean(axis=0).tolist(), rgb_all.std(axis=0).tolist()


# 5) Compute per‐variable MSE, RMSE, Huber
def evaluate_metrics(preds, targets, delta=0.05):
    """
    preds, targets: 1D numpy arrays of length N for a single variable.
    Returns (mse, rmse, huber_mean).
    """
    preds = np.array(preds)
    targets = np.array(targets)
    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    err = preds - targets
    small = np.abs(err) <= delta
    huber_elem = np.where(
        small,
        0.5 * err ** 2,
        delta * (np.abs(err) - 0.5 * delta)
    )
    return mse, rmse, float(np.mean(huber_elem))


# 6) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 7) Specify your (model, test_csv, test_image_dir) triples
test_configurations = [
    # Example: replace these paths with your actual saved .pt and test CSV/image folders
    (
        "LEAF/human-leaf-top-and-angled-multi7-huber-lr1e7-delta5e2.pt",
        "LEAF/human/test/test_resnet-leaf-top-and-angled.csv",
        "LEAF/human/test/images"
    ),
    # Add more tuples if you want to evaluate multiple models
]


results = {}

for model_path, test_labels_file, test_dir in test_configurations:
    # --- a) Compute image normalization for the test set ---
    mu_rgb, std_rgb = compute_dataset_stats(test_dir)
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mu_rgb, std=std_rgb)
    ])

    # --- b) Read test CSV (must contain the seven raw columns + "image_file") ---
    df = pd.read_csv(test_labels_file)

    # --- c) Build DataLoader that returns (image, [gsw, gtw, VPleaf, VPDleaf, H2O_leaf, Fs, Fm']) ---
    target_cols = ["gsw", "gtw", "VPleaf", "VPDleaf", "H2O_leaf", "Fs", "Fm'"]
    ds = MultiTargetImageDataset(
        df,
        feature_column="image_file",
        label_columns=target_cols,
        root_dir=test_dir,
        transforms=test_transforms
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # --- d) Load the model and move to .half() if you used fp16 at training (optional) ---
    model = load_multi7_model(model_path).to(device).half()
    model.eval()

    # --- e) Create a dummy Learner just so DeepML's TF logging doesn’t blow up. We won't call it. ---
    regression = MultiTargetRegression(model, "resnet50_multi7")
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion  = RMSELoss()
    learner    = Learner(regression, optimizer, criterion)

    # --- f) Run inference and collect predictions + targets in two lists of length N ---
    all_preds = []   # each element will be a list of length 7
    all_targs = []   # each element will be a list of length 7

    with torch.inference_mode():
        for inputs, targets in loader:
            # inputs: (1, 3, H, W); targets: (1, 7) raw floats
            inputs = inputs.half().to(device)
            targets = targets.to(device)  # shape (1,7), raw

            outputs = model(inputs)       # shape = (1,7)
            preds = outputs.squeeze(0).cpu().float().numpy() # → 1D array length 7
            targs = targets.squeeze(0).cpu().float().numpy()

            all_preds.append(preds.tolist())
            all_targs.append(targs.tolist())

    # --- g) Convert to numpy arrays of shape (N, 7) ---
    all_preds_np = np.array(all_preds, dtype=np.float32)  # shape = (N,7)
    all_targs_np = np.array(all_targs, dtype=np.float32)  # shape = (N,7)

    # --- h) Compute metrics for each of the 7 variables individually ---
    var_names = ["gsw", "gtw", "VPleaf", "VPDleaf", "H2O_leaf", "Fs", "Fm'"]
    per_var_results = {}

    for i, var in enumerate(var_names):
        preds_i = all_preds_np[:, i]
        targs_i = all_targs_np[:, i]
        mse_i, rmse_i, huber_i = evaluate_metrics(preds_i, targs_i, delta=0.05)
        per_var_results[var] = {
            "mse":   mse_i,
            "rmse":  rmse_i,
            "huber": huber_i
        }

    model_name = os.path.basename(model_path)
    results[model_name] = per_var_results

    print(f"\nResults for {model_name}:")
    for var, m in per_var_results.items():
        print(f"  {var:8s} → MSE: {m['mse']:.4f},  RMSE: {m['rmse']:.4f},  Huber: {m['huber']:.4f}")

    # --- i) Cleanup GPU memory ---
    del model, learner, regression, optimizer, criterion
    torch.cuda.empty_cache()

# --- j) Write everything to JSON ---
with open("test_results_per_variable.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll model‐vs‐variable results saved to test_results_per_variable.json")
