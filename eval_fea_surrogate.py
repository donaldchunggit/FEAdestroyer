import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "fea_realistic_40k.csv"

def sincos_degrees(angle_deg):
    rad = np.deg2rad(angle_deg.astype(np.float32))
    return np.stack([np.sin(rad), np.cos(rad)], axis=1)

class MLP(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(128, 2)

    def forward(self, x):
        return self.head(self.net(x))

def main():
    ckpt = torch.load("fea_surrogate_best.pt", map_location="cpu")
    mats = ckpt["materials"]
    num_cols = ckpt["num_cols"]
    use_log = ckpt["use_log_targets"]

    df = pd.read_csv(CSV_PATH)

    y = df[["VM_Stress_MPa", "Disp_mm"]].astype(np.float32).to_numpy()
    if use_log:
        y = np.log1p(np.clip(y, 0.0, None))

    X_num = df[num_cols].astype(np.float32).to_numpy()
    X_ang = sincos_degrees(df["Angle"].to_numpy())
    mat = df["Material"].astype(str)
    mat_to_idx = {m:i for i,m in enumerate(mats)}
    X_mat = np.zeros((len(df), len(mats)), dtype=np.float32)
    X_mat[np.arange(len(df)), mat.map(mat_to_idx).to_numpy()] = 1.0

    X = np.concatenate([X_num, X_ang, X_mat], axis=1).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=mat.to_numpy()
    )

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    X_val = scaler.transform(X_val).astype(np.float32)

    model = MLP(X_val.shape[1])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(X_val)).numpy()

    # Convert back to MPa/mm if log targets were used
    if use_log:
        pred_lin = np.expm1(pred)
        y_lin = np.expm1(y_val)
    else:
        pred_lin = pred
        y_lin = y_val

    err = pred_lin - y_lin
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))

    # MAPE can blow up if targets near 0; use eps
    eps = 1e-6
    mape = np.mean(np.abs(err) / np.maximum(np.abs(y_lin), eps), axis=0) * 100.0

    print("Validation metrics (original units):")
    print(f"  Stress MAE: {mae[0]:.3f} MPa | RMSE: {rmse[0]:.3f} MPa | MAPE: {mape[0]:.2f}%")
    print(f"  Disp   MAE: {mae[1]:.3f} mm  | RMSE: {rmse[1]:.3f} mm  | MAPE: {mape[1]:.2f}%")

if __name__ == "__main__":
    main()
