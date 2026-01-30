import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "fea_realistic_40k.csv"  # put the csv next to this file


def sincos_degrees(angle_deg: np.ndarray) -> np.ndarray:
    """Encode angle as sin/cos to avoid wraparound discontinuity at 360/0."""
    rad = np.deg2rad(angle_deg.astype(np.float32))
    return np.stack([np.sin(rad), np.cos(rad)], axis=1)


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, 2)  # [stress, disp]

    def forward(self, x):
        h = self.net(x)
        return self.head(h)


def main():
    df = pd.read_csv(CSV_PATH)

    # ---- Targets ----
    y = df[["VM_Stress_MPa", "Disp_mm"]].astype(np.float32).to_numpy()

    # Optional: log-transform targets to stabilise scale (often helps)
    # If you do this, remember to expm1() when interpreting predictions.
    use_log_targets = True
    if use_log_targets:
        # log1p handles small values safely
        y = np.log1p(np.clip(y, a_min=0.0, a_max=None))

    # ---- Features ----
    # Numeric
    num_cols = ["E_MPa", "L", "W", "H", "Force_N"]
    X_num = df[num_cols].astype(np.float32).to_numpy()

    # Angle -> sin/cos
    X_ang = sincos_degrees(df["Angle"].to_numpy())

    # Material -> one-hot
    mat = df["Material"].astype(str)
    mats = sorted(mat.unique().tolist())
    mat_to_idx = {m: i for i, m in enumerate(mats)}
    X_mat = np.zeros((len(df), len(mats)), dtype=np.float32)
    X_mat[np.arange(len(df)), mat.map(mat_to_idx).to_numpy()] = 1.0

    # Combine features
    X = np.concatenate([X_num, X_ang, X_mat], axis=1).astype(np.float32)

    # ---- Split ----
    # Keep distribution similar across materials
    strat = mat.to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=strat
    )

    # ---- Scale numeric-ish features (all features) ----
    # For one-hot, scaling isn't critical, but scaling the full vector is fine.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # ---- Torch ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = MLP(in_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()  # robust regression loss

    best_val = float("inf")

    for epoch in range(1, 21):
        # ---- Train ----
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)

        tr_loss /= max(tr_n, 1)
        val_loss /= max(val_n, 1)

        print(f"Epoch {epoch:02d} | train={tr_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "materials": mats,
                "use_log_targets": use_log_targets,
                "num_cols": num_cols,
            }
            torch.save(ckpt, "fea_surrogate_best.pt")

    print("Saved best checkpoint to fea_surrogate_best.pt")
    print("Materials:", mats)


if __name__ == "__main__":
    main()
