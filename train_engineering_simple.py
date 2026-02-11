# train_engineering_simple.py - IMPLIED SCALE SUPERVISION (matches your final metric)
import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.engineering_gnn import EngineeringGNN
    print("✓ Engineering GNN imported")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

from utils.data_loader import load_npz_dataset
from torch_geometric.loader import DataLoader


config = {
    "experiment_name": "engineering_v1_improved",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "train_dir": "dataset/train",
    "val_dir": "dataset/val",
    "batch_size": 2,
    "hidden_dim": 128,
    "num_layers": 3,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,

    "yield_stress": 250e6,
    "min_safety_factor": 1.5,
    "bc_feature_idx": -1,

    "phase1_epochs": 10,
    "phase2_epochs": 10,

    "huber_delta": 1e-3,
    "rel_eps": 1e-6,
    "rel_weight": 0.5,
    "abs_weight": 0.5,

    # very light prior only
    "scale_prior_weight": 1e-4,
    "target_disp_scale_m": 1.0e-2,

    # IMPORTANT: implied scale supervision weights
    "implied_scale_weight_p1": 10.0,
    "implied_scale_weight_p2": 5.0,
    "implied_scale_weight_p3": 1.0,

    "stress_log_mse_weight": 1.0,
    "corr_weight": 0.2,

    "yield_soft_weight": 0.05,
    "sf_soft_weight": 0.05,
    "stress_floor_soft_weight": 0.02,
    "stress_floor_pa": 20e6,
}

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

if config["device"] == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def get_free_mask(batch, bc_feature_idx=None):
    fm = getattr(batch, "free_mask", None)
    if fm is not None:
        if fm.dim() == 1:
            fm = fm.unsqueeze(-1)
        return fm.float().clamp(0.0, 1.0)

    for name in ("fixed_mask", "node_fixed", "is_fixed"):
        m = getattr(batch, name, None)
        if m is not None:
            if m.dim() == 1:
                m = m.unsqueeze(-1)
            return (1.0 - m.float().clamp(0.0, 1.0))

    if bc_feature_idx is not None and hasattr(batch, "x") and batch.x is not None:
        bc = batch.x[:, bc_feature_idx]
        is_fixed = (bc > 0.5)
        return (~is_fixed).float().unsqueeze(-1)

    return None


def displacement_loss(u_pred, u_true, free_mask, huber_delta, rel_eps, rel_weight, abs_weight):
    if free_mask is None:
        mag = u_true.norm(dim=-1, keepdim=True)
        w = (mag / (mag.mean() + 1e-8)).clamp(0.2, 5.0)

        abs_l = F.smooth_l1_loss(u_pred, u_true, beta=huber_delta, reduction="none")
        abs_l = (abs_l * w).mean()

        rel = (u_pred - u_true).norm(dim=-1) / (u_true.norm(dim=-1) + rel_eps)
        rel_l = (rel * w.squeeze(-1)).mean()

        return abs_weight * abs_l + rel_weight * rel_l

    abs_l = F.smooth_l1_loss(u_pred, u_true, beta=huber_delta, reduction="none")
    abs_l = (abs_l * free_mask).sum() / (free_mask.sum() * u_pred.shape[-1] + 1e-8)

    rel = (u_pred - u_true).norm(dim=-1, keepdim=True) / (u_true.norm(dim=-1, keepdim=True) + rel_eps)
    rel_l = (rel * free_mask).sum() / (free_mask.sum() + 1e-8)

    return abs_weight * abs_l + rel_weight * rel_l


def corr_loss(a, b, mask=None):
    if a.dim() > 1:
        a = a.squeeze(-1)
    if b.dim() > 1:
        b = b.squeeze(-1)

    if mask is not None:
        m = mask.squeeze(-1)
        a = a[m > 0.5]
        b = b[m > 0.5]
        if a.numel() < 4:
            return torch.tensor(0.0, device=a.device)

    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return 1.0 - (a * b).mean()


def has_tensor_attr(obj, name: str) -> bool:
    v = getattr(obj, name, None)
    return v is not None and torch.is_tensor(v)


def get_disp_scale_scalar(out: dict) -> torch.Tensor:
    if "disp_scale" in out and torch.is_tensor(out["disp_scale"]):
        return out["disp_scale"].reshape(())
    if "disp_scale_graph" in out and torch.is_tensor(out["disp_scale_graph"]):
        return out["disp_scale_graph"].mean().reshape(())
    return torch.tensor(float("nan"))


def graph_max_abs_component(u: torch.Tensor, batch_idx: torch.Tensor, free_mask):
    """
    Per-graph max of abs(component) across nodes and xyz.
    This matches: u.abs().max()
    Returns [B,1]
    """
    if free_mask is not None:
        m = free_mask.squeeze(-1) > 0.5
        u = u[m]
        b = batch_idx[m]
    else:
        b = batch_idx

    if u.numel() == 0:
        B = int(batch_idx.max().item() + 1)
        return torch.zeros((B, 1), device=batch_idx.device)

    umax_per_node = u.abs().max(dim=-1).values  # [Nfree]
    B = int(batch_idx.max().item() + 1)
    out = torch.zeros((B,), device=u.device)

    if hasattr(out, "scatter_reduce"):
        out = out.scatter_reduce(0, b, umax_per_node, reduce="amax", include_self=True)
    else:
        for bi in range(B):
            mask = (b == bi)
            out[bi] = umax_per_node[mask].max() if mask.any() else 0.0

    return out.unsqueeze(-1)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = Path(f"engineering_experiments/{config['experiment_name']}_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment: {exp_dir}")
print(f"Device: {config['device']}")

print("\nLoading data...")
train_data = load_npz_dataset(config["train_dir"], max_samples=80)
val_data = load_npz_dataset(config["val_dir"], max_samples=20)

print(f"Train: {len(train_data)} samples")
print(f"Val: {len(val_data)} samples")
if len(train_data) == 0:
    print("No training data!")
    sys.exit(1)

loader_kwargs = {"num_workers": 0, "pin_memory": (config["device"] == "cuda")}
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, **loader_kwargs)

sample = train_data[0]
model = EngineeringGNN(
    node_dim=sample.x.shape[1],
    edge_dim=sample.edge_attr.shape[1],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    init_disp_scale=1.0e-2,
    min_disp_scale=1e-5,
)
model.yield_stress = config["yield_stress"]

device = torch.device(config["device"])
model = model.to(device)

# boost base scale LR
base_scale_param_name = "log_base_disp_scale"
base_scale_params, other_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (base_scale_params if base_scale_param_name in n else other_params).append(p)

param_groups = [{"params": other_params, "lr": config["learning_rate"]}]
if base_scale_params:
    param_groups.append({"params": base_scale_params, "lr": config["learning_rate"] * 2.0})

optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Starting training for {config['epochs']} epochs...")

best_val = float("inf")

for epoch in range(1, config["epochs"] + 1):
    phase1_end = config["phase1_epochs"]
    phase2_end = config["phase1_epochs"] + config["phase2_epochs"]
    phase = 1 if epoch <= phase1_end else (2 if epoch <= phase2_end else 3)

    if phase == 1:
        w_implied = config["implied_scale_weight_p1"]
    elif phase == 2:
        w_implied = config["implied_scale_weight_p2"]
    else:
        w_implied = config["implied_scale_weight_p3"]

    model.eval()
    with torch.no_grad():
        base_scale = float(model.get_base_disp_scale().item()) if hasattr(model, "get_base_disp_scale") else None

    print(f"\nEpoch {epoch}/{config['epochs']}  (Phase {phase})")
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f} | base_disp_scale: {base_scale if base_scale is not None else 'n/a'} m")

    model.train()
    train_losses = []

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        out = model(batch)

        u_pred = out["displacement"]
        raw_u = out["raw_displacement"]
        u_true = batch.u_true
        free_mask = get_free_mask(batch, bc_feature_idx=config["bc_feature_idx"])

        # node loss
        loss_u = displacement_loss(
            u_pred, u_true, free_mask,
            huber_delta=config["huber_delta"],
            rel_eps=config["rel_eps"],
            rel_weight=config["rel_weight"],
            abs_weight=config["abs_weight"],
        )
        total = loss_u

        # per-graph implied scale supervision
        bidx = getattr(batch, "batch", None)
        if bidx is None:
            bidx = torch.zeros(u_true.size(0), dtype=torch.long, device=device)

        gt_max = graph_max_abs_component(u_true, bidx, free_mask)  # [B,1]
        raw_max = graph_max_abs_component(raw_u, bidx, free_mask)  # [B,1]
        scale_target = gt_max / (raw_max + 1e-9)

        disp_scale_graph = out["disp_scale_graph"]  # [B,1]

        loss_implied = F.mse_loss(torch.log(disp_scale_graph + 1e-12), torch.log(scale_target + 1e-12))
        total = total + w_implied * loss_implied

        # tiny prior (optional)
        if phase <= 2 and config["scale_prior_weight"] > 0:
            scale = get_disp_scale_scalar(out).to(device)
            target = torch.tensor(config["target_disp_scale_m"], device=device)
            loss_scale = (torch.log(scale + 1e-12) - torch.log(target + 1e-12)) ** 2
            total = total + config["scale_prior_weight"] * loss_scale

        if phase >= 2:
            if has_tensor_attr(batch, "stress_true"):
                total = total + config["stress_log_mse_weight"] * F.mse_loss(
                    torch.log(out["stress"] + 1.0),
                    torch.log(batch.stress_true + 1.0),
                )

            u_mag = u_pred.norm(dim=-1)
            s_mag = out["stress"].squeeze(-1)
            total = total + config["corr_weight"] * corr_loss(u_mag, s_mag, mask=free_mask)

        if phase >= 3:
            s = out["stress"]
            sf = out["safety_factor"]

            stress_ratio = s / config["yield_stress"]
            yield_pen = torch.relu(stress_ratio - 0.95).pow(2).mean()
            sf_pen = torch.relu(config["min_safety_factor"] - sf).pow(2).mean()
            floor_pen = torch.relu(config["stress_floor_pa"] - s).mean() / config["stress_floor_pa"]

            total = (
                total
                + config["yield_soft_weight"] * yield_pen
                + config["sf_soft_weight"] * sf_pen
                + config["stress_floor_soft_weight"] * floor_pen
            )

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_losses.append(total.item())

        if batch_idx == 0:
            pbar.set_postfix({
                "loss": float(total.item()),
                "Lu": float(loss_u.item()),
                "Limplied": float(loss_implied.item()),
                "scale": float(out["disp_scale"].item()),
            })
        else:
            pbar.set_postfix({"loss": float(total.item())})

    avg_train = float(np.mean(train_losses))

    model.eval()
    val_losses = []
    stats = {"mean_rel_disp": [], "pred_max_disp_mm": [], "pred_avg_stress_mpa": [], "min_sf": [], "max_stress_ratio": []}

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            free_mask = get_free_mask(batch, bc_feature_idx=config["bc_feature_idx"])

            loss_u_val = displacement_loss(
                out["displacement"], batch.u_true, free_mask,
                huber_delta=config["huber_delta"],
                rel_eps=config["rel_eps"],
                rel_weight=config["rel_weight"],
                abs_weight=config["abs_weight"],
            )
            val_losses.append(loss_u_val.item())

            # MATCH YOUR FINAL METRIC: max abs component over free nodes
            bidx = getattr(batch, "batch", None)
            if bidx is None:
                bidx = torch.zeros(batch.u_true.size(0), dtype=torch.long, device=device)

            gt_max = graph_max_abs_component(batch.u_true, bidx, free_mask).max().item() + 1e-12
            pr_max = graph_max_abs_component(out["displacement"], bidx, free_mask).max().item()
            rel_err = abs(pr_max - gt_max) / gt_max

            stats["mean_rel_disp"].append(rel_err)
            stats["pred_max_disp_mm"].append(pr_max * 1000.0)
            stats["pred_avg_stress_mpa"].append(out["stress"].mean().item() / 1e6)
            stats["min_sf"].append(out["safety_factor"].min().item())
            stats["max_stress_ratio"].append(out["stress"].max().item() / config["yield_stress"])

    avg_val = float(np.mean(val_losses))
    scheduler.step(avg_val)

    print(f"Train Loss: {avg_train:.6f}")
    print(f"Val Loss:   {avg_val:.6f}")
    print(f"Mean Rel Max Disp Error: {np.mean(stats['mean_rel_disp']) * 100:.1f}%")
    print(f"Pred Max Disp (avg):     {np.mean(stats['pred_max_disp_mm']):.2f} mm")
    print(f"Avg Stress (avg):        {np.mean(stats['pred_avg_stress_mpa']):.1f} MPa")
    print(f"Min Safety Factor (avg): {np.mean(stats['min_sf']):.2f}")
    print(f"Max Stress/Yield (avg):  {np.mean(stats['max_stress_ratio']):.3f}")

    if epoch % 5 == 0 or epoch == config["epochs"]:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
                "config": config,
            },
            exp_dir / f"model_epoch_{epoch}.pt",
        )
        print("  ✓ Checkpoint saved")

    if avg_val < best_val:
        best_val = avg_val
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
                "config": config,
            },
            exp_dir / "model_best.pt",
        )
        print(f"  ✓ New best model saved (val loss: {best_val:.6f})")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print(f"Models saved in: {exp_dir}")
print("=" * 80)

print("\n" + "=" * 80)
print("FINAL ENGINEERING VALIDATION")
print("=" * 80)

model.eval()
with torch.no_grad():
    test_sample = val_data[0].to(device)
    out = model(test_sample)

    free_mask = getattr(test_sample, "free_mask", None)
    if free_mask is not None:
        m = free_mask.squeeze(-1) > 0.5
        gt_disp = test_sample.u_true[m]
        pred_disp = out["displacement"][m]
    else:
        gt_disp = test_sample.u_true
        pred_disp = out["displacement"]

    gt_max_disp_mm = gt_disp.abs().max().item() * 1000.0
    pred_max_disp_mm = pred_disp.abs().max().item() * 1000.0

    abs_err_mm = abs(pred_max_disp_mm - gt_max_disp_mm)
    rel_err_pct = abs_err_mm / (gt_max_disp_mm + 1e-12) * 100.0

    pred_stress = out["stress"]
    pred_max_stress_mpa = pred_stress.max().item() / 1e6
    pred_min_stress_mpa = pred_stress.min().item() / 1e6
    pred_avg_stress_mpa = pred_stress.mean().item() / 1e6

    has_gt_stress = has_tensor_attr(test_sample, "stress_true")
    if has_gt_stress:
        gt_stress = test_sample.stress_true
        gt_max_stress_mpa = gt_stress.max().item() / 1e6
        gt_avg_stress_mpa = gt_stress.mean().item() / 1e6

    max_stress_ratio = pred_stress.max().item() / config["yield_stress"]
    min_safety_factor = out["safety_factor"].min().item()

    print("\n📊 DISPLACEMENT")
    print(f"   Ground Truth Max:  {gt_max_disp_mm:8.2f} mm")
    print(f"   Predicted Max:     {pred_max_disp_mm:8.2f} mm")
    print(f"   Absolute Error:    {abs_err_mm:8.2f} mm")
    print(f"   Relative Error:    {rel_err_pct:7.2f} %")

    print("\n⚡ STRESS")
    print(f"   Predicted Max:     {pred_max_stress_mpa:8.1f} MPa")
    print(f"   Predicted Min:     {pred_min_stress_mpa:8.1f} MPa")
    print(f"   Predicted Avg:     {pred_avg_stress_mpa:8.1f} MPa")
    if has_gt_stress:
        print(f"   Ground Truth Max:  {gt_max_stress_mpa:8.1f} MPa")
        print(f"   Ground Truth Avg:  {gt_avg_stress_mpa:8.1f} MPa")

    print("\n🛡️ SAFETY")
    print(f"   Yield Stress:      {config['yield_stress'] / 1e6:8.1f} MPa")
    print(f"   Max Stress/Yield:  {max_stress_ratio:8.3f}")
    print(f"   Min Safety Factor: {min_safety_factor:8.2f}")

    print("\n" + "-" * 60)
    print("ENGINEERING ASSESSMENT")
    print("-" * 60)

    passes = True
    issues = []

    if rel_err_pct > 15.0:
        passes = False
        issues.append(f"• Displacement error {rel_err_pct:.1f}% > 15%")
    else:
        print(f"✅ Displacement accuracy OK ({rel_err_pct:.1f}%)")

    if max_stress_ratio > 0.95:
        passes = False
        issues.append(f"• Stress ratio {max_stress_ratio:.3f} > 0.95")
    else:
        print(f"✅ Stress below yield limit ({max_stress_ratio:.3f})")

    if min_safety_factor < 1.3:
        passes = False
        issues.append(f"• Safety factor {min_safety_factor:.2f} < 1.3")
    else:
        print(f"✅ Safety factor OK ({min_safety_factor:.2f})")

    if pred_min_stress_mpa < 1.0:
        issues.append(f"⚠️  Very low minimum stress ({pred_min_stress_mpa:.1f} MPa)")
    else:
        print(f"✅ Minimum stress reasonable ({pred_min_stress_mpa:.1f} MPa)")

    if pred_max_stress_mpa > 500.0:
        issues.append(f"⚠️  Very high maximum stress ({pred_max_stress_mpa:.1f} MPa)")
    else:
        print(f"✅ Maximum stress reasonable ({pred_max_stress_mpa:.1f} MPa)")

    print("\n" + "-" * 60)
    if passes:
        print("🎉 SUCCESS: Model meets engineering requirements")
    else:
        print("⚠️  ISSUES DETECTED:")
        for issue in issues:
            print("   " + issue)

print("\n" + "=" * 80)
print("MODEL OUTPUTS")
print("=" * 80)
print(f"Best model:  {exp_dir / 'model_best.pt'}")
print(f"Final model: {exp_dir / ('model_epoch_' + str(config['epochs']) + '.pt')}")
print("=" * 80)
