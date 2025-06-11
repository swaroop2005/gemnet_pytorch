"""
Minimal GemNet-OC IS2RE trainer for swaroop2005/gemnet_pytorch
--------------------------------------------------------------
• reads flat YAML (config.yaml)
• uses OC20LmdbDataset (gemnet_pytorch/data/oc20_lmdb.py)
• trains energy-only MAE
• validates every epoch and saves best checkpoint
"""

import yaml, argparse, math, os, time, torch
from torch_geometric.loader import DataLoader

from gemnet.gemnet_oc import GemNetOC          # adjust if your model file name differs
from gemnet_pytorch.data.oc20_lmdb import OC20LmdbDataset


# ────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ────────────────────────────────
def build_model(cfg):
    kw = {k: cfg[k] for k in cfg
          if k.startswith("emb_size_")
          or k in ("num_spherical","num_radial","num_blocks",
                    "cutoff","max_neighbors","extensive","activation")}
    return GemNetOC(**kw)


# ────────────────────────────────
def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    # 1. datasets
    train_ds = OC20LmdbDataset(cfg["dataset"])
    val_ds   = OC20LmdbDataset(cfg["val_dataset"])
    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    val_dl   = DataLoader(
        val_ds, batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=2, pin_memory=True
    )

    # 2. model + optim
    device = args.device
    model  = build_model(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    loss_fn = torch.nn.L1Loss() if cfg["loss_energy"].lower() == "mae" else torch.nn.MSELoss()
    best_val = math.inf
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg["max_epochs"] + 1):
        # ---------- train ----------
        model.train(); running = 0.0
        for batch in train_dl:
            batch = batch.to(device)
            pred  = model(batch).view(-1)      # (B,)
            y_true = batch.y.view(-1)
            loss  = loss_fn(pred, y_true)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

        train_mae = running / len(train_dl)
        # ---------- validate ----------
        model.eval(); val_running = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device)
                pred  = model(batch).view(-1)
                val_running += loss_fn(pred, batch.y.view(-1)).item()

        val_mae = val_running / len(val_dl)
        print(f"Epoch {epoch:3d}  train_MAE {train_mae:.4f}  val_MAE {val_mae:.4f}")

        # ---------- checkpoint ----------
        if val_mae < best_val:
            best_val = val_mae
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_mae": val_mae,
                "config": cfg,
            }
            torch.save(ckpt, f"checkpoints/best_gemnet_oc.pt")
            print(f"  ✓ saved new best checkpoint  (MAE {best_val:.4f})")

    print("Done. Best val MAE:", best_val)


# ────────────────────────────────
if __name__ == "__main__":
    main()
