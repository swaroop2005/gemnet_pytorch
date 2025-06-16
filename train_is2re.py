import argparse
import pathlib
import yaml
import time
import csv
import math
import os
import torch
from torch_geometric.loader import DataLoader

from data.oc20_lmdb import OC20LmdbDataset
from gemnet.model.gemnet import GemNet


# ───────────────────────── helper ─────────────────────────
def build_model(cfg):
    """
    Build GemNet with the minimal surgery required to accept the
    keys you placed in config.yaml.
    """
    model_cfg = cfg["model"]  # your YAML does have a 'model:' block

    # ---- rename legacy keys -------------------------------------------------
    translation = {
        "emb_size_trip_in": "emb_size_trip",
        "emb_size_quad_in": "emb_size_quad",
        "emb_size_trip_out": "emb_size_bil_trip",
        "emb_size_quad_out": "emb_size_bil_quad",
    }
    for old, new in translation.items():
        if old in model_cfg and new not in model_cfg:
            model_cfg[new] = model_cfg.pop(old)

    # ---- mandatory defaults (if omitted) ------------------------------------
    defaults = dict(
        num_before_skip=1,
        num_after_skip=2,
        num_concat=1,
        num_atom=100,  # max atoms per structure
        triplets_only=False,
    )
    for k, v in defaults.items():
        model_cfg.setdefault(k, v)

    # ---- GemNet-accepted kwargs whitelist -----------------------------------
    allowed = {
        "emb_size_atom", "emb_size_edge",
        "emb_size_trip", "emb_size_quad",
        "emb_size_bil_trip", "emb_size_bil_quad",
        "num_before_skip", "num_after_skip",
        "num_concat", "num_atom",
        "cutoff", "max_neighbors",
        "num_layers", "num_blocks",
        "num_spherical", "num_radial",
        "triplets_only", "extensive",
        "activation",
        "emb_size_rbf", "emb_size_cbf", "emb_size_sbf",
        "emb_size_aint_in", "emb_size_aint_out",  
        "scale_file",
    }
    kw = {k: v for k, v in model_cfg.items() if k in allowed}

    # Warn if some keys in model_cfg are ignored
    ignored = set(model_cfg.keys()) - allowed
    if ignored:
        print(f"Warning: Ignored model config keys: {ignored}")

    return GemNet(**kw)


def save_checkpoint(path, epoch, model, optim, scheduler, best_val, ema=None):
    ckpt = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
    }
    if ema is not None:
        ckpt["ema_state"] = ema.state_dict()
    try:
        torch.save(ckpt, path)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint at {path}: {e}")


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[n] = p.detach().clone()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, shadow_dict):
        # Optionally check key match
        if set(shadow_dict.keys()) != set(self.shadow.keys()):
            print("Warning: EMA keys in checkpoint do not match model parameters.")
        self.shadow = shadow_dict


# ─────────────────────────── main ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="runs_is2re")
    ap.add_argument("--resume", default=None, help="checkpoint path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Save a copy of config for reproducibility
    run_dir = pathlib.Path(args.outdir)
    run_dir.mkdir(exist_ok=True)
    config_copy_path = run_dir / "config_used.yaml"
    if not config_copy_path.exists():
        with open(config_copy_path, "w") as f:
            yaml.dump(cfg, f)

    csv_path = run_dir / "log.csv"
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    # ─── Data ─────────────────────────────────────────────────
    train_ds = OC20LmdbDataset(lmdb_path=cfg["dataset"])
    val_ds = OC20LmdbDataset(lmdb_path=cfg["val_dataset"])

    # Use more flexible num_workers and pin_memory
    num_workers = cfg.get("num_workers", max(2, os.cpu_count() // 2 if os.cpu_count() else 2))
    use_cuda = args.device == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    # ─── Model / Optim / Sched ───────────────────────────────
    device = args.device
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    loss_fn_name = cfg.get("loss_energy", "mae")
    if loss_fn_name == "mae":
        loss_fn = torch.nn.L1Loss()
    elif loss_fn_name == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_energy type: {loss_fn_name}")

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    ema = EMA(model, decay=0.999) if cfg.get("use_ema", False) else None
    clip = cfg.get("grad_clip_max", None)

    # ─── Resume ───────────────────────────────────────────────
    start_epoch = 1
    best_val = math.inf
    if args.resume and pathlib.Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        if ema and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
        print(f"✓ resumed from {args.resume} at epoch {start_epoch}")

    # ─── CSV header ───────────────────────────────────────────
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            csv.writer(f).writerow(["epoch", "train_mae", "val_mae", "lr"])

    # ─── Training loop ────────────────────────────────────────
    patience = cfg.get("early_stop_patience", 10)
    bad_epochs = 0

    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        t0 = time.time()
        # ----- train -----
        model.train()
        train_sum = 0
        for batch in train_dl:
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                pred = model(batch).view(-1)
                loss = loss_fn(pred, batch.y.view(-1))
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            train_sum += loss.item()
            if ema:
                ema.update(model)
        train_mae = train_sum / len(train_dl)

        # ----- validate -----
        model_to_eval = model
        if ema:
            # evaluate EMA weights
            shadow = {n: p.clone() for n, p in model.named_parameters()}
            for n, p in model.named_parameters():
                p.data.copy_(ema.shadow[n])
            model_to_eval = model

        model_to_eval.eval()
        val_sum = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device)
                pred = model_to_eval(batch).view(-1)
                val_sum += loss_fn(pred, batch.y.view(-1)).item()
        val_mae = val_sum / len(val_dl)

        if ema:
            # restore weights
            for n, p in model.named_parameters():
                p.data.copy_(shadow[n])

        # ----- LR schedule & early stop -----
        scheduler.step(val_mae)
        lr_now = opt.param_groups[0]["lr"]
        if val_mae < best_val:
            best_val = val_mae
            bad_epochs = 0
            save_checkpoint(best_ckpt, epoch, model, opt, scheduler, best_val, ema)
        else:
            bad_epochs += 1

        save_checkpoint(last_ckpt, epoch, model, opt, scheduler, best_val, ema)

        # ----- CSV log -----
        with open(csv_path, "a") as f:
            csv.writer(f).writerow([epoch, f"{train_mae:.5f}", f"{val_mae:.5f}", f"{lr_now:.6f}"])

        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d} | train {train_mae:.4f} | val {val_mae:.4f} | lr {lr_now:.2e} | {dt/60:.1f} min"
        )

        if bad_epochs >= patience:
            print("Early stopping criterion met.")
            break

    print("Training finished. Best val MAE:", best_val)


if __name__ == "__main__":
    main()
