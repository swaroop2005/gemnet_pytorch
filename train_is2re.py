import argparse, pathlib, yaml, time, csv, math, os
import torch
from torch_geometric.loader import DataLoader

from gemnet_pytorch.data.oc20_lmdb import OC20LmdbDataset
from gemnet.model.gemnet import GemNet                      


# ─────────────────────────── helper ────────────────────────────
def build_model(cfg):
    kw = {k: cfg[k] for k in cfg
          if k.startswith("emb_size_")
          or k in ("num_spherical","num_radial","num_blocks",
                    "cutoff","max_neighbors","extensive","activation")}
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
    torch.save(ckpt, path)


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def state_dict(self):
        return self.shadow


# ─────────────────────────── main ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="runs_is2re")
    ap.add_argument("--resume", default=None, help="checkpoint path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # ─── Dirs ──────────────────────────────────────────────────
    run_dir = pathlib.Path(args.outdir); run_dir.mkdir(exist_ok=True)
    csv_path = run_dir / "log.csv"
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    # ─── Data ─────────────────────────────────────────────────
    train_ds = OC20LmdbDataset(cfg["dataset"])
    val_ds   = OC20LmdbDataset(cfg["val_dataset"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=cfg["eval_batch_size"], shuffle=False,
                          num_workers=2, pin_memory=True)

    # ─── Model / Optim / Sched ───────────────────────────────
    device = args.device
    model  = build_model(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    loss_fn = torch.nn.L1Loss()
    scaler  = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    ema     = EMA(model, decay=0.999) if cfg.get("use_ema", False) else None
    clip    = cfg.get("grad_clip_max", None)

    # ─── Resume ───────────────────────────────────────────────
    start_epoch = 1; best_val = math.inf
    if args.resume and pathlib.Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val   = ckpt["best_val"]
        start_epoch= ckpt["epoch"] + 1
        if ema and "ema_state" in ckpt: ema.shadow = ckpt["ema_state"]
        print(f"✓ resumed from {args.resume} at epoch {start_epoch}")

    # ─── CSV header ───────────────────────────────────────────
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            csv.writer(f).writerow(["epoch","train_mae","val_mae","lr"])

    # ─── Training loop ────────────────────────────────────────
    patience = cfg.get("early_stop_patience", 10)
    bad_epochs = 0

    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        t0 = time.time()
        # ----- train -----
        model.train(); train_sum=0
        for batch in train_dl:
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                pred = model(batch).view(-1)
                loss = loss_fn(pred, batch.y.view(-1))
            scaler.scale(loss).backward()
            if clip: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            train_sum += loss.item()
            if ema: ema.update(model)
        train_mae = train_sum/len(train_dl)

        # ----- validate -----
        model_to_eval = model
        if ema:                                     # evaluate EMA weights
            shadow = {n: p.clone() for n,p in model.named_parameters()}
            for n,p in model.named_parameters():
                p.data.copy_(ema.shadow[n])
            model_to_eval = model

        model_to_eval.eval(); val_sum=0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device)
                pred = model_to_eval(batch).view(-1)
                val_sum += loss_fn(pred, batch.y.view(-1)).item()
        val_mae = val_sum/len(val_dl)

        if ema:                                     # restore weights
            for n,p in model.named_parameters():
                p.data.copy_(shadow[n])

        # ----- LR schedule & early stop -----
        scheduler.step(val_mae)
        lr_now = opt.param_groups[0]["lr"]
        if val_mae < best_val:
            best_val = val_mae; bad_epochs = 0
            save_checkpoint(best_ckpt, epoch, model, opt, scheduler, best_val, ema)
        else:
            bad_epochs += 1

        save_checkpoint(last_ckpt, epoch, model, opt, scheduler, best_val, ema)

        # ----- CSV log -----
        with open(csv_path, "a") as f:
            csv.writer(f).writerow([epoch, f"{train_mae:.5f}", f"{val_mae:.5f}", f"{lr_now:.6f}"])

        dt = time.time() - t0
        print(f"Epoch {epoch:3d} | train {train_mae:.4f} | val {val_mae:.4f} | lr {lr_now:.2e} | {dt/60:.1f} min")

        if bad_epochs >= patience:
            print("Early stopping criterion met.")
            break

    print("Training finished. Best val MAE:", best_val)


if __name__ == "__main__":
    main()
