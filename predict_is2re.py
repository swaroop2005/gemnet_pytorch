# predict_is2re.py  –  rank candidate catalysts by GemNet-OC energy
import argparse, yaml, torch, pathlib
from torch_geometric.loader import DataLoader

from gemnet_pytorch.data.oc20_lmdb import OC20LmdbDataset
from gemnet.model.gemnet import GemNet  

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--config",  default="config.yaml")
p.add_argument("--ckpt",    default="checkpoints/best_gemnet_oc.pt")
p.add_argument("--candidates", required=True,
               help="LMDB folder (data.lmdb) or single file with list of LMDB keys")
p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--topk",    type=int, default=20)
args = p.parse_args()

# ---------- load config + model ----------
cfg = yaml.safe_load(open(args.config))
model_kwargs = {k: cfg[k] for k in cfg if k.startswith("emb_size_")
                or k in ("num_spherical","num_radial","num_blocks",
                         "cutoff","max_neighbors","extensive","activation")}
model = GemNet(**model_kwargs).to(args.device)
state = torch.load(args.ckpt, map_location=args.device)
model.load_state_dict(state["state_dict"])
model.eval()

# ---------- candidate dataset ----------
cand_ds = OC20LmdbDataset(args.candidates)
cand_dl = DataLoader(cand_ds, batch_size=cfg["eval_batch_size"], shuffle=False)

# ---------- predict ----------
results = []      # (energy, lmdb_key)
with torch.no_grad():
    for batch, keys in zip(cand_dl, cand_ds.keys):   # keys list preserved in dataset
        batch = batch.to(args.device)
        pred  = model(batch).view(-1).cpu().tolist()
        results.extend(list(zip(pred, keys)))

# ---------- rank ----------
results.sort()                       # ascending energy
topk = results[:args.topk]

print(f"Top-{args.topk} lowest-energy candidates:")
for i,(e,k) in enumerate(topk, 1):
    print(f"{i:2d} | energy {e: .3f} eV | key {k.decode() if isinstance(k,bytes) else k}")

# Optionally write to CSV
with open("recommendations.csv","w") as f:
    f.write("rank,energy_eV,lmdb_key\n")
    for i,(e,k) in enumerate(topk,1):
        f.write(f"{i},{e},{k.decode() if isinstance(k,bytes) else k}\n")
print("recommendations.csv written")
