# data/oc20_lmdb.py  – IS2RE-only, zero-scan version
import lmdb, pickle, torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

class OC20LmdbDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, cutoff=6.0, max_nbh=50, transform=None):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=32,
        )
        with self.env.begin() as txn:
            # OC20 LMDB always stores dataset length under the key b"length"
            self.length = pickle.loads(txn.get(b"length"))
        self.cutoff = cutoff
        self.max_nbh = max_nbh
        self.transform = transform            # optional extra transforms

    def __len__(self):
        return self.length

    def _fetch(self, idx):
        """Return one raw OC20 sample as a Python dict."""
        key = str(idx).encode()               # integer keys: b"0", b"1", …
        with self.env.begin() as txn:
            return pickle.loads(txn.get(key))

    def __getitem__(self, idx):
        sample = self._fetch(idx)

        # ---- build PyG Data object -----------------------------------------
        z   = torch.tensor(sample["atomic_numbers"], dtype=torch.long)
        pos = torch.tensor(sample["coords"],         dtype=torch.float32)
        cell= torch.tensor(sample["cell"],           dtype=torch.float32)

        # neighbour list on-the-fly (fast ― <2 ms per structure on A100)
        edge_index = radius_graph(
            pos, r=self.cutoff, loop=False,
            max_num_neighbors=self.max_nbh
        )
        row, col   = edge_index
        vec        = pos[col] - pos[row]
        data = Data(
            z=z, pos=pos, cell=cell,
            edge_index=edge_index,
            edge_vec=vec,
            edge_dist=vec.norm(dim=-1, keepdim=True),
            y=torch.tensor(sample["y_relaxed"], dtype=torch.float32).unsqueeze(0),
            sid=torch.tensor([idx])           # keep original UID if you wish
        )

        if self.transform:
            data = self.transform(data)

        return data
