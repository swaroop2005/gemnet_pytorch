"""
Minimal OC20 LMDB → PyG Dataset
--------------------------------
• Reads OC20 IS2RE (or S2EF) LMDB files.
• Pass-through: uses the pre-computed edge_index, edge_attr, cell_offsets
  already stored in each LMDB entry.
• Returns a torch_geometric.data.Data object with keys used by GemNet.
"""

import lmdb
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class OC20LmdbDataset(Dataset):
    """
    Args
    ----
    lmdb_path : str
        Path to `data.lmdb` file (train or val).
    target_key : str
        Which energy label to use.  'y_relaxed' for IS2RE
        or 'y' for generic S2EF frame.
    """

    def __init__(self, lmdb_path: str, target_key: str = "y_relaxed"):
        self.lmdb_path = lmdb_path
        self.target_key = target_key

        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin() as txn:
            self.keys = [k for k, _ in txn.cursor()]

    # --------------------------------------------------------------------- #
    def __len__(self):
        return len(self.keys)

    # --------------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        with self.env.begin(write=False) as txn:
            sample = pickle.loads(txn.get(self.keys[idx]))

        z = sample.atomic_numbers.long()
        pos = sample.pos
        cell = sample.cell.squeeze()
        edge_index = sample.edge_index
        cell_offsets = sample.cell_offsets

        # Use distances as edge_attr if edge_attr is None
        edge_attr = sample.edge_attr
        if edge_attr is None and hasattr(sample, "distances") and sample.distances is not None:
            edge_attr = sample.distances.unsqueeze(-1)
        elif edge_attr is None:
            raise ValueError("No edge_attr or distances found in sample!")

        y = torch.tensor([sample.y_relaxed], dtype=torch.float32)

        data = Data(
            z=z,
            pos=pos,
            cell=cell,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cell_offsets=cell_offsets,
            y=y,
            natoms=torch.tensor([pos.size(0)], dtype=torch.long),
        )
        return data
