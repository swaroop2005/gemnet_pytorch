import torch
from data.oc20_lmdb import OC20LmdbDataset

class DataContainer:
    """
    Wrapper for OC20LmdbDataset for use with LMDB datasets (IS2RE/S2EF).
    
    Parameters:
        path (str): Path to the LMDB file.
        target_key (str): Energy label to use ('y_relaxed' for IS2RE).
        transform (callable, optional): Transformation to apply to each sample.
        device (torch.device or str, optional): Device to move samples to.
    """
    def __init__(self, path, target_key="y_relaxed", transform=None, device=None, **kwargs):
        self.dataset = OC20LmdbDataset(path, target_key=target_key)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.device is not None:
            data = data.to(self.device)
        return data

    @property
    def raw_dataset(self):
        return self.dataset

    def get_targets(self):
        return [d.y for d in self.dataset]
