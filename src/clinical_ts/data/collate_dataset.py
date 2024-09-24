import torch.utils.data

class CollateDataset(torch.utils.data.Dataset):
    r"""Dataset for collating several existing datasets
    """

    def __init__(self, datasets, sample_idx_from_first_dataset=False) -> None:
        super().__init__()
        self.datasets = datasets
        self.sample_idx_from_first_dataset = sample_idx_from_first_dataset

    def __getitem__(self, idx):
        if(self.sample_idx_from_first_dataset):
            sample_idx = self.datasets[0][idx][0]
            res = tuple(self.datasets[0][idx][1:])
        else:
            sample_idx = idx
            res = tuple(self.datasets[0][idx])
        
        for d in self.datasets[1:]:
            res += tuple(d[sample_idx])
        return res
            
    def __len__(self):
        return len(self.datasets[0])