from torch.utils.data import Dataset

BATCH_SIZE = 64


class CustomDataset(Dataset):
    def __init__(self, context, target, transform=None):
        super().__init__()
        (self.context, self.target_in), self.target_out = transform(context, target)
        
    def __len__(self):
        return len(self.target_out)
    
    def __getitem__(self, idx):
        return (self.context[idx], self.target_in[idx]), self.target_out[idx]