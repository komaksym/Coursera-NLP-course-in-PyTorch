from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, context, target, transform=None):
        super().__init__()
        self.context, self.target = transform(context), transform(target)
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]