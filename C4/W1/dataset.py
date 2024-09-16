from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, context, target, transform=None):
        super().__init__()
        self.context = transform(context)
        self.target = transform(target)

    def __len__(self):
        return len(self.target)
    
    def getitem(self, idx):
        return self.context[idx], self.target[idx]