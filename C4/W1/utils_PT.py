import torch
import pathlib
import numpy as np
from dataset import CustomDataset
from torch.utils.data import DataLoader



path_to_file = pathlib.Path("por-eng/por.txt")

np.random.seed(2024)
torch.manual_seed(2024)


def load_data(path):
    text = path.read_text(encoding="utf-8")
  
    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    context = np.array([context for _, context, _ in pairs])
    target = np.array([target for target, _, _ in pairs])

    return context, target


portuguese_sentences, english_sentences = load_data(path_to_file)
sentences = (portuguese_sentences, english_sentences)

BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(portuguese_sentences),)) < 0.8

train_dataset = CustomDataset(english_sentences[is_train], portuguese_sentences[is_train])
val_dataset = CustomDataset(english_sentences[~is_train], portuguese_sentences[~is_train])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)