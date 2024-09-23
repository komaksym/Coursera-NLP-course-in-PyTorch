import re
import torch
import torch.nn as nn
import pathlib
import unicodedata
import numpy as np
from dataset import CustomDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


np.random.seed(2024)
torch.manual_seed(2024)

path_to_file = pathlib.Path("por.txt")


def load_data(path):
    """
    Uploading Data from the .txt file
    """
    text = path.read_text(encoding="utf-8")
  
    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    context = [context for context, _, _ in pairs]
    target = [target for _, target, _ in pairs]

    return context, target


# Extracting source and target data from the dataset
english_sentences, portuguese_sentences = load_data(path_to_file)

sentences = (english_sentences, portuguese_sentences)

# Splitting into train/val sets
X_train, X_val, y_train, y_val = train_test_split(english_sentences, portuguese_sentences, test_size=0.2)

def pt_lower_and_split_punct(text):
    """
    Preprocessing text (unicode normalizing, punctuation removal,
    [SOS], [EOS] token joining)
    """
    results = []
    
    # Wrapping a single string in a list
    if isinstance(text, str):
        text = [text]
    
    # Normalizing the text
    for sequence in text:
        sequence = unicodedata.normalize('NFKD', sequence)
        sequence = sequence.lower()
        sequence = re.sub("[^ a-z.?!,¿]", "", sequence)
        sequence = re.sub("[.?!,¿]", r" \g<0> ", sequence)
        sequence = sequence.strip()
        sequence = ' '.join(['[SOS]', sequence, '[EOS]'])
        results.append(sequence)

    return results

# Applying text normalization before training tokenizers on it
english_sentences, portuguese_sentences = pt_lower_and_split_punct(english_sentences), pt_lower_and_split_punct(portuguese_sentences)


def batch_iterator(text, batch_size=1000):
    """
    Iterating over text to then train a tokenizer on it
    """
    for i in range(0, len(text), batch_size):
        yield text[i: i + batch_size]


# Creating tokenizers
tokenizer_eng = Tokenizer(models.WordLevel(unk_token='[UNK]'))
tokenizer_eng.enable_padding()
tokenizer_eng.normalizer = normalizers.NFKD()
tokenizer_eng.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
trainer_eng = trainers.WordLevelTrainer(vocab_size=12000,
                                    special_tokens=['[PAD]', '[UNK]'])

tokenizer_por = Tokenizer(models.WordLevel(unk_token='[UNK]'))
tokenizer_por.enable_padding()
tokenizer_por.normalizer = normalizers.NFKD()
tokenizer_por.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
trainer_por = trainers.WordLevelTrainer(vocab_size=12000, 
                                    special_tokens=['[PAD]', '[UNK]'])

# Training tokenizers
tokenizer_eng.train_from_iterator(batch_iterator(english_sentences), trainer_eng)
tokenizer_por.train_from_iterator(batch_iterator(portuguese_sentences), trainer_por)


def process_text(context, target):
    """
    Wrapping all text processing into a single function
    """
    context = pt_lower_and_split_punct(context)
    context = tokenizer_eng.encode_batch(context)
    context = np.array([seq.ids for seq in context])

    target = pt_lower_and_split_punct(target)
    target = tokenizer_por.encode_batch(target)
    target = np.array([seq.ids for seq in target])

    target_in = []
    target_out = target[:, 1:]

    # Handling slicing up to the last non pad token
    for sub in target:
        idx = np.argmax(sub == 0)
    
        if idx != 0:
            sub = np.delete(sub, idx-1)
        else:
            sub = np.delete(sub, len(sub)-1)
        
        target_in.append(sub)

    return (context, target_in), target_out


BATCH_SIZE = 64

# Building datasets
train_dataset = CustomDataset(X_train, y_train, process_text)
val_dataset = CustomDataset(X_val, y_val, process_text)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


del english_sentences, portuguese_sentences, X_train, X_val, y_train, y_val


def masked_loss(y_true, y_pred):
    # Transposing seq_length and embedding dim to fit the Cross Entropy requirements
    y_pred = torch.transpose(y_pred, 1, 2)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(y_pred, y_true)

    # Check which elements of y_true are padding
    mask = y_true != 0
    loss *= mask

    return torch.sum(loss) / torch.sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = y_pred == y_true
    mask = y_true != 0
    acc *= mask

    return torch.sum(acc) / torch.sum(mask)


def ids_to_text(tokens, decoder):
    words = decoder.decode(tokens)
    
    return words


def encode_sample(sample):
    text = pt_lower_and_split_punct(sample)
    encoded_text = tokenizer_eng.encode(*text)

    return encoded_text.ids
