from torch.nn.utils.rnn import pad_sequence
from collections import Counter


class SentenceVectorizer:
    """
    Custom word-level text encoder
    """

    #Initializing needed variables
    def __init__(self, pad_token="", unk_token="[UNK]"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.vocab = [pad_token, unk_token]


    def fit(self, sentences):
        #Converting the single string if passed to a list for further processing
        if isinstance(sentences, str):
            sentences = [sentences]

        #Populating the dictionary with our vocabulary
        word_counts = Counter(word for sentence in sentences for word in sentence.split())
        for word, _ in word_counts.items():
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
                self.vocab.append(word)


    def transform(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        #Vectorizing the words by pulling the values from the dictionary, if none is found -> assign the UNK token
        vectorized = [[self.word2idx.get(word, self.word2idx[self.unk_token])
                       for word in sentence.split()]
                      for sentence in sentences]

        #Padding to the biggest sequence received
        return pad_sequence([torch.tensor(sentence) for sentence in vectorized],
                            batch_first=True,
                            padding_value=self.word2idx[self.pad_token])


def get_sentence_vectorizer(sentences):
    torch.manual_seed(33)

    # Creating the object of the Vectorizer
    sentence_vectorizer = SentenceVectorizer()

    #Building vocabulary
    sentence_vectorizer.fit(sentences)

    # Get the vocabulary
    vocab = sentence_vectorizer.vocab

    return sentence_vectorizer, vocab