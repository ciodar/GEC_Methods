from typing import Iterable, List

import numpy as np
import pathlib as pl
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from utils import Hdf5Dataset

token_transform = get_tokenizer('basic_english')


def yield_tokens(data_iter: Iterable, index: int) -> List[str]:
    for data_sample in tqdm(data_iter):
        if data_sample[index] and isinstance(data_sample[index], str):
            yield token_transform(data_sample[index])


def build_vocab(dataset_iterator, col=1, vocab_size=None, out_folder=None,filename='vocab.pth'):
    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<UNK>', '<PAD>', '<BOS>', '<EOS>']

    # Create torchtext's Vocab object
    vocab_transform = build_vocab_from_iterator(yield_tokens(train_iter, col),
                                                max_tokens=vocab_size,
                                                specials=special_symbols,
                                                special_first=True)
    print("Built vocabulary")
    vocab_transform.set_default_index(UNK_IDX)
    torch.save(vocab_transform, pl.Path(out_folder) / filename)


def load_pretrained_embs(embedding_path, vocab):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<UNK>','<PAD>', '<BOS>', '<EOS>']

    embeddings_index = dict()
    f = open(embedding_path, encoding="utf-8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    vocab_size = len(vocab.vocab.itos_)

    embedding_matrix = np.zeros((vocab_size, 300))
    missing = 0

    for i, word, in tqdm(enumerate(vocab.vocab.itos_)):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            missing = missing +1
    print("Built embeddings. %d embedding missing"%missing)
    # pre-trained embeddings

    vocab.set_default_index(UNK_IDX)
    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.

    # print(vocab_npa[:10])
    embedding_matrix[1,:] = np.zeros((1, embedding_matrix.shape[1]))  # embedding for '<pad>' token.
    embedding_matrix[0,:] = np.mean(embedding_matrix, axis=0, keepdims=True)  # embedding for '<unk>' token.
    embedding_matrix[2,:] = np.random.rand(1, embedding_matrix.shape[1])
    embedding_matrix[3,:] = np.random.rand(1, embedding_matrix.shape[1])
    # insert embeddings for pad and unk tokens at top of embs_npa.
    return embedding_matrix, vocab

    # print(embs_npa.shape)
    # vec = Vectors(embedding_path)
    # vocab = Vocab(vec)
    # vocab_transform = vocab


if __name__ == "__main__":
    folder = "D:\\Datasets\\c4_200m\\data\\hdf5"
    vocab_folder = "vocab/"
    train_filename = 'C4_200M.hf5-00000-of-00010'

    N_SAMPLES = 1000000

    train_iter = Hdf5Dataset(pl.Path(folder) / train_filename, num_entries=N_SAMPLES)

    #Build vocabulary
    build_vocab(train_iter, out_folder=vocab_folder,vocab_size=20000,filename='vocab_20K.pth')

    #Build embeddings
    vocab = torch.load('vocab/vocab_20K.pth')
    embeddings, vocab = load_pretrained_embs(pl.Path('D:\Datasets\glove')/'glove.42B.300d.txt',vocab)
    torch.save(embeddings, pl.Path('vocab') / 'glove_42B_300d_20K.pth')
