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


def load_pretrained_embs(embedding_path, vocab_npa, vocab_size=None):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>','<pad>', '[CLS]', '[SEP]']

    vocab, embeddings = [], []
    with open(embedding_path, 'rt', encoding='utf-8') as fi:
        full_content = fi.read().strip().split('\n')
        if vocab_size is None:
            vocab_size = len(full_content)
    for i in tqdm(range(vocab_size)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    # pre-trained embeddings
    vocab_npa = np.insert(vocab_npa, UNK_IDX, '<pad>')
    vocab_npa = np.insert(vocab_npa, PAD_IDX, '<unk>')
    vocab_npa = np.insert(vocab_npa, BOS_IDX, '<bos>')
    vocab_npa = np.insert(vocab_npa, EOS_IDX, '<eos>')

    vocab_npa.set_default_index(UNK_IDX)
    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.

    # print(vocab_npa[:10])
    pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)  # embedding for '<unk>' token.
    bos_emb_npa = np.random.rand(1, embs_npa.shape[1])
    eos_emb_npa = np.random.rand(1, embs_npa.shape[1])
    # insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((unk_emb_npa, pad_emb_npa, bos_emb_npa, eos_emb_npa, embs_npa))
    return embs_npa, vocab_npa

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
    build_vocab(train_iter, out_folder=vocab_folder,vocab_size=20000,filename='vocab_20K.pth')
