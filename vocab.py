from typing import Iterable, List

import numpy as np
import pathlib as pl
import torch
import torchtext as text
from tqdm import tqdm

from utils import Hdf5Dataset

# Create a blank Tokenizer with just the English vocab
token_transform = text.data.utils.get_tokenizer('spacy', language='en_core_web_sm')

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
    vocab_transform = text.vocab.build_vocab_from_iterator(yield_tokens(train_iter, col),
                                                max_tokens=vocab_size,
                                                specials=special_symbols,
                                                special_first=True)
    print("Built vocabulary")
    vocab_transform.set_default_index(UNK_IDX)
    torch.save(vocab_transform, pl.Path(out_folder) / filename)

    # print(embs_npa.shape)
    # vec = Vectors(embedding_path)
    # vocab = Vocab(vec)
    # vocab_transform = vocab

if __name__ == "__main__":
    folder = "dataset/"
    vocab_folder = "vocab/"
    train_filename = 'train.hf5'

    N_SAMPLES = 1000000

    train_iter = Hdf5Dataset(pl.Path(folder) / train_filename, num_entries=N_SAMPLES)

    #Build vocabulary
    build_vocab(train_iter,col=1, out_folder=vocab_folder,vocab_size=None,filename='tgt_vocab_20K_spacy.pth')

    #Build embeddings
    # vocab = torch.load('vocab/vocab_nltk.pth')
    # embeddings, vocab = load_pretrained_embs(pl.Path('D:\Datasets\glove')/'glove.42B.300d.txt',vocab)
    # torch.save(embeddings, pl.Path('vocab') / 'glove_42B_300d_20K.pth')
