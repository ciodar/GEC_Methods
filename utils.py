import time

from tqdm import tqdm
import pathlib as pl
import pandas as pd
import torch.utils.data.datapipes as dp
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
df_cols_to_index = ['input', 'labels']


def csv_to_iterable(p, nr):
    data = pd.read_csv(p, sep='\t', nrows=nr, header=None, names=df_cols_to_index, dtype=(str, str))
    data = data.reset_index(drop=True)
    return data.itertuples(index=False)


def csv_to_df(p, nr):
    data = pd.read_csv(p, sep='\t', nrows=nr, header=None, names=df_cols_to_index)
    data = data.reset_index(drop=True)
    return list(zip(data.iloc[:, 0].to_numpy(), data.iloc[:, 1].to_numpy()))


def folder_to_dp(p):
    datapipe = dp.iter.FileLister([p]).filter(filter_fn=lambda filename: filter in filename)
    datapipe = dp.iter.FileOpener(datapipe, mode='rt', encoding='utf-8')
    datapipe = datapipe.parse_csv(delimiter="\t", skip_lines=0)
    # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
    datapipe = datapipe.shuffle()
    return datapipe


def csv_to_hf5(csv_path, num_lines=1000000, chunksize=100000, columns=None):
    if columns is None:
        columns = ['input', 'labels']
    csv_path = pl.Path(csv_path)

    hdf_filename = csv_path.parent / pl.Path(csv_path).name.replace('.tsv', '.hf5')

    # suppose this is a large CSV that does not
    # fit into memory:

    # Get number of lines in the CSV file if it's on your hard drive:
    # num_lines = subprocess.check_output(['wc', '-l', in_csv])
    # num_lines = int(nlines.split()[0])
    # use 10,000 or 100,000 or so for large files

    dt = h5py.special_dtype(vlen=str)

    # this is your HDF5 database:
    with h5py.File(hdf_filename, 'w') as h5f:

        # use num_features-1 if the csv file has a column header
        # store = pd.HDFStore(hdf_filename,format="table")
        dset1 = h5f.create_dataset('input',
                                   shape=(num_lines,),
                                   compression=4,
                                   dtype=dt
                                   )
        dset2 = h5f.create_dataset('labels',
                                   shape=(num_lines,),
                                   compression=4,
                                   dtype=dt
                                   )

        times = []
        # change range argument from 0 -> 1 if your csv file contains a column header
        for i in tqdm(range(0, num_lines, chunksize)):

            df = pd.read_csv(csv_path,
                             sep='\t',
                             names=columns,
                             header=None,  # no header, define column header manually later
                             nrows=chunksize,  # number of rows to read at each iteration
                             skiprows=i,
                             )  # skip rows that were already read

            features = df.input.values.astype(str)
            labels = df.labels.values.astype(str)
            # store.append(value=)
            start_time = time.time()
            # use i-1 and i-1+10 if csv file has a column header
            dset1[i:i + chunksize] = features
            dset2[i:i + chunksize] = labels
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
    return times

def csv_line_count(path):
    n = 0
    with pd.read_csv(path, sep="\t", chunksize=int(2e15), iterator=True,header=None) as reader:
        for chunk in reader:
            n += len(chunk)
    return n

class CSVDataset(Dataset):
    def __init__(self, csv_path, chunkSize=10000,length=18000000):
        self.chunksize = chunkSize
        self.length = length
        self.reader = pd.read_csv(csv_path, sep='\t', chunksize=self.chunksize, header=None, iterator=True)
        self.index = -1
        self.data = None

    def __len__(self):
        return self.length // self.chunksize

    def __getitem__(self, index):
        data = self.reader.get_chunk(self.chunksize)
        inputs = data.iloc[:, 0].values.astype(str)
        labels = data.iloc[:, 1].values.astype(str)
        return list(zip(inputs,labels))



class Hdf5Dataset(Dataset):
    """Custom Dataset for loading entries from HDF5 databases"""

    def __init__(self, h5_path, transform=None,num_entries = None):

        self.h5f = h5py.File(h5_path, 'r')
        if num_entries:
            self.num_entries = num_entries
        else:
            self.num_entries = self.h5f['labels'].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        if index > self.num_entries:
            raise StopIteration
        input = self.h5f['input'][index].decode('utf-8')
        label = self.h5f['labels'][index].decode('utf-8')
        if self.transform is not None:
            features = self.transform(input)
        return input, label

    def __len__(self):
        return self.num_entries

# # Define special symbols and indices
# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# # Make sure the tokens are in order of their indices to properly insert them in vocab
# special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
#
# vector = GloVe(name='6B', dim=100)
# for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
#     # Training data Iterator
#     train_iter = Hdf5Dataset(pl.Path(folder)/train_filename)
#     # Create torchtext's Vocab object
#     # i = (train_iter.__getitem__(2))
#     # ret = vec.get_vecs_by_tokens(examples, lower_case_backup=True)
#     # vocab_transform[ln] = vec.stoi
#     # vocab_transform[ln] = Vocab(c,specials=special_symbols,vectors=vectors)
#
#     vocab_transform[ln] = torch.load(str(pl.Path('D:\Datasets\c4_200m\checkpoints')/'vocab_input.pth'))
#     # build_vocab_from_iterator(yield_tokens(train_iter, ln),
#     #                                             max_tokens=VOCAB_SIZE,
#     #                                             specials=special_symbols,
#     #                                             special_first=True)
# #
# # # # Set UNK_IDX as the default index. This index is returned when the token is not found.
# # # # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
# for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
#     vocab_transform[ln].set_default_index(UNK_IDX)
# torch.save(vocab_transform[SRC_LANGUAGE], str(vocab_path))
# vocab_transform

if __name__ == "__main__":
    folder = "D:\\Datasets\\c4_200m\\data\\tsv"
    # for x in pl.Path(folder).iterdir():
    #     if x.is_file() and '.tsv' in x.name:
    #         csv_path = pl.Path(folder) / x.name
    #         n = csv_line_count(csv_path)
    #         csv_to_hf5(str(csv_path),num_lines=n)
            # print_rows(x)

    # 18386521
    filename = 'C4_200M.tsv-00003-of-00010'
    csv_path = pl.Path(folder) / filename
    n = csv_line_count(csv_path)
    elapsed = csv_to_hf5(str(csv_path),num_lines=n)

    plt.plot(elapsed)

    # data = pd.read_csv(csv_path, sep='\t', nrows=10000)
    # hdf_filename = csv_path.parent / pl.Path(csv_path).name.replace('.tsv', '.hf5')
    #
    # print("reading ", hdf_filename)
    # with h5py.File(hdf_filename, 'r') as h5f:
    #     print(h5f['input'].shape)
    #     print(h5f['labels'].shape)
    #
    # with h5py.File(hdf_filename, 'r') as h5f:
    #     print('Features of entry no. 99:', h5f['input'][0])
    #     print('Class label of entry no. 99:', h5f['labels'][0])
