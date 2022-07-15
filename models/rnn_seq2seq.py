from random import random

import torch
from torch import nn
import torch.nn.functional as F

MAX_LENGTH = 512

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout,embedding_weights=None):
        super().__init__()

        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim,padding_idx=1)

        if embedding_weights is not None:
            self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_weights))

        self.rnn = nn.GRU(emb_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src len, batch size]

        embedded = self.dropout(self.embedding(src).float())

        #embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded) #no cell state!

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout,embedding_weights=None):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim,padding_idx=1)

        if embedding_weights is not None:
            self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_weights))

        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        # self.fc_out = nn.Linear(hid_dim, output_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, context):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]

        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input).float())

        #embedded = [1, batch size, emb dim]
        emb_con = torch.cat((embedded, context), dim = 2)

        #emb_con = [1, batch size, emb dim + hid dim]
        output, hidden = self.rnn(emb_con, hidden)

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),dim = 1)

        #output = [batch size, emb dim + hid dim * 2]

        prediction = self.softmax(self.fc_out(output))

        #prediction = [batch size, output dim]

        return prediction, hidden



from torch.utils.data import DataLoader
#TEST configuration e2e
# train_iter = Hdf5Dataset(pl.Path(folder)/train_filename,num_entries=100000)
# train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE,collate_fn=collate_fn)
#
# encoder_hidden = encoder1.initHidden(BATCH_SIZE)
#
# for src, tgt in train_dataloader:
#     src = src.to(DEVICE)
#     tgt = tgt.to(DEVICE)
#     print("input shape: ",src.shape,encoder_hidden.shape)
#     enc_out,enc_hidden = encoder1(src,encoder_hidden)
#     print("encoder out shape: ",enc_out.shape,enc_hidden.shape)
#     decoder_input = tgt[0,:]
#     decoder_hidden = encoder_hidden
#     decoder_output, decoder_hidden = decoder1(decoder_input, decoder_hidden,input)
#     print("decoder out shape: ",enc_out.shape,enc_hidden.shape)
#
#     break