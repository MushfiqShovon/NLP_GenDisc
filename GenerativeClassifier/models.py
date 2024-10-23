import torch
import torch.nn as nn
import torch.nn.functional as F

class Gen(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot):

        
        super(Gen, self).__init__()

        print("dropout: ", dropout)
        print("use_cuda: ", use_cuda)
        print("tied: ", tied)
        print("use_bias: ", use_bias)
        print("concat_label: ", concat_label)
        print("avg_loss: ", avg_loss)
        print("one_hot: ", one_hot)
        
        self.drop = nn.Dropout(dropout)
        
        self.encoder = nn.Embedding(ntoken, ninp)
          
        self.label_encoder = nn.Embedding(nclass, nlabelembed)
        
        #self.bias_encoder = nn.Embedding(nclass, ntoken)

        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)

        self.decoder = nn.Linear((nhid + nlabelembed), ntoken, False)

        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()

    def get_one_hot(self, batch):
        ones = torch.eye(self.nclass)
        if self.use_cuda:
            ones = ones.cuda()
        return ones.index_select(0,batch)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, is_infer = False):
        
        
        embedded_sents = self.encoder(x.data)

        embedded_label = self.label_encoder(y_ext.data)

        print("Shape of embedded_sents:", embedded_sents.shape) 
        #print("Shape of hidden:", hidden.shape)  

        output, (_, _) = self.rnn(embedded_sents, hidden)

        print("Shape of output (from LSTM):", output.shape)  # Should be [batch_size, seq_len, nhid]
        print("Shape of embedded_label (before unsqueeze and repeat):", embedded_label.shape)  # Should be [batch_size, label_emb_dim]

        hidden_data = torch.cat((output.data, embedded_label), 1)

        print("Shape of concatenated hidden_data:", hidden_data.shape)

        hidden_data = self.drop(hidden_data)

        # out: seq_len * n_token.
        out = self.decoder(hidden_data)

        loss = criterion(out, x_pred.data)

        if is_infer:
            LM_loss = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(
                loss, x.batch_sizes))[0].transpose(0,1)
            total_loss = torch.sum(LM_loss, dim = 1)
            return total_loss
        else:
            if self.avg_loss:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        # Return hidden state and cell state as 3D tensors
        return (weight.new_zeros(self.nlayers, self.nhid),
                weight.new_zeros(self.nlayers, self.nhid))
