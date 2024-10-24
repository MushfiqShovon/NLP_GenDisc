class Gen(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot):
        
        super(Gen, self).__init__()
        
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

        output, (_, _) = self.rnn(embedded_sents, hidden)

        hidden_data = torch.cat((output.data, embedded_label), 1)

        hidden_data = self.drop(hidden_data)

        out = self.decoder(hidden_data)
        
        return out


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, self.nhid),
                weight.new_zeros(self.nlayers, self.nhid))
