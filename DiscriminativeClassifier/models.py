import torch.nn as nn
import torch.nn.utils.rnn

#Class for full precision Discriminative Model
class LSTMNet(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(LSTMNet,self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)
        
        packed_output,(hidden_state,cell_state) = self.lstm(embedded)

        hidden = hidden_state[-1,:,:]
        
        dense_outputs=self.fc(hidden)

        return dense_outputs