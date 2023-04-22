import torch
import torch.nn as nn
from torchcrf import CRF


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = 1970
        self.embedding_dim = 128
        self.pad_id = 0
        
        self.embed = nn.Embedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.pad_id
        )
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.config.hidden_size,
            batch_first = True,
            bidirectional = True
        )       # self - attention
        self.attention = nn.MultiheadAttention(2*self.config.hidden_size, 4)
        self.dropout = nn.Dropout(self.config.dropout)
        self.linear = nn.Linear(2*self.config.hidden_size, self.config.label_size)
        self.crf = CRF(self.config.label_size, batch_first=True)
    
    def forward(self, input, labels, mask):   
        output = self.get_lstm_results(input)
        output, _ = self.attention(output, output, output)
        output = self.linear(output)
        output = self.crf.decode(output, mask)
        return output

    def get_lstm_results(self, input):
        output = self.embed(input)
        output, _ = self.lstm(output)       # 只取hidden state
        return self.dropout(output)
    
    def loss_fn(self, input, label, mask):
        output = self.get_lstm_results(input)
        output, _ = self.attention(output, output, output)
        output = self.linear(output)
        if label != None:
            label = torch.where(label == -100, 0, label)
            loss = self.crf(output, label, mask)
        return (-1)*loss