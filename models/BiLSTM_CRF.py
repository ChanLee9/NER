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
        )
        self.linear = nn.Linear(2*self.config.hidden_size, self.config.label_size)      # 2*HIDDEN_SIZE  since bidirectional
        self.dropout = nn.Dropout(self.config.dropout)
        self.crf = CRF(self.config.label_size, batch_first=True)
    
    def forward(self, input, labels, mask):  
        
        output = self.get_lstm_results(input)
        output = self.crf.decode(output, mask)
        return output

    def get_lstm_results(self, input):
        # input: (batch_size, seq_len) 
        output = self.embed(input)
        # output: (batch_size, seq_len, EMBEDDING_DIM)
        output, _  = self.lstm(output)       # 只取结果，不要隐状态和细胞态
        # output: (batch_size, seq_len, 2*HIDDEN_SIZE)
        output = self.linear(output)
        # out_put: (batch_size, seq_len, LABEL_SIZE)
        return self.dropout(output)
    
    def loss_fn(self, input, label, mask):
        y_pred = self.get_lstm_results(input)
        if label != None:
            label = torch.where(label == -100, 0, label)
            loss = self.crf(y_pred, label, mask)
        return (-1)*loss
