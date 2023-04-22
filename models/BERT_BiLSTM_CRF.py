import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_path)
        self.lstm = nn.LSTM(
            1024,
            self.config.hidden_size,
            batch_first = True,
            bidirectional = True
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.linear = nn.Linear(self.config.hidden_size*2, self.config.label_size)
        self.crf = CRF(self.config.label_size, batch_first=True)
        
    def forward(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        # bert_output: (last_hidden_state, pooler_output)
        sequence_output = bert_output[0]    # (last_hidden_state, pooler_output)
        # sequence_output: (batch_size, seq_len, embed_dim)
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        # sequence_output: (batch_size, seq_len, HIDDEN_SIZE*2)
        logits = self.linear(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE)
        tags = self.crf.decode(logits, att_mask)
        return tags 
    
    def loss_fn(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        # bert_output: (last_hidden_state, pooler_output)
        sequence_output = bert_output[0]      # (last_hidden_state, pooler_output)
        # sequence_output: (batch_size, seq_len, embed_dim)
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        # sequence_output: (batch_size, seq_len, HIDDEN_SIZE*2)
        logits = self.linear(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE)
        if label_ids != None:
            label_ids = torch.where(label_ids == -100, 0, label_ids)
            loss = self.crf(logits, label_ids, att_mask)
        return loss*(-1)  


