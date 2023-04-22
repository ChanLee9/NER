import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_path)
        self.dropout = nn.Dropout(self.config.dropout)
        self.clf = nn.Linear(1024, self.config.label_size)
        
    def forward(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        # bert_output: (last_hidden_state, pooler_output)
        sequence_output = bert_output.last_hidden_state
        # sequence_output: (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        cls_vector = self.clf(sequence_output)
        # cls_vector: (batch_size, seq_len, label_size)
        cls = torch.argmax(cls_vector, dim=2)
        # cls: (batch_size, seq_len)
        return cls
    
    def loss_fn(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        sequence_output = bert_output[0]    # last_hidden_state
        sequence_output = self.dropout(sequence_output)
        # sequence_output: (batch_size, sequence_length, embed_dim=768)
        logits = self.clf(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        if label_ids != None:
            label_ids = torch.where(label_ids == -100, 0, label_ids)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.config.label_size), label_ids.view(-1))
            # loss = criterion(torch.tensor(logits, dtype=torch.float), torch.tensor(label_ids, dtype=torch.float))
        return loss   