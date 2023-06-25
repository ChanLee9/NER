import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_path)
        self.dropout = nn.Dropout(self.config.dropout)
        self.start_fc = nn.Linear(1024, self.config.label_size)
        self.end_fc = nn.Linear(1024, self.config.label_size)
        
    def forward(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)

        # bert_output: (last_hidden_state, pooler_output)
        bert_output = self.bert(input_ids, attention_mask=att_mask)

        # sequence_output: (batch_size, sequence_length, embed_dim=1024)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        start_logits = self.start_fc(sequence_output)
        end_logits = self.end_fc(sequence_output)

        start_idx = torch.argmax(start_logits, dim=2)
        end_idx = torch.argmax(end_logits, dim=2)
        pred_idx = start_idx + end_idx

        return pred_idx 
    
    def loss_fn(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        sequence_output = bert_output.last_hidden_state   

        # sequence_output: (batch_size, sequence_length, embed_dim=1024)
        sequence_output = self.dropout(sequence_output)

        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        start_logits = self.start_fc(sequence_output)
        end_logits = self.end_fc(sequence_output)

        start_idx = torch.argmax(start_logits, dim=2)
        end_idx = torch.argmax(end_logits, dim=2)
        pred_idx = start_idx + end_idx

        criterion = CrossEntropyLoss()
        label_ids = torch.where(label_ids == -100, 0, label_ids)
        loss = criterion(pred_idx, label_ids)
        return loss    

