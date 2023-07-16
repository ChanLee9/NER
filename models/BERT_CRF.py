import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_path)
        self.dropout = nn.Dropout(self.config.dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.config.label_size)
        self.crf = CRF(self.config.label_size, batch_first=True)
        
    def forward(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        # bert_output: (last_hidden_state, pooler_output)
        sequence_output = bert_output.last_hidden_state
        # sequence_output: (batch_size, sequence_length, embed_dim=768)
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        tags = self.crf.decode(logits, att_mask)
        # tags: List of list containing the best tag sequence for each batch.
        
        return tags 
    
    def loss_fn(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        sequence_output = bert_output[0]    # last_hidden_state
        sequence_output = self.dropout(sequence_output)
        # sequence_output: (batch_size, sequence_length, embed_dim=768)
        logits = self.linear(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        if label_ids != None:
            label_ids = torch.where(label_ids == -100, 0, label_ids)
            loss = self.crf(logits, label_ids, att_mask)
        return loss*(-1)     