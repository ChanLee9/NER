import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchcrf import CRF
from config import *

class Bert_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.bert = BertModel.from_pretrained(BERT_PATH, output_hidden_states=True)
        self.fc1 = nn.Linear(12*768, 768)       # weight merge
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(768, LABEL_SIZE)
        self.crf = CRF(LABEL_SIZE, batch_first=True)
        
    def forward(self, input_ids, labels=None, mask=None):
        att_mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        hidden_states = bert_output[2][1:]  # 除去embdding层
        B, S, _ = hidden_states[0].shape    # batch_size, seq_len, embed_dim
        hidden_states = torch.stack(hidden_states, dim=3)   # 把新维度放到最后: (batch_size, seq_len, embed_dim, num_layers=12)
        hidden_states = hidden_states.view(B, S, -1)    
        # hidden_states： （batch_size, seq_len, 12*embed_dim)
        sequence_output = self.fc1(hidden_states)
        sequence_output = self.dropout(sequence_output)
        # sequence_output: (batch_size, sequence_length, embed_dim=768)
        logits = self.fc2(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        tags = self.crf.decode(logits, att_mask)
        # tags: List of list containing the best tag sequence for each batch.
        
        return tags 
    
    def loss_fn(self, input_ids, label_ids, mask):
        att_mask = torch.ne(input_ids, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask)
        hidden_states = bert_output[2][1:]  # 除去embdding层
        B, S, _ = hidden_states[0].shape    # batch_size, seq_len, embed_dim
        hidden_states = torch.stack(hidden_states, dim=3)   # 把新维度放到最后: (batch_size, seq_len, embed_dim, num_layers=12)
        hidden_states = hidden_states.view(B, S, -1)    
        # hidden_states： （batch_size, seq_len, 12*embed_dim)
        sequence_output = self.fc1(hidden_states)
        sequence_output = self.dropout(sequence_output)
        # sequence_output: (batch_size, sequence_length, embed_dim=768)
        logits = self.fc2(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        if label_ids != None:
            loss = self.crf(logits, label_ids, att_mask)
        return loss*(-1)      