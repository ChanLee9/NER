import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_path, output_hidden_states=True)
        self.fc1 = nn.Linear(24*1024, 1024)       # weight merge
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(1024, self.config.label_size)
        self.crf = CRF(self.config.label_size, batch_first=True)
        
    def forward(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask, output_hidden_states=True)
        # hidden_states: (1+num_layers, batch_size, seq_len, hidden_size)
        hidden_states = bert_output.hidden_states[1:]       # 不要embedding层的结果
        B, S, _ = hidden_states[0].shape    # batch_size, seq_len, embed_dim
        hidden_states = torch.stack(hidden_states, dim=3)   # 把新维度放到最后: (batch_size, seq_len, hidden_size, num_layers=24)
        hidden_states = hidden_states.view(B, S, -1)   
        # hidden_states： （batch_size, seq_len, 24*hidden_size)
        sequence_output = self.fc1(hidden_states)
        sequence_output = self.dropout(sequence_output)
        # sequence_output: (batch_size, sequence_length, embed_dim=768)
        logits = self.fc2(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        tags = self.crf.decode(logits, att_mask)
        # tags: List of list containing the best tag sequence for each batch.
        
        return tags 
    
    def loss_fn(self, input_ids, label_ids, mask):
        att_mask = torch.ne(mask, 0)
        bert_output = self.bert(input_ids, attention_mask=att_mask, output_hidden_states=True)
        # hidden_states: (1+num_layers, batch_size, seq_len, hidden_size)
        hidden_states = bert_output.hidden_states[1:]       # 不要embedding层的结果
        B, S, _ = hidden_states[0].shape    # batch_size, seq_len, embed_dim
        hidden_states = torch.stack(hidden_states, dim=3)   # 把新维度放到最后: (batch_size, seq_len, hidden_size, num_layers=24)
        hidden_states = hidden_states.view(B, S, -1)   
        # hidden_states： （batch_size, seq_len, 24*hidden_size)
        sequence_output = self.fc1(hidden_states)
        sequence_output = self.dropout(sequence_output)
        # sequence_output: (batch_size, sequence_length, embed_dim=768)
        logits = self.fc2(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        if label_ids != None:
            label_ids = torch.where(label_ids == -100, 0, label_ids)
            loss = self.crf(logits, label_ids, att_mask)
        return loss*(-1)      