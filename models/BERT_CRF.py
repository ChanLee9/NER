import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.merge_weight = config.merge_weight
        
        self.bert = BertModel.from_pretrained(self.config.model_name_or_path)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.merge_weight is not None:
            self.merge_fc = nn.Linear(
                self.bert.config.num_hidden_layers * self.bert.config.hidden_size, 
                self.bert.config.hidden_size
                )
        self.fc = nn.Linear(self.bert.config.hidden_size, self.config.label_size)
        self.crf = CRF(self.config.label_size, batch_first=True)
        
    def forward(self, item):
        logits, mask = self.comput_logits(item)
        tags = self.crf.decode(logits, mask)
        # tags: List of list containing the best tag sequence for each batch.
        return tags 
    
    def comput_logits(self, item):
        texts_encoding = item["texts_encoding"]
        input_ids, mask = texts_encoding["input_ids"], texts_encoding["attention_mask"]
        bert_output = self.bert(input_ids, attention_mask=mask, output_hidden_states=True)
        
        # bert_output: (last_hidden_state, pooler_output, hidden_states)
        sequence_output = bert_output.last_hidden_state
        if self.merge_weight is not None:
            sequence_output = torch.cat(bert_output.hidden_states[1:], dim=-1)
            sequence_output = self.merge_fc(sequence_output)
            
        # sequence_output: (batch_size, sequence_length, hidden_dim)
        sequence_output = self.dropout(sequence_output)
        
        logits = self.fc(sequence_output)
        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        return logits, mask
    
    def loss_fn(self, item):
        logits, mask = self.comput_logits(item)
        labels = item["labels"]
        if labels is not None:
            labels = torch.where(labels == -100, 0, labels)
            loss = self.crf(logits, labels, mask)
        return loss*(-1)     
