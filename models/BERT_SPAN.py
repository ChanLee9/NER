import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel

class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.label2id, self.id2label = self.get_label_id_map(config.label_path)
        self.merge_weight = config.merge_weight
        self.bert = BertModel.from_pretrained(self.config.model_name_or_path)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.merge_weight:
            self.merge_fc = nn.Linear(
                self.bert.config.num_hidden_layers * self.bert.config.hidden_size, 
                self.bert.config.hidden_size
                )
        self.start_fc = nn.Linear(self.bert.config.hidden_size, self.config.label_size)
        self.end_fc = nn.Linear(self.bert.config.hidden_size, self.config.label_size)
    
    
    def get_label_id_map(self, label_path):
        """
        获取label到id的映射字典和id到label的映射字典
        """
        labels = pd.read_csv(label_path)
        label2id = {}
        for idx, item in labels.iterrows():
            label, id = item["label"], item["id"]
            label2id[label] = int(id)
        id2label = {}
        for item in label2id:
            id2label[label2id[item]] = item
        return label2id, id2label
    
    def compute_logits(self, item):
        encodings = item["texts_encoding"]
        input_ids, mask = encodings["input_ids"], encodings["attention_mask"]

        # bert_output: (last_hidden_state, pooler_output, last_hidden_states)
        bert_output = self.bert(input_ids, attention_mask=mask, output_hidden_states=True)
        sequence_output = bert_output.last_hidden_state
        
        if self.merge_weight:
            sequence_output = torch.cat(bert_output.hidden_states[1:], dim=-1)
            
            # sequence_output: (batch_size, sequence_length, hidden_dim)
            sequence_output = self.merge_fc(sequence_output)

        sequence_output = self.dropout(sequence_output)

        # logits: (batch_size, seq_len, LABEL_SIZE=9)
        start_logits = self.start_fc(sequence_output)
        end_logits = self.end_fc(sequence_output)
        
        return start_logits, end_logits 
        
    def forward(self, item):
        start_logits, end_logits = self.compute_logits(item)
        
        start_idx = torch.argmax(start_logits, dim=2)
        end_idx = torch.argmax(end_logits, dim=2)
        
        pred_idx = self.make_pred_idx(start_idx, end_idx)
        return pred_idx.clone().detach().tolist()
    
    def loss_fn(self, item):
        start_logits, end_logits = self.compute_logits(item)
        starts, ends = item["starts"], item["ends"]
        criterion = CrossEntropyLoss(reduction="sum")
        start_logits = start_logits.view(-1, self.config.label_size)
        end_logits = end_logits.view(-1, self.config.label_size)
        starts = starts.view(-1).long()
        ends = ends.view(-1).long()
        loss = criterion(start_logits, starts) + criterion(end_logits, ends)
        return loss    

    def make_pred_idx(self, start_idx, end_idx):
        """从starts_idx和endd_idx中得到pred_idx

        Args:
            satrt_idx (_type_): (batch_size, seq_len, 1)
            end_idx (_type_): (batch_size, seq_len, 1)
        """
        pred_idx = torch.zeros(start_idx.size())
        batch_size, seq_len = start_idx.shape
        
        for batch in range(batch_size):
            i = 0
            while i < seq_len:
                j = i
                if start_idx[batch, i] != 0:
                    start = start_idx[batch, i]
                    pred_idx[batch, i] = start
                    
                    while  j < seq_len and end_idx[batch, j] == 0:
                        j += 1
                    
                    if j < seq_len:
                        # 此时[i+1:j]的值应该取end_idx[batch, j+1,]
                        pred_idx[batch, i+1:j+1] = end_idx[batch, j] 
                    else:
                        # 被截断了，导致后面没取到end_idx就结束了，此时我们应该把剩下部分填充为self.label2id[f"I-"{self.id2label[start][2:]}]
                        entity = self.id2label[int(start)][2:]
                        pred_idx[batch, i+1:] = torch.tensor(self.label2id[f"I-{entity}"], device=start_idx.device)
                    
                i = j + 1
        
        return pred_idx 
        