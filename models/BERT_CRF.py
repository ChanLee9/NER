import torch
import torch.nn as nn
from transformers import BertModel
# from torchcrf import CRF

class CRF(object):
    def __init__(self, num_tags, batch_first=True):
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
        self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.transitions)
        nn.init.normal_(self.start_transitions)
        nn.init.normal_(self.end_transitions)
    
    def forward(self, emissions, tags, mask):
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        
        scores = self._compute_scores(emissions, tags, mask)
        partition = self._compute_log_partition(emissions, mask)
        return torch.mean(partition - scores)

    def decode(self, emissions, mask):
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        
        return self._viterbi_decode(emissions, mask)
    
    def _viterbi_decode(self, emissions, mask):
        seq_length = emissions.shape[0]
        mask = mask.float()
        # mask: (seq_length, batch_size)
        # emissions: (seq_length, batch_size, num_tags)
        transitions = self.transitions.unsqueeze(0)
        # transitions: (1, num_tags, num_tags)
        # emissions: (seq_length, batch_size, num_tags)
        scores = self.start_transitions + emissions[0]
        # scores: (batch_size, num_tags)
        paths = []
        for i in range(1, seq_length):
            # scores: (batch_size, num_tags, 1)
            # transitions: (1, num_tags, num_tags)
            # emissions[i]: (batch_size, num_tags)
            scores = scores.unsqueeze(2) + transitions + emissions[i].unsqueeze(1)
            # scores: (batch_size, num_tags, num_tags)
            scores, paths = torch.max(scores, dim=1)
            # scores: (batch_size, num_tags)
            # paths: (batch_size, num_tags)
            scores = scores.squeeze(1) * mask[i] + scores.squeeze(1) * (1 - mask[i])
            # scores: (batch_size, num_tags)
            paths = paths * mask[i].long() + paths * (1 - mask[i]).long()
            # paths: (batch_size, num_tags)
        scores = scores + self.end_transitions
        # scores: (batch_size, num_tags)
        best_score, best_path = torch.max(scores, dim=1)
        # best_score: (batch_size)
        # best_path: (batch_size)
        return best_path.tolist()
    
    def _compute_scores(self, emissions, tags, mask):
        seq_length = emissions.shape[0]
        mask = mask.float()
        # mask: (seq_length, batch_size)
        # emissions: (seq_length, batch_size, num_tags)
        scores = self.start_transitions + emissions[0]
        # scores: (batch_size, num_tags)
        for i in range(1, seq_length):
            # scores: (batch_size, num_tags, 1)
            # transitions: (1, num_tags, num_tags)
            # emissions[i]: (batch_size, num_tags)
            scores = scores.unsqueeze(2) + self.transitions + emissions[i].unsqueeze(1)
            # scores: (batch_size, num_tags, num_tags)
            new_scores = torch.logsumexp(scores, dim=1)
            # new_scores: (batch_size, num_tags)
            new_scores = new_scores * mask[i] + scores.squeeze(1) * (1 - mask[i])
            # new_scores: (batch_size, num_tags)
            scores = new_scores
        scores = scores + self.end_transitions
        # scores: (batch_size, num_tags)
        return torch.logsumexp(scores, dim=1)


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
