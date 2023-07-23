from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class MyCRF(CRF):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        super().__init__(num_tags, batch_first)
    
    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            condition = torch.unsqueeze(mask[i], 1).bool()
            score = torch.where(condition, next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            condition = condition = torch.unsqueeze(mask[i], 1).bool()
            score = torch.where(condition, next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

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
        self.crf = MyCRF(self.config.label_size, batch_first=True)
        
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
