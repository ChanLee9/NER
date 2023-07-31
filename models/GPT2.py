import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel, AutoTokenizer

class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.whole_model = GPT2LMHeadModel.from_pretrained(config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.gpt = self.whole_model.transformer
        self.lm_head = self.whole_model.lm_head
        self.criterion = nn.CrossEntropyLoss()
        
    
    def forward(self, item):
        encodings = item["texts_encoding"]
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        
        logits = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = self.lm_head(logits.last_hidden_state)
        logits = logits.argmax(dim=-1).detach().cpu().numpy()
        
        ret_chars = []
        for logit in logits:
            decoded_tokeens = self.tokenizer.batch_decode(logit, skip_special_tokens=True)
            li = "".join(decoded_tokeens).split("###它的答案是：")[1].replace("[", "").replace("]", "").upper().split(",")
            ret_chars.append([char.replace("'", "") for char in li])
        breakpoint()
        return ret_chars
        
    
    def loss_fn(self, item):
        encodings = item["texts_encoding"]
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
        labels = input_ids.clone()
        
        logits = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        logits = logits.last_hidden_state
        logits = self.lm_head(logits)
        
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        
        loss = self.criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        
        return loss

