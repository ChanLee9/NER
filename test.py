from transformers import AutoModel, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch
import torch.nn as nn

model_name_or_path = "pretrained_models/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path)
print(model)

if __name__ =="__main__":
    s = "你好，今天星期几？"
    encodings = tokenizer(s, return_tensors="pt")
    bert_output = model(**encodings, output_hidden_states=True)
    breakpoint()
