# class Dataset(data.Dataset):
#     def __init__(self, config, data) -> None:
#         super().__init__()
#         self.config = config
#         self.data = data
#         self.label2id = self.get_dict('data/label.txt')
#         self.char2id = self.get_dict('data/vocab.txt')
        
#     def get_dict(self, dict_path):
#         '''
#         将存起来的vocab和labels文件转化成字典映射
#         '''
#         df = pd.read_csv(dict_path, names=['char', 'id'])
#         mapping = {}
#         for i in range(1, len(df)):
#             mapping[df['char'][i]] = df['id'][i]
#         return mapping
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         text = self.data['text'][idx]
#         label = self.data['BIO_anno'][idx]
#         # 把label中的字符转化成对应的数值标签
#         # label_num = [self.label2id.get(v, self.config.pad_id) for v in label.split(' ')]
#         return text, label

# class Dataloader():
#     def __init__(self, config, dataset) -> None:
#         self.config = config
#         self.dataset = dataset
#         self.unk = 1
#         self.label2id = self.dataset.label2id
#         self.dataloader = self.get_dataloader(dataset)
    
#     def tokenize(self, texts, labels):
#         # 给无预训练tokenizer的模型编码
#         text_nums, label_nums, masks = [], [], []
#         char2id = self.dataset.char2id
#         max_len = len(texts[0])
#         for i in range(len(texts)):
#             mask = [1] * max_len
#             label = labels[i]
#             label_num = [int(self.label2id.get(v, 0)) for v in label]
#             pad_len = max_len - len(texts[i])
#             mask[len(texts[i]):] = pad_len * [-100]
#             masks.append(mask)
#             text_li = list(texts[i])   
#             label_li = label_num + pad_len * [-100]     # [PAD]取-100不计算
#             row_text_num = [int(char2id.get(v, self.unk)) for v in text_li]
#             text_nums.append(row_text_num + pad_len * [0] )
#             label_nums.append(label_li)
#             assert len(text_nums) == len(label_nums), 'wrong len of text and label'
            
#         return text_nums, label_nums, masks
                   
#     def collate_fn(self, batch_samples):
#         # 按照当前批次中文本长度排序，方便pad
#         batch_samples.sort(key=lambda x:len(x[0]), reverse=True)
#         # max_len = len(batch_samples[0][0])
#         texts, labels = [], []
#         for sample in batch_samples:
#             texts.append(sample[0])
#             labels.append(sample[1].split(' '))
#             assert len(sample[0]) == len(sample[1].split(' ')), 'wrong length of text and label'
#         if 'BERT' not in self.config.model:
#             # 不用预训练模型
#             text_encod, label_encod, masks = self.tokenize(texts, labels)
#             return torch.LongTensor(text_encod).to(self.config.device), \
#                 torch.LongTensor(label_encod).to(self.config.device), \
#                 torch.BoolTensor(masks).to(self.config.device)
#         else:
#             tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_path)
#             text_encod = tokenizer(
#                 texts, 
#                 padding=True,
#                 truncation=True,
#                 return_tensors='pt',
#                 max_length=256
#             )
#             formated_labels = self.formated_label(labels)
#             label_encod = torch.zeros_like(
#                 text_encod['input_ids']
#             )
#             for idx, text in enumerate(texts):
#                 encoding = tokenizer(text, truncation=True, max_length=256, padding=True)
#                 label_encod[idx][0] = -100      # [CLS]  
#                 label_encod[idx][len(encoding.tokens())-1:] = -100      # [SEP], [PAD] 
#                 for char_start, char_end, tag in formated_labels[idx]:
#                     token_start = encoding.char_to_token(char_start)
#                     token_end = encoding.char_to_token(char_end)
#                     label_encod[idx][token_start] = int(self.label2id[f'B-{tag}'])
#                     label_encod[idx][token_start+1:token_end+1] = int(self.label2id[f'I-{tag}'])
#         return text_encod.to(self.config.device), torch.LongTensor(label_encod).to(self.config.device)
    
#     # def formated_label(self, labels):
#     #     formated_labels = [[] for _ in range(len(labels))]
#     #     for i, label in enumerate(labels):
#     #         idx = 0
#     #         while idx < len(label):
#     #             if label[idx].startswith('B-'):
#     #                 start_idx = idx
#     #                 while idx < len(label) and label[idx] != 'O':
#     #                     idx += 1
#     #                 end_idx = idx - 1
#     #                 formated_labels[i].append([start_idx, end_idx, label[start_idx][2:]])
#     #                 # 结果类似于[7, 9, BANK]
#     #             idx += 1
#     #     return formated_labels       
    
#     def formated_label(self, labels):
#         formated_labels = [[] for _ in range(len(labels))]
#         for i, label in enumerate(labels):
#             for j, v in enumerate(label):
#                 if v.startswith('B-'):
#                     formated_labels[i].append([j, j, v[2:]])
#                 elif v.startswith('I-'):
#                     formated_labels[i][-1][1] = j
#         return formated_labels     
                    
#     def get_dataloader(self, dataset):
#         dataloader = data.DataLoader(
#             dataset,
#             batch_size=self.config.batch_size,
#             shuffle=True,
#             # num_workers=self.config.num_workers,
#             collate_fn=self.collate_fn
#         )
#         return dataloader