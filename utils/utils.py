import torch
import torch.utils.data as data
from transformers import AutoTokenizer
import pandas as pd

from preprocess import DataProcessor
from dataclasses import dataclass

class MyDataset(data.Dataset):
    def __init__(self, raw_data, label_path=None) -> None:
        super().__init__()
        if label_path is None:
            self.LABEL_PATH = "../data/label.txt"
        self.label2id = self.get_label_id_map()
        self.dataset = self.get_dataset(raw_data)
    
    def get_dataset(self, data):
        """
        在原始数据集中新增一列label列, 包含这一行样本所包含的实体以及它的位置：{
            "entities": [(entity1, type_of_entity1, start1, end1), (entity2, type_of_entity2, start2, end2), ...], 
                注意这里的实体范围为[start, end)
            "starts": [list of starts],
            "ends": [list of ends]
        }
        """
        def make_label(item):
            """
            把每行元素转化成需要的label列
            """
            text, anno = item["text"], item["BIO_anno"]
            res = {
                "entities": [],
                "starts": [],
                "ends": []
            }
            i = 0
            while i < len(anno):
                # 如果是杂项
                if anno[i] == "O":
                    res['starts'].append(self.label2id["O"])
                    res['ends'].append(self.label2id["O"])
                    i += 1

                # 如果是某个实体开头
                else:
                    res['starts'].append(self.label2id[anno[i]])
                    entity = anno[i][2:]

                    # 定位当前实体
                    j = i + 1
                    while j < len(anno) and anno[j] == f"I-{entity}":
                        j += 1
                    res['entities'].append((text[i:j], entity, i, j))

                    # 中间跳过了 j-i-1 个字符
                    res['starts'] += [self.label2id["O"]] * (j-i-1)
                    res['ends'] += [self.label2id["O"]] * (j-i-1)
                    res['ends'].append(self.label2id[anno[j-1]])

                    # 跳过这个实体
                    i = j

            return res        

        data["labels"] = data.apply(lambda x: make_label(x), axis=1)

        return data

    def get_label_id_map(self):
        """
        获取label到id的映射字典
        """
        labels = pd.read_csv(self.LABEL_PATH)
        label2id = {}
        for idx, item in labels.iterrows():
            label, id = item["label"], item["id"]
            label2id[label] = int(id)
        return label2id

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text, anno, label = self.dataset.iloc[idx, 1], self.dataset.iloc[idx, 2], self.dataset.iloc[idx, 4]
        return text, anno, label


class MyDataLoader():
    def __init__(self, config, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.label2id = dataset.label2id
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    def collate_fn(self, batch):
        # 把batch中的数据按照长度排序，以便添加padding, batch: (text, anno, label)
        batch.sort(key=lambda x: len(x[0]), reverse=True)    
        texts, annos, entities, starts, ends = [], [], [], [], []
        for item in batch:
            text, anno, label = item
            texts.append(text)
            annos.append(anno)
            entities.append(label["entities"])
            starts.append(label["starts"])
            ends.append(label["ends"])
        
        texts_encoding = self.tokenizer(texts, 
                                      return_tensors='pt', 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=self.max_length
                                      )

        # 生成labels
        labels = torch.fill(texts_encoding['input_ids'], self.label2id["O"])    
            
        for idx, text in enumerate(texts):
            encoding = self.tokenizer(text, 
                                      return_tensors='pt', 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=self.max_length
                                    )
            
            # [CLS]和[SEP]和[PADDING]的label设为-100    
            labels[idx][0] = -100
            labels[idx][len(encoding.tokens())-1:] = -100
            
            # 将 labels 其余部分转化成 id
            for item in entities[idx]:
                entity, char_start, char_end = item[1], item[2], item[3]
                
                # 通过char_start和char_end找到对应的token_start和token_end
                token_start, token_end = encoding.char_to_token(char_start), encoding.char_to_token(char_end-1)
                labels[idx][token_start] = self.label2id[f"B-{entity}"]
                labels[idx][token_start+1:token_end+1] = self.label2id[f"I-{entity}"]
                
        return texts_encoding, labels, starts, ends
    
                
    def get_dataloader(self):
        dataloader = data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return dataloader

if __name__ == "__main__":
    @dataclass
    class Config():
        span = 60
        data_path = "../data/train_data_public.csv"
        save_path = "../save_dir"
        augmentation_level = 4
        batch_size = 32
        max_length = 64
        model_name_or_path = "../pretrained_model/bert-base-chinese"

    config = Config()
    data_processer = DataProcessor(config)
    raw_data = data_processer.raw_data
    print(
        f"class distribution before: {data_processer.raw_data['class'].value_counts()}")
    
    data_processer.data_augmentation()
    data_processer.split_long_texts()
    print(
        f"class distribution after: {data_processer.raw_data['class'].value_counts()}")
    
    my_dataset = MyDataset(data_processer.raw_data)
    print(my_dataset.dataset)
    my_dataloader = MyDataLoader(config, my_dataset).get_dataloader()
    temp_data = next(iter(my_dataloader))
    breakpoint()
    