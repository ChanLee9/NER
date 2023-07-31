import torch
import torch.utils.data as data
from transformers import AutoTokenizer
import pandas as pd

from .preprocess import DataProcessor
from dataclasses import dataclass

class MyDataset(data.Dataset):
    def __init__(self, config, raw_data, label_path=None) -> None:
        super().__init__()
        if label_path is None:
            self.LABEL_PATH = config.label_path
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
                "starts": [-100,],
                "ends": [-100,]
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
        self.device = config.device
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.label2id = dataset.label2id
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        
        if "GPT" in config.model:
            # 由于要加Prompt，所以最大长度要增加
            self.prompt = self.make_prompt(text="", label="")
            self.max_length = 2 * config.max_length
    
    def make_prompt(self, text, label):
        """根据text和label生成prompt

        Args:
            text (_type_): _description_
        """
        return f"你是一名中文语言学家，现在你需要对一个句子标注实体类别，###现在你需要标注的句子是：{text}。###它的答案是：{label}。"
    
    def collate_fn(self, batch):
        # 把batch中的数据按照长度排序，以便添加padding, batch: (text, anno, label)
        batch.sort(key=lambda x: len(x[0]), reverse=True)    
        texts, annos, entities, starts, ends = [], [], [], [], []
        
        for item in batch:
            text, anno, label = item
            
            # 如果有prompt，就加上prompt
            if hasattr(self, "prompt"):
                text = self.make_prompt(text, anno)
                
            texts.append(text)
            annos.append(anno)
            entities.append(label["entities"])
            starts.append(label["starts"])
            ends.append(label["ends"])
        
        # 在使用gpt2时，不加特殊字符
        if hasattr(self, "prompt"):
            texts_encoding = self.tokenizer(texts, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True, 
                                        max_length=self.max_length,
                                        add_special_tokens=False
                                        )
        else:
            texts_encoding = self.tokenizer(texts, 
                                        return_tensors='pt', 
                                        padding=True, 
                                        truncation=True, 
                                        max_length=self.max_length
                                        )

        if hasattr(self, "prompt"):
            return {
                "texts_encoding": texts_encoding.to(self.device),
                "entities": entities
            }
        
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
            try:
                for item in entities[idx]:
                    entity, char_start, char_end = item[1], item[2], item[3]
                    
                    # 通过char_start和char_end找到对应的token_start和token_end
                    token_start, token_end = encoding.char_to_token(char_start), encoding.char_to_token(char_end-1)
                    labels[idx][token_start] = self.label2id[f"B-{entity}"]
                    labels[idx][token_start+1:token_end+1] = self.label2id[f"I-{entity}"]
            except:
                breakpoint()
        
        # 把starts和ends用-100填充到同一长度，
        seq_len = texts_encoding['input_ids'].shape[1]
        for row in range(len(starts)):
            pad_len = seq_len - len(starts[row])
            if pad_len > 0:
                starts[row] += [-100] * pad_len
                ends[row] += [-100] * pad_len 
            else:
                starts[row] = starts[row][:seq_len]
                ends[row] = ends[row][:seq_len]
            if len(starts[row]) != seq_len or len(ends[row]) != seq_len:
                breakpoint()
        
        # 为global pointer制作标签：
        entity_dict = {}
        for label in self.label2id.keys():
            if label[2:] and label[2:] not in entity_dict:
                entity_dict[label[2:]] = len(entity_dict)
                
        label_for_gp = torch.zeros((len(batch), len(entity_dict), seq_len, seq_len))
        for row in range(len(entities)):
            for item in entities[row]:
                entity_type, start_idx, end_idx = item[1], item[2], item[3]
                if end_idx > seq_len:
                    continue
                
                # 注意我们在之前的处理中，start_idx和end_idx都是包含的，因此这里要减一
                label_for_gp[row][entity_dict[entity_type]][start_idx][end_idx-1] = 1
        
        item = {
            "texts_encoding": texts_encoding.to(self.device),
            "entities": entities,
            "labels": torch.Tensor(labels).to(self.device),
            "label_for_gp": torch.Tensor(label_for_gp).to(self.device),
            "starts": torch.Tensor(starts).to(self.device),
            "ends": torch.Tensor(ends).to(self.device)
        }
        return item
                
    def get_dataloader(self):
        dataloader = data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        return dataloader

# test for make_pred_idx
class Test():
    def __init__(self) -> None:
        self.label_path = "../data/label.txt"
        self.label2id, self.id2label = self.get_label_id_map(self.label_path)
    
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

if __name__ == "__main__":
    @dataclass
    class Config():
        span = 60
        data_path = "../data/train_data_public.csv"
        save_path = "../save_dir"
        label_path = "../data/label.txt"
        augmentation_level = 0
        batch_size = 32
        max_length = 64
        model_name_or_path = "../pretrained_models/bert-base-chinese"
        device = "cpu"

    config = Config()
    data_processer = DataProcessor(config)
    raw_data = data_processer.raw_data
    print(
        f"class distribution before: {data_processer.raw_data['class'].value_counts()}")
    
    data_processer.data_augmentation()
    data_processer.split_long_texts()
    print(
        f"class distribution after: {data_processer.raw_data['class'].value_counts()}")
    
    my_dataset = MyDataset(config, data_processer.raw_data)
    print(my_dataset.dataset)
    my_dataloader = MyDataLoader(config, my_dataset).get_dataloader()
    temp_data = next(iter(my_dataloader))
    test = Test()
    start_idx, end_idx = temp_data["starts"], temp_data["ends"]
    labels = temp_data["labels"]
    pred_idx = test.make_pred_idx(start_idx, end_idx)
    breakpoint()
    