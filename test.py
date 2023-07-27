from utils.preprocess import DataProcessor
from dataclasses import dataclass
import torch
import pandas as pd
class Test():
    def __init__(self) -> None:
        self.label_path = "data/label.txt"
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

# test for make_pred_idx            
if __name__ == "__main__":
    import os
    print(os.getcwd())
    @dataclass
    class Config():
        span = 64
        data_path = "data/train_data_public.csv"
        save_path = "save_dir"
        augmentation_level = 0
        k_folds = 5

    config = Config()
    data_processer = DataProcessor(config)
    raw_data = data_processer.raw_data
    data_processer.data_augmentation()
    
    test = Test()
    breakpoint()
