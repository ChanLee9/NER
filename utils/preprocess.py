import pandas as pd
import torch

class DataProcessor(object):
    def __init__(self, config) -> None:
        self.span = config.span
        self.raw_data = pd.read_csv(config.data_path)
        self.data = self.split_long_texts(self.raw_data)
    
    def split_long_texts(self, data, keep_ori=False, split_symbols=None):
        """_summary_
            用快慢指针来定位应该分割的位置，快指针领先一个分割符，slow < span < fast
        Args:
            keep_ori (bool, optional): 是否保留原来的句子. Defaults to False.
            split_symbols (_type_, optional): 分割符. Defaults to None.
        Return:
            data: pandas.Dataframe
        """
        drop_list = []

        if split_symbols is None:
            split_symbols = [",", ".", "?", "!", "，", "。", "？", "！"]
            
        for id, item in data.iterrows():
            item = data.iloc[id, [1, 2]]
            text, BIO_anno = item["text"], item["BIO_anno"].split()
            if len(text) < self.span:
                continue

            # 选取划分点
            split_idx = [0]
            slow = 0
            for fast in range(len(text)):
                if text[fast] in split_symbols:
                    if fast - slow < self.span:
                        split_idx[-1] = fast
                    else:
                        slow = split_idx[-1]
                        split_idx.append(slow)

            # 添加新样本
            prev_idx = 0
            for idx in split_idx:
                new_text = text[prev_idx: idx]       
                new_anno = " ".join(BIO_anno[prev_idx: idx])
                if len(new_text) < 4:
                    continue
                new_row = pd.DataFrame({
                    "id": [len(data)],
                    "text": [new_text],
                    "BIO_anno": [new_anno],
                    "class": [-1]
                })
                data = pd.concat([data, new_row], ignore_index=True)
                prev_idx = idx

            if not keep_ori:
                drop_list.append(id)

        if not keep_ori:
            data = data.drop(drop_list)
        
        return data  
    
    def split_train_test_data(self, raw_data):
        """_summary_
            根据生成的数据获取训练集和测试集
        Args:
            raw_data (_type_): pandas.Dataframe
        Return:
            train_data, val_data, test_data
        """
        pass