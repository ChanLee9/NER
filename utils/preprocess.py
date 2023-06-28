import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from dataclasses import dataclass

class DataProcessor(object):
    def __init__(self, config) -> None:
        self.span = config.span
        self.k_folds = config.k_folds
        self.raw_data = pd.read_csv(config.data_path)
        self.data = self.split_long_texts(self.raw_data)
        self.augmentation_level = config.augmentation_level
        self.split_symbols = [",", ".", "?", "!", "，", "。", "？", "！"]

    def data_augmentation(self):
        """
        根据增强等级来确定数据增强方式：
        0 表示不增强
        1 表示随机地填充一些分隔符
        2 表示随机地删除一些字符
        3 表示随机地替换一些相同长度的实体
        """
        if self.augmentation_level == 0:
            return self.raw_data
        elif self.augmentation_level == 1:
            return self.add_sep()
        elif self.augmentation_level == 2:
            return self.del_char()
        elif self.augmentation_level == 3:
            entity_dict = self.get_entity_dict()
            return self.replace_entity(entity_dict)
        else:
            raise NotImplementedError

    def get_entity_dict(self):
        """
        获得raw_data的实体分布
        """   
         


    def split_long_texts(self, data, keep_ori=False):
        """_summary_
            用快慢指针来定位应该分割的位置，快指针领先一个分割符，slow < span < fast
        Args:
            keep_ori (bool, optional): 是否保留原来的句子. Defaults to False.
            split_symbols (_type_, optional): 分割符. Defaults to None.
        Return:
            data: pandas.Dataframe
        """
        drop_list = []

        for id, item in data.iterrows():
            item = data.iloc[id, [1, 2]]
            text, BIO_anno = item["text"], item["BIO_anno"].split()
            if len(text) < self.span:
                continue

            # 选取划分点
            split_idx = [0]
            slow = 0
            for fast in range(len(text)):
                if text[fast] in self.split_symbols:
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

    def generate_data_for_kfolds(self):
        """_summary_
            根据生成的数据获取训练集和测试集
        Args:
            raw_data (_type_): pandas.Dataframe
        Return:
            induces: k-折交叉验证的train, val index
            test: 测试集
        """
        train, test = train_test_split(self.raw_data, test_size=0.2)
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        induces = kfold.split(train)
        
        return induces, test
    
    


# if __name__ == "__main__":
#     @dataclass
#     class Config():
#         span = 64
#         data_path = "../data/train_data_public.csv"
#         k_folds = 5

#     config = Config()
#     data_processer = DataProcessor(config=config)
#     tmp = data_processer.data
#     train, val, test = data_processer.split_train_test_data(tmp)
#     for idx, item in train.iterrows():
#         print(item)
#         breakpoint()
