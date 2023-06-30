import os
import pandas as pd
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from dataclasses import dataclass

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataProcessor(object):
    def __init__(self, config) -> None:
        self.span = config.span
        self.save_path = config.save_path
        self.k_folds = config.k_folds
        self.raw_data = pd.read_csv(config.data_path)
        self.data = self.split_long_texts()
        self.augmentation_level = config.augmentation_level
        self.split_symbols = [",", ".", "?", "!", "，", "。", "？", "！"]
    
    def add_data_in_the_end(self, text, anno):
        """
        将生成的数据添加到原始数据最后，新增数据的class设为-1
        Args:
            text (_type_): 新增的text
            anno (_type_): 新增text对应的BIO_anno
        """
        new_row = pd.DataFrame({
                            "id": [len(self.raw_data)],
                            "text": [text],
                            "BIO_anno": [anno],
                            "class": [-1]
                        })
        self.raw_data = pd.concat([self.raw_data, new_row], ignore_index=True)

    def data_augmentation(self):
        """
        根据增强等级来确定数据增强方式：
        0 表示不增强
        1 表示随机地填充一些分隔符
        2 表示随机地删除一些字符
        3 表示随机地替换一些相同长度的实体
        """
        logger.info(f"data augmentation level: {self.augmentation_level}")
        
        if self.augmentation_level == 0:
            return 
        elif self.augmentation_level == 1:
            self.add_sep()
            return 
        elif self.augmentation_level == 2:
            self.del_char()
            return 
        elif self.augmentation_level == 3:
            entity_dict = self.get_entity_dict()
            self.replace_entity(entity_dict)
            return 
        else:
            raise NotImplementedError

    def add_sep(self):
        """
        以0.2的概率选取样本，然后在随机位置插入随机分隔符
        """
        logger.info("randomly adding sep...")
        
        for idx, item in self.raw_data.iterrows():
            # 以0.2的概率选取样本
            
            rand_num = random.random()
            if rand_num < 0.2 and item["class"] != -1:
                text, annotations = item["text"], item["BIO_anno"].split(" ")
                # 随机选取插入的字符
                
                sep = random.choice(self.split_symbols)
                # 我们只在某个字符后面插入，因此最多遍历到倒数第二个字符
                
                for i in range(len(text) - 1):
                    rand_num = random.random()
                    # 以0.1的概率插入一个字符在第i个字符后面
                    
                    if rand_num < 0.1:
                        new_text = text[:i] + sep + text[i:]
                        new_annotation = " ".join(annotations[:i] + "O" + annotations[i:])
                        self.add_data_in_the_end(new_text, new_annotation)
    
    def del_char(self):
        """
        以0.2的概率随机地删除一些非实体字符
        """
        logger.info("randomly deleting chars...")
        
        for idx, item in self.raw_data.iterrows():
            # 以0.2的概率选取样本
            
            rand_num = random.random()
            if rand_num < 0.2 and item["class"] != -1:
                text, annotations = item["text"], item["BIO_anno"].split(" ")
                
                for i in range(len(text)):
                    # 以0.1的概率删除第i个非实体的字符
                    
                    rand_num = random.random()
                    if rand_num < 0.1 and annotations[i] == "O":
                        new_text = text[:i] + text[i+1:]
                        new_annotation = " ".join(annotations[:i]+ annotations[i+1:])
                        self.add_data_in_the_end(new_text, new_annotation)
                   
    def get_entity_dict(self):
        """
        获得raw_data的实体分布, 并按照实体类别将其分类，如PRODUCT, COMMENT...
        然后把每一个类别的实体再根据实体长度分类
        """   
        from collections import defaultdict
        entity_dict = {}
        
        for idx, item in self.raw_data.iterrows():
            if item["class"] == -1:
                continue
            text, anno = item["text"], item["BIO_anno"].split(" ")
            i = 0
            while i < len(anno):
                # 如果第i个字符是某个实体开头，那么我们把这个实体放进实体字典中
                
                if anno[i] != "O":
                    # 去掉 "B-"
                    
                    entity = anno[i][2:]
                    j = i
                    # 这样可以确保同一个实体被分到一起，避免连续不同的实体被分到一起
                    
                    while j < len(anno) and anno[j] == f"I-{entity}":
                        j += 1
                    # 按照实体长度和实体类型将其放入到实体字典中
                    
                    if entity not in entity_dict:
                        entity_dict[entity] = defaultdict(list)
                        
                    len_dict = entity_dict[entity]
                    len_dict[j-i].append(entity)
                    # 跳过这个实体
                    
                    i = j

        # 把这个实体字典保存起来,方便以后查看
        with open(os.path.join(self.save_path, entity_dict.json), "w") as f:
            json.dump(entity_dict, ensure_ascii=False, indent=4)
        
        logger.info(f"entity dict saved in {os.path.join(self.save_path, entity_dict.json)}.")
        
        return entity_dict
    
    def replace_entity(self, entity_dict):
        """
        根据实体字典来寻找替换的实体, 以0.2的概率选取样本
        对于选取到的样本, 先寻找它的实体, 然后以0.3的概率决定是否替换
        如果需要替换该实体, 那么从实体字典中选取同类型,相同长度的实体进行替换
        如果没有相同长度的实体, 那么就选取不同长度的实体进行替换
        Args:
            entity_dict (_type_): {
                entity_type1:{
                    length=2:[xx, yy, zz],
                    length=4:[xxxx, yyyy, zzz],
                    ...
                },
                ...
            }
        """
        logger.info("replacing entities...")
        
        for idx, item in self.raw_data.iterrows():
            rand_num = random.random()
            # 以0.2的概率选取要替换的样本
            
            if rand_num < 0.2 and item["class"] != -1:
                text, anno = item["text"], item["BIO_anno"].split(" ")
                i = 0
                while i < len(anno):
                    # 寻找实体
                    
                    if anno[i][:2] == "B-":
                        # 获取当前实体
                        
                        entity = anno[i][2:]
                        j = i
                        while j < len(anno) and anno[j] == f"I-{entity}":
                            j += 1
                        # 决定是否替换当前实体
                        
                        rand_num = random.random()
                        # 只有0.3的概率会替换当前实体
                        
                        if rand_num < 0.3:
                            pass

        
    def split_long_texts(self, keep_ori=False):
        """_summary_
            用快慢指针来定位应该分割的位置，快指针领先一个分割符，slow < span < fast
        Args:
            keep_ori (bool, optional): 是否保留原来的句子. Defaults to False.
            split_symbols (_type_, optional): 分割符. Defaults to None.
        Return:
            data: pandas.Dataframe
        """
        drop_list = []

        for id, item in self.raw_data.iterrows():
            item = self.raw_data.iloc[id, [1, 2]]
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
                self.add_data_in_the_end(new_text, new_anno)
                prev_idx = idx

            if not keep_ori:
                drop_list.append(id)

        if not keep_ori:
            self.raw_data = self.raw_data.drop(drop_list)

        return self.raw_data

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
