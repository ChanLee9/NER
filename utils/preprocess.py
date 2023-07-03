import os
import json
import pandas as pd
import random

from dataclasses import dataclass

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor(object):
    """
    1. 读取原始数据
    2. 进行数据增强，获取实体字典
    3. 将长句子分割为短句子
    """

    def __init__(self, config) -> None:
        self.data_path = config.data_path
        self.raw_data = self.get_raw_data()

        self.span = config.span
        self.save_path = config.save_path

        self.augmentation_level = config.augmentation_level
        self.split_symbols = [",", ".", "?", "!", "，", "。", "？", "！"]

    def get_raw_data(self):
        """
        读取原始数据，把BIO_anno列改成列表
        """
        raw_data = pd.read_csv(self.data_path)
        raw_data["BIO_anno"] = raw_data["BIO_anno"].apply(
            lambda x: x.split(" "))
        return raw_data

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
        4 表示融合前三种增强方式
        """
        logger.info(f"data augmentation level: {self.augmentation_level}")

        if self.augmentation_level == 0:
            logger.info(
                f"data augmentation level is {self.augmentation_level}, so there won't be any augmentations...")
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
        elif self.augmentation_level == 4:
            logger.info(
                f"data augmentation level is {self.augmentation_level}, so there will be 3 different types of data augmentation...")
            self.add_sep()
            self.del_char()
            entity_dict = self.get_entity_dict()
            self.replace_entity(entity_dict)
            return
        else:
            raise NotImplementedError

    def add_sep(self):
        """
        以0.1的概率选取样本，然后在随机位置插入随机分隔符
        """
        logger.info("randomly adding sep...")

        for idx, item in self.raw_data.iterrows():
            # 以0.1的概率选取样本
            rand_num = random.random()
            if rand_num < 0.1 and item["class"] != -1:
                text, anno = item["text"], item["BIO_anno"]

                # 随机选取插入的字符
                sep = random.choice(self.split_symbols)

                # 我们只在某个字符后面插入，因此最多遍历到倒数第二个字符
                for i in range(len(text) - 1):
                    rand_num = random.random()

                    # 以2/len(text)的概率插入一个字符在第i个字符后面，并且这个字符不能是实体，否则可能会破坏原有实体信息
                    # 这样需要注意一个问题：如果这个句子里面 "O" 的占比很高，那么我们可能会对这个一个样本加很多次，这样会导致 "O" 的占比更高
                    # 因此我们需要考虑整个句子的长度，用2/len(text)来大致控制增加的样本数为2
                    if rand_num < 2/len(text) and anno[i] == "O":
                        new_text = text[:i] + sep + text[i:]
                        new_anno = anno[:i] + ["O"] + anno[i:]
                        self.add_data_in_the_end(new_text, new_anno)

    def del_char(self):
        """
        以0.1的概率随机地删除一些非实体字符
        """
        logger.info("randomly deleting chars...")

        for idx, item in self.raw_data.iterrows():
            # 以0.1的概率选取样本
            rand_num = random.random()
            if rand_num < 0.1 and item["class"] != -1:
                text, annotations = item["text"], item["BIO_anno"]
                for i in range(len(text)):

                    # 以0.1的概率删除第i个非实体的字符，和增加分割符同样的思想，我们对于每个选中的样本，我们大致增加两个新的样本
                    rand_num = random.random()
                    if rand_num < 2/len(text) and annotations[i] == "O":
                        new_text = text[:i] + text[i+1:]
                        new_anno = annotations[:i] + annotations[i+1:]
                        self.add_data_in_the_end(new_text, new_anno)

    def get_entity_dict(self):
        """
        获得raw_data的实体分布, 并按照实体类别将其分类，如PRODUCT, COMMENT...
        然后把每一个类别的实体再根据实体长度分类
        """
        logger.info("getting entity dict...")

        from collections import defaultdict
        entity_dict = {}

        for idx, item in self.raw_data.iterrows():
            if item["class"] == -1:
                continue
            text, anno = item["text"], item["BIO_anno"]
            i = 0
            while i < len(anno):
                # 如果第i个字符是某个实体开头，那么我们把这个实体放进实体字典中
                if anno[i] != "O":

                    # 去掉 "B-"
                    entity = anno[i][2:]
                    j = i + 1

                    # 这样可以确保同一个实体被分到一起，避免连续不同的实体被分到一起
                    while j < len(anno) and anno[j] == f"I-{entity}":
                        j += 1
                        
                    # 按照实体长度和实体类型将其放入到实体字典中
                    if entity not in entity_dict:
                        entity_dict[entity] = defaultdict(list)
                    len_dict = entity_dict[entity]
                    if text[i:j] not in len_dict[j-i]:
                        len_dict[j-i].append(text[i:j])

                    # 跳过这个实体
                    i = j

                i += 1

        # 把这个实体字典按照实体长度排序，然后保存起来,方便以后查看
        entity_dict = {
            entity: dict(sorted(len_entity.items(), key=lambda x: x[0]))
            for entity, len_entity in entity_dict.items()
        }
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, "entity_dict.json"), "w") as f:
            json.dump(entity_dict, f, ensure_ascii=False, indent=4)

        logger.info(
            f"entity dict saved in {os.path.join(self.save_path, 'entity_dict.json')}.")

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
                text, anno = item["text"], item["BIO_anno"]
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
                            cur_entity_dict = entity_dict[entity]
                            
                            # 如果存在和当前实体长度相同的其他同类型实体，则从中随机选取一个进行替换
                            if j-i in cur_entity_dict:
                                replaced_entity = random.choice(
                                    cur_entity_dict[j-i])

                                # 处理边界情况，j 可能是最后一个字符
                                if j == len(anno):
                                    new_text = text[:i] + \
                                    replaced_entity
                                else:
                                    new_text = text[:i] + \
                                        replaced_entity + text[j:]
                                self.add_data_in_the_end(new_text, anno)
                            else:
                                # 如果没有和当前实体长度相同的同类型实体，那么我们随机抽一个与当前实体类型相同的其他长度的实体
                                rand_len = random.choice(
                                    list(cur_entity_dict.keys()))
                                replaced_entity = random.choice(
                                    cur_entity_dict[rand_len])
                                new_text = text[:i] + \
                                    replaced_entity + text[j:]
                                new_anno = anno[:i] + [f"B-{entity}"] + \
                                    [f"I-{entity}"] * (rand_len - 1) + anno[j:]
                                self.add_data_in_the_end(new_text, new_anno)

                    i += 1

    def split_long_texts(self, keep_ori=False):
        """_summary_
            用快慢指针来定位应该分割的位置，快指针领先一个分割符，slow < span < fast
        Args:
            keep_ori (bool, optional): 是否保留原来的句子. Defaults to False.
        Return:
            data: pandas.Dataframe
        """
        logger.info("splitting long texts...")

        drop_list = []

        for id, item in self.raw_data.iterrows():
            item = self.raw_data.iloc[id, [1, 2]]
            text, BIO_anno = item["text"], item["BIO_anno"]
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
                new_anno = BIO_anno[prev_idx: idx]
                if len(new_text) < 4:
                    continue
                self.add_data_in_the_end(new_text, new_anno)
                prev_idx = idx

            if not keep_ori:
                drop_list.append(id)

        if not keep_ori:
            self.raw_data = self.raw_data.drop(drop_list)

        self.raw_data.reset_index(drop=True, inplace=True)
        return


if __name__ == "__main__":
    @dataclass
    class Config():
        span = 64
        data_path = "../data/train_data_public.csv"
        save_path = "../save_dir"
        augmentation_level = 4

    config = Config()
    data_processer = DataProcessor(config)
    raw_data = data_processer.raw_data
    data_processer.data_augmentation()
    print(
        f"class distribution before: {data_processer.raw_data['class'].value_counts()}")
    for idx, item in data_processer.raw_data.iterrows():
        if len(item["text"]) != len(item["BIO_anno"]):
            print(f"{idx}, not equal length")
        if len(item["text"]) >= 64:
            print(f"{idx}, length over 64")
    data_processer.split_long_texts()
    print(
        f"class distribution after: {data_processer.raw_data['class'].value_counts()}")
    for idx, item in data_processer.raw_data.iterrows():
        if len(item["text"]) != len(item["BIO_anno"]):
            print(f"{idx}, not equal length")
        if len(item["text"]) >= 64:
            print(f"{idx}, length over 64")
    