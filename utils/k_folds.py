import time
import json
import os
import logging
from importlib import import_module
from sklearn.model_selection import KFold, train_test_split
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import loralib as lora


from .training import train_loop, get_optimizer, test_loop
from .utils import MyDataLoader, MyDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_results(Ps, Rs, F1s, save_path):
    assert len(Ps) == len(Rs) == len(F1s), "length of Ps, Rs, F1s should be equal!"
    plt.figure(figsize=(10, 5))
    xx = np.arange(1, len(Ps)+1)
    plt.plot(xx, Ps, label='precision')
    plt.plot(xx, Rs, label='recall')
    plt.plot(xx, F1s, label='f1_score')
    plt.plot(xx, np.mean(F1s)*np.ones(len(F1s)), label='average f1_score', linestyle='--')
    plt.legend()
    plt.savefig(save_path)

def generate_data_for_kfolds(data, config):
    """_summary_
        根据生成的数据获取训练集和测试集
    Args:
        raw_data (_type_): pandas.Dataframe
    Return:
        induces: k-折交叉验证的train, val index
        test: 测试集
    """
    train, test = train_test_split(data, test_size=config.test_size)
    kfold = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    induces = kfold.split(train)
    
    return induces, test

def unfreeze_params(model):
    """在冻结了除lora层以外层的参数后，把lstm和crf中的参数设置为可学习
    
    Args:
        model (_type_): _description_
    """
    if hasattr(model, 'lstm'):
        for _, params in model.lstm.named_parameters():
            params.requires_grad = True
    
    if hasattr(model, 'crf'):
        for _, params in model.crf.named_parameters():
            params.requires_grad = True

def print_trainable_params(model):
    """打印模型可训练参数量占比

    Args:
        model (_type_): _description_

    """
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    trainable_ratio = trainable_params / total_params
    logger.info(f'total parameters: {total_params} || \
            trainable parameters: {trainable_params} || \
            trainable ratio: {100*trainable_ratio:.2f}%'
        )

def save_results(save_path, res, config):
    """根据当前轮次结果决定是否保存结果，如果当前更好则除了保存训练结果之外还要保存用于训练的超参数

    Args:
        save_path (_type_): _description_
        res (_type_): dict of results like {
            "average_precision": Float,
            "average_recall": Float,
            "average_f1_score": Float,
            "precision": List[float],
            "recall": List[float],
            "f1_score": List[float]
            }
        config (_type_): hyperparameters in main.py
    """
    res_save_path = os.path.join(save_path, 'results.json')
    pic_save_path = os.path.join(save_path, 'results.png')
    if os.path.exists(res_save_path):
        with open(res_save_path, 'r') as f:
            results = json.load(f)
            if "average_f1_score" in results:
                f1 = results['average_f1_score']
            else:
                f1 = 0.
                
        # 小于历史最好结果，不保存
        if f1 > res["average_f1_score"]:
            logger.info(f"current f1 score: {res['average_f1_score']:.4f} < history f1 score: {f1:.4f}, no need to save!")
        else:
            logger.info(f"current f1 score: {res['average_f1_score']:.4f} > history f1 score: {f1:.4f}, saving results...")
            with open(res_save_path, 'w') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
            # 保存超参数
            with open(os.path.join(save_path, 'config.json'), 'w') as f:
                json.dump(config.__dict__, f, ensure_ascii=False, indent=4)
            # 保存结果图
            logger.info(f"plot saved in {pic_save_path}...")              
            plot_results(res["precision"], res["recall"], res["f1_score"], pic_save_path) 
    else:
        logger.info(f"current f1 score: {res['average_f1_score']:.4f}, saving results...")
        with open(res_save_path, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        # 保存超参数
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config.__dict__, f, ensure_ascii=False, indent=4)
        # 保存结果图
        logger.info(f"plot saved in {pic_save_path}...")              
        plot_results(res["precision"], res["recall"], res["f1_score"], pic_save_path) 
    

def k_folds(config, data):
    """
    利用k折交叉验证减少方差
    Args:
        config (_type_): _description_
    """
    save_path = os.path.join(config.save_path, config.model)
    os.makedirs(save_path, exist_ok=True)
    
    induces, test_data = generate_data_for_kfolds(data, config)
    test_data.reset_index(drop=True, inplace=True)
    test_dataset = MyDataset(config, test_data)
    test_dataloader = MyDataLoader(config, test_dataset).get_dataloader()
    
    start_time = time.time()
    best_f1 = 0.
    # losses, Ps, Rs, F1s = [], [], [], []
    averaged = [[], [], []]     # p, r, f1
    for idx, item in enumerate(induces):
        logger.info(f'------------the {idx+1}th round begin, {config.k_folds} rounds in total--------------------')
        
        # -------------------------------data--------------------------------
        train_data, val_data = data.iloc[item[0]], data.iloc[item[1]]
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        
        train_dataset = MyDataset(config, train_data)
        val_dataset = MyDataset(config, val_data)
        train_dataloader = MyDataLoader(config, train_dataset).get_dataloader()
        val_dataloader = MyDataLoader(config, val_dataset).get_dataloader()
        
        # -------------------------------model--------------------------------
        model = import_module('models.' + config.model)
        model = model.Model(config).to(config.device)
        
        if config.use_lora:
            logger.info('using lora...\n')
            if "GPT" not in config.model:
                model_dim = model.bert.embeddings.word_embeddings.embedding_dim

                # 默认把lora模块应用到 kqvo 中
                for layer in model.bert.encoder.layer:
                    layer.attention.self.query = lora.Linear(model_dim, model_dim, r=config.lora_r)
                    layer.attention.self.key = lora.Linear(model_dim, model_dim, r=config.lora_r)
                    layer.attention.self.value = lora.Linear(model_dim, model_dim, r=config.lora_r)
                    layer.attention.output.dense = lora.Linear(model_dim, model_dim, r=config.lora_r)
                    
                lora.mark_only_lora_as_trainable(model)
                
                # 把 lstm和crf中的参数设置为可学习
                unfreeze_params(model)
            elif "GPT" in config.model:
                model_dim = model.gpt.wte.embedding_dim
                
                # 默认把lora模块应用到 attn.c_attn、attn.c_proj、mlp.c_fc和mlp.c_proj 中
                for layer in model.gpt.transformer.h:
                    layer.attn.c_attn = lora.Conv1d(model_dim, model_dim, kerner_size=1, r=config.lora_r)
                    layer.attn.c_proj = lora.Conv1d(model_dim, model_dim, kerner_size=1, r=config.lora_r)
                    layer.mlp.c_fc = lora.Conv1d(model_dim, model_dim, kerner_size=1, r=config.lora_r)
                    layer.mlp.c_proj = lora.Conv1d(model_dim, model_dim, kerner_size=1, r=config.lora_r)
                
                lora.mark_only_lora_as_trainable(model)
            
        # 查看模型可训练参数量
        print_trainable_params(model)
                
        if idx == 0:
            logger.info(f'-------------------------Using { config.model } model----------------------------')
        optimizer, lr_scheduler = get_optimizer(model, train_dataloader, config)
        
        # -------------------------------training----------------------------
        loss = []
        for epoch in range(config.epochs):
            total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch, config)
            loss.append(total_loss)
            res = test_loop(config, val_dataloader, model, mode='validating')
            
            if np.mean(res[2]) > best_f1:
                best_f1 = np.mean(res[2])
                logger.info('saving weights...')
                torch.save(model.state_dict(), os.path.join(save_path, 'weights.bin'))
                
        # losses.append(loss)
        
        # ---------------------------validation-----------------------------
        res = test_loop(config, val_dataloader, model, mode='validating')
        averaged_p = np.mean(res[0])
        averaged_r = np.mean(res[1])
        averaged_f1 = np.mean(res[2])
        
        if averaged_f1 > best_f1:
            best_f1 = averaged_f1
            logger.info('saving weights...')
            torch.save(model.state_dict(), os.path.join(save_path, 'weights.bin'))
        
        averaged[0].append(averaged_p)
        averaged[1].append(averaged_r)
        averaged[2].append(averaged_f1)
        
    logger.info(f"overall precision: {np.mean(averaged[0])} || overall recall: {np.mean(averaged[1])} || overall f1 score: {np.mean(averaged[2])}")
    
    # -------------------------------------testing----------------------------------
    logger.info(f"loading best model...")
    model.load_state_dict(torch.load(os.path.join(save_path, 'weights.bin')))
    model = model.to(config.device)
    
    res = test_loop(config, test_dataloader, model, mode='testing')

    results = {
        "average_precision": np.mean(res[0]),
        "average_recall": np.mean(res[1]),
        "average_f1_score": np.mean(res[2]),
        "precision": res[0],
        "recall": res[1],
        "f1_score": res[2]
    }
    
    save_results(save_path, results, config)
    
    logger.info('Done')
    end_time = time.time()
    time_consumed = end_time - start_time
    logger.info(f'total time : {time_consumed // 60}m {time_consumed % 60}s')
    
