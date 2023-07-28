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


def get_train_val(i, config):       # 第i轮
    data = pd.read_csv(config.train_path)
    total_len = len(data)
    val_len = total_len//config.k_folds
    train_data = pd.concat([data[:i*val_len], data[(i+1)*val_len:]], join='inner')
    val_data = data[i*val_len:(i+1)*val_len]
    return train_data.reset_index(), val_data.reset_index()

def visualize_loss(losses, img_save_path, K):
    figure, axs = plt.subplots(1, K, sharey=True)
    figure.suptitle(f'Loss in {K} folds')
    for i in range(K):
        xx = range(len(losses[i]))
        axs[i].plot(xx, losses[i])
        axs[i].set_xlabel('round '+str(i+1))
    plt.savefig(img_save_path + 'Losses.jpg')

def visualize_p_r_f1(data, mode, img_save_path, K):
    figure, axs = plt.subplots(1, K, sharey=True)
    figure.suptitle(f'{mode} in {K} folds')
    for i in range(K):
        axs[i].boxplot(data[i])
        axs[i].set_xlabel('round '+str(i+1))
    plt.savefig(img_save_path+mode+'.jpg')


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
    
    start_time = time.time()
    best_f1 = 0.4
    losses, Ps, Rs, F1s = [], [], [], []
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
            if "BERT" in config.model:
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
            else:
                raise NotImplementedError
            
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
                print('saving weights...\n')
                torch.save(model.state_dict(), os.path.join(save_path, config.model + '_weights.bin'))
                
        losses.append(loss)
        
        # ---------------------------validation-----------------------------
        res, res_with_o = test_loop(config, val_dataloader, model, mode='validating')
        averaged_p = np.mean(res[0])
        averaged_r = np.mean(res[1])
        averaged_f1 = np.mean(res[2])
        
        if averaged_f1 > best_f1:
            best_f1 = averaged_f1
            print('saving weights...\n')
            
            torch.save(model.state_dict(), save_path+'_weights.bin')
        print(f'validation averaged: precision: {averaged_p},  recall: {averaged_r},  F1 score: {averaged_f1}')
        
        Ps.append(res[0])
        Rs.append(res[1])
        F1s.append(res[2])
        
        averaged[0].append(averaged_p)
        averaged[1].append(averaged_r)
        averaged[2].append(averaged_f1)
        
    
    # -------------------------------visualising-------------------------------
    visualize_loss(losses, save_path, config.k_folds)
    visualize_p_r_f1(F1s, 'F1 score', save_path, config.k_folds)
    visualize_p_r_f1(Ps, 'Precision', save_path, config.k_folds)
    visualize_p_r_f1(Rs, 'Recall', save_path, config.k_folds)
    print(f'validation final averaged: precision: {np.mean(averaged[0])}, recall: {np.mean(averaged[1])}, f1 score: {np.mean(averaged[2])}')
    

    # -------------------------------------testing----------------------------------
    test_data = pd.read_csv(config.test_path)
    # test_dataset = Dataset(config, test_data)
    # test_dataloader = Dataloader(config, test_dataset)
    model.load_state_dict(torch.load(save_path+'_weights.bin'))
    model = model.to(config.device)
    P, R, F1 = test_loop(config, test_dataloader, model, mode='testing')    
    print(f'test averaged: precision: {np.mean(P)},  recall: {np.mean(R)},  F1 score: {np.mean(F1)}')    
        
    print('Done')
    end_time = time.time()
    print(f'total time : {end_time - start_time}')
