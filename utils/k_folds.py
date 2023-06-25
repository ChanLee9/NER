from .training import train_loop, get_optimizer, test_loop
from .utils import Dataloader, Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import loralib as lora
from importlib import import_module

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
    for _, params in model.lstm.named_parameters():
        params.requires_grad = True
    
    for _, params in model.crf.named_parameters():
        params.requires_grad = True

def get_trainable_params(model):
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return trainable_params

def k_folds(config):
    if 'augmented' in config.train_path:
        # 增强后的训练集
        save_path = os.path.join('data', config.model+'augmented')
    else:
        save_path = os.path.join('data', config.model)
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    best_f1 = 0.
    losses, Ps, Rs, F1s = [], [], [], []
    averaged = [[], [], []]     # p, r, f1
    for i in range(config.k_folds):
        print(f'------------the {i+1}th round begin, {config.k_folds} rounds in total--------------------')
        train_data, val_data = get_train_val(i, config)
        model = import_module('models.' + config.model)
        model = model.Model(config).to(config.device)
        train_dataset = Dataset(config, train_data)
        val_dataset = Dataset(config, val_data)
        train_dataloader = Dataloader(config, train_dataset).dataloader
        val_dataloader = Dataloader(config, val_dataset).dataloader
        if config.use_lora and "LORA" in config.model:
            print('using lora...\n')
            lora.mark_only_lora_as_trainable(model)
            # 把 lstm和crf中的参数设置为可学习
            unfreeze_params(model)
        if config.use_amp:
            print('using amp...\n')
        if config.use_grad_accumulat:
            print('using gradient accumulating...\n')
            
        # 查看模型可训练参数量
        trainable_params = get_trainable_params(model)
        print(f'trainable parameters: {trainable_params}')      
                
        if i == 0:
            print(f'-------------------------Using { config.model } model----------------------------')
        optimizer, lr_scheduler = get_optimizer(model, train_dataloader, config)
        
        # -------------------------------training----------------------------
        loss = []
        for epoch in range(config.epochs):
            total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch, config)
            loss.append(total_loss)
            P, R, F1 = test_loop(config, val_dataloader, model, mode='validat')
            
            if np.mean(F1) > best_f1:
                best_f1 = np.mean(F1)
                print('saving weights...\n')
                torch.save(model.state_dict(), os.path.join(save_path, config.model + '_weights.bin'))
                
        losses.append(loss)
        
        # ---------------------------validation-----------------------------
        P, R, F1 = test_loop(config, val_dataloader, model, mode='validat')
        averaged_p = np.mean(P)
        averaged_r = np.mean(R)
        averaged_f1 = np.mean(F1)
        
        if averaged_f1 > best_f1:
            best_f1 = averaged_f1
            print('saving weights...\n')
            
            torch.save(model.state_dict(), save_path+'_weights.bin')
        print(f'validation averaged: precision: {averaged_p},  recall: {averaged_r},  F1 score: {averaged_f1}')
        
        F1s.append(F1)
        Ps.append(P)
        Rs.append(R)
        averaged[0].append(averaged_p)
        averaged[1].append(averaged_r)
        averaged[2].append(averaged_f1)
        
    
    # -------------------------------visualising-------------------------------
    visualize_loss(losses, save_path)
    visualize_p_r_f1(F1s, 'F1 score', save_path)
    visualize_p_r_f1(Ps, 'Precision', save_path)
    visualize_p_r_f1(Rs, 'Recall', save_path)
    print(f'validation final averaged: precision: {np.mean(averaged[0])}, recall: {np.mean(averaged[1])}, f1 score: {np.mean(averaged[2])}')
    

    # -------------------------------------testing----------------------------------
    test_data = pd.read_csv(config.test_path)
    test_dataset = Dataset(config, test_data)
    test_dataloader = Dataloader(config, test_dataset)
    model.load_state_dict(torch.load(save_path+'_weights.bin'))
    model = model.to(config.device)
    P, R, F1 = test_loop(config, test_dataloader, model, mode='test')    
    print(f'test averaged: precision: {np.mean(P)},  recall: {np.mean(R)},  F1 score: {np.mean(F1)}')    
        
    print('Done')
    end_time = time.time()
    print(f'total time : {end_time - start_time}')
