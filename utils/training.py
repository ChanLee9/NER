import numpy as np
import torch
from tqdm import tqdm
from transformers import get_scheduler
from sklearn.metrics import f1_score, precision_score, recall_score

WEIGHT_DECAY = 1e-2

def get_optimizer(model, dataloader, config):
    optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=config.lr, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=100,
        num_training_steps=config.epochs*len(dataloader)
    )
    return optimizer, lr_scheduler
    
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, config):     # 一轮训练
    total_loss = 0.
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'epoch: {epoch}, loss: {0:>4f}')
    
    model.train()  
    for batch, item in enumerate(dataloader):
        loss = model.loss_fn(item)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_description(f'epoch: {epoch+1}, loss: {total_loss/(batch+1)}')
        progress_bar.update(1)
    return total_loss

def test_loop(config, dataloader, model, mode):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'{mode}... ')
    model.eval()
    Ps, Rs, F1s = [], [], []
    R_with_O, P_with_O, F1_with_O = [], [], []
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            # y_pred的例子 ：[[1, 2, 3], [2, 3], [1]], 之所以长度不同是因为有mask的存在。长度递减是因为dataloader中按照长度给输入排序了。
            y_pred = model(item)
            label_ids = item["labels"]
            res, res_with_o = eval(y_pred, label_ids)

            Ps.append(res[0])
            Rs.append(res[1])
            F1s.append(res[2])

            P_with_O.append(res_with_o[0])
            R_with_O.append(res_with_o[1])
            F1_with_O.append(res_with_o[2])
            
            progress_bar.update(1)
        print(f'without class O: precision: {np.mean(Ps)} || recall: {np.mean(Rs)} || f1_score: {np.mean(F1s)}')
        print(f'with class O: precision: {np.mean(P_with_O)} || recall: {np.mean(R_with_O)} || f1_score: {np.mean(F1_with_O)}\n')
    return (Ps, Rs, F1s), (P_with_O, R_with_O, F1_with_O)

def eval(y_pred, label_ids):
    '''
    y_pred : model预测的结果
    label_ids : 真值
    我们这里用strict F1 score， 定义见<https://www.datafountain.cn/competitions/529/datasets>
    我们考虑有实体类别 O 和没有实体类别 O 这两种情况的 p-r-f1值
    '''
    # 我们只考虑主要实体部分，如果标签是-100，我们就不考虑
    R_with_O, P_with_O, F1_with_O = [], [], []
    R, P, F1 = [], [], []
    labels = label_ids.tolist()
    for row in range(len(labels)):
        # 去掉 [CLS] 和 [SEP] 特殊标记
        labels[row] = labels[row]
        y_pred[row] = y_pred[row]
        
        # 由于有mask的存在，某些y_pred长度会小于labels长度，我们需要补齐
        while len(y_pred[row]) < len(labels[row]):
            y_pred[row] += [0]
            
        if len(labels[row]) != len(y_pred[row]):
            breakpoint()

        # 有实体类别 O 的情况
        P_with_O.append(
            precision_score(labels[row], y_pred[row], average="macro", zero_division=0)
        )
        R_with_O.append(
            recall_score(labels[row], y_pred[row], average="macro", zero_division=0)
        )
        F1_with_O.append(
            f1_score(labels[row], y_pred[row], average="macro", zero_division=0)
        )
        # 没有实体类别 O 的情况
        cur_true, cur_pred = [], []
        for i in range(len(y_pred[row])):
            # 如果标签不是 0 或者标签是 0 但预测结果不是 0 
            if labels[row][i] != 0 and y_pred[row][i] != 0:
                cur_true.append(labels[row][i])
                cur_pred.append(y_pred[row][i])
        # 如果这一行没有一个除 O 之外的实体类别，则直接跳过这一行
        if not cur_pred:
            continue

        P.append(
            precision_score(cur_true, cur_pred, average="macro", zero_division=0)
        )
        R.append(
            recall_score(cur_true, cur_pred, average="macro", zero_division=0)
        )
        F1.append(
            f1_score(cur_true, cur_pred, average="macro", zero_division=0)
        )
    
    precision = np.mean(P)
    recall = np.mean(R)
    f1score = np.mean(F1)

    precision_with_o = np.mean(P_with_O)
    recall_with_o = np.mean(R_with_O)
    f1_score_with_o = np.mean(F1_with_O)
    
    return (precision, recall, f1score), (precision_with_o, recall_with_o, f1_score_with_o)

