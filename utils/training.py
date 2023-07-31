import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import get_scheduler

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
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            # y_pred的例子 ：[[1, 2, 3], [2, 3], [1]], 之所以长度不同是因为有mask的存在。长度递减是因为dataloader中按照长度给输入排序了。
            y_pred = model(item)
            if config.model == "GLOBALPOINTERS":
                res = eval_globalpointers(y_pred, item)
            else:
                res = eval(y_pred, item, config)

            Ps.append(res[0])
            Rs.append(res[1])
            F1s.append(res[2])
            
            progress_bar.update(1)
        print(f'{mode}: precision: {np.mean(Ps)} || recall: {np.mean(Rs)} || f1_score: {np.mean(F1s)}')
    return Ps, Rs, F1s

def eval_globalpointers(y_pred, item):
    """评估globalpointers模型

    Args:
        y_pred (_type_): y_pred: shape of (batchsize, entity_type_num, seq_len, seq_len)
        item (_type_): 只取y_true = item["label_for_gp"]

    Returns:
        _type_: _description_
    """
    y_true = item["label_for_gp"]
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    pred = []
    true = []
    for b, l, start, end in zip(*np.where(y_pred > 0)):
        pred.append((b, l, start, end))
    for b, l, start, end in zip(*np.where(y_true > 0)):
        true.append((b, l, start, end))

    P = set(pred)
    T = set(true)
    X = len(P & T)
    Y = len(P)
    Z = len(T)
    if Y * Z == 0:
        return 0, 0, 0
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return precision, recall, f1

def get_label_id_map(label_path):
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

def extract_entity(y_pred, config):
    """从预测的BIO序列中提取实体

    Args:
        y_pred (_type_): list of shape (batch_size, seq_len)
    """
    # 去掉 [CLS] 和 [SEP] 特殊标记
    for row in range(len(y_pred)):
        y_pred[row] = y_pred[row][1:-1]
    pred_entities = []
    label2id, id2label = get_label_id_map(config.label_path)
                
    for row in range(len(y_pred)):
        pred_entities.append([])
        i = 0
        while i < len(y_pred[row]):
            if y_pred[row][i] == 0:
                i += 1
                continue
            
            # 定位实体
            entity = id2label[y_pred[row][i]][2:]
            j = i + 1
            while j < len(y_pred[row]) and id2label[y_pred[row][j]] == f"I-{entity}":
                j += 1
            pred_entities[row].append((entity, i, j))
            i = j

    return pred_entities
    
def eval(y_pred, item, config):
    '''
    y_pred : model预测的结果
    label_ids : 真值
    我们考虑实体级别的F1值，因此我们需要先从BIO标注提取实体然后计算F1值
    '''
    entities = item["entities"]
    pred_entities = extract_entity(y_pred, config)
    true_entities = []
    
    Ps, Rs, F1s = [], [], []
    
    for row in range(len(entities)):
        true_entities.append([])
        if not entities[row]:
            continue
        for item in entities[row]:
            true_entities[row].append((item[1], item[2], item[3]))
            
    for row in range(len(pred_entities)):
        P = len(set(pred_entities[row]))
        T = len(set(true_entities[row]))
        S = len(set(pred_entities[row]) & set(true_entities[row]))
        if P * T == 0:
            continue
        Ps.append(S / P)
        Rs.append(S / T)
        F1s.append(2 * S / (P + T))     
           
    return np.mean(Ps), np.mean(Rs), np.mean(F1s)    

