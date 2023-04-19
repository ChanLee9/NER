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
    if config.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    model.train()  
    iter_num = 3    # 梯度累计的更新频率
    for batch, (text_encoding, label_ids) in enumerate(dataloader):
        input_ids = text_encoding['input_ids']
        mask = text_encoding['attention_mask']        
        if config.use_amp:
            # 把计算loss过程中可以转化成混合精度的自动转换
            with torch.cuda.amp.autocast():
                loss = model.loss_fn(input_ids, label_ids, mask)
                # scaler的作用是防止梯度过小，造成下溢出，导致无法更新参数。因此我们先放大再进行后向传播
                scaler.scale(loss).backward()
                # 梯度累计
                if config.use_grad_accumulat:
                    if (batch+1)%iter_num == 0:
                        scaler.step(optimizer)
                        lr_scheduler.step()
                        scaler.update()
                else:
                    scaler.step(optimizer)
                    lr_scheduler.step()
                    scaler.update()
                    
        else:
            loss = model.loss_fn(input_ids, label_ids, mask)
            loss.backward()
            if config.use_grad_accumulat:
                if (batch+1)%iter_num == 0:
                    optimizer.step()
                    lr_scheduler.step()
            else:
                optimizer.step()
                lr_scheduler.step()
            
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_description(f'epoch: {epoch+1}, loss: {total_loss/(batch+1)}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode):
    assert mode in ['validat', 'test'], 'mode must be validation or test!'
    print(f'-------------------------{mode}ing----------------------------')
    model.eval()
    P, R, F1 = [], [], []
    with torch.no_grad():
        for batch, (text_encoding, label_ids) in enumerate(dataloader):
            input_ids = text_encoding['input_ids']
            mask = text_encoding['attention_mask'] 
            # y_pred的例子 ：[[1, 2, 3], [2, 3], [1]], 之所以长度不同是因为有mask的存在。长度递减是因为dataloader中按照长度给输入排序了。
            y_pred = model(input_ids, label_ids, mask)
            p, r, f1 = eval(y_pred, label_ids)
            P.append(p)
            R.append(r)
            F1.append(f1)
        print(f'precision: {np.mean(P)}, recall: {np.mean(R)}, f1_score: {np.mean(F1)}')
    return P, R, F1

def eval(y_pred, label_ids):
    '''
    y_pred : model预测的结果
    label_num : 真值
    我们这里用strict F1 score， 定义见<https://www.datafountain.cn/competitions/529/datasets>
    '''
    # 先把y_pred填充成label_num的大小：缺的补0(BERT_PAD_ID=0)
    max_len = len(label_ids[0])
    for row in y_pred:
        if len(row) < max_len:
            row += [0]*(max_len-len(row))
    y_pred = torch.tensor(y_pred)  
    assert y_pred.shape == label_ids.shape, 'wrong dimension!'
    y_pred = y_pred.flatten().cpu()
    label_ids = label_ids.flatten().cpu()
    R = recall_score(label_ids, y_pred, average='macro', zero_division=0)
    P = precision_score(label_ids, y_pred, average='macro', zero_division=0)
    F1 = f1_score(label_ids, y_pred, average='macro', zero_division=0)
    
    return P, R, F1
