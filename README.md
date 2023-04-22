# 项目背景
数据集来源：<https://www.datafountain.cn/competitions/529>

随着互联网的不断深入普及，越来越多的用户在体验企业提供的产品和服务时，会将其感受和评论分享在互联网上。这些评价和反馈信息对企业针对性地改善产品和服务有极强的指导意义，但互联网的海量信息容量让人工查找并处理评价内容的方案代价高昂。本赛题提供了一个银行业产品评价的场景，探索利用自然语言处理技术来完成评论观点的自动化提取，为行业的进一步发展提高提供参考。

# 项目结构
`.data`: 存放数据集和增强结果

`.models`: 存放用到的模型

`.pretrained_models`: 存放用到的预训练模型

`.transformers_lora`: 直接复制的`transformers`库，把 `.transformers_lora/models/bert/modelling_bert.py`中的线性层替换为`lora`模块

`.utils`: 存放用到的工具函数

`bad_case_analysis.ipynb`: 数据增强流程及思路

`main.py`: 主函数入口

`pre_process.ipynb`: 预处理部分，包括划分数据集，构建词表，构建标签映射等

# 工作流程
- 使用`pre_process.ipynb`生成`.data`文件中的数据
- 运行主函数`main.py`得到实验结果并保存模型权重
- 运行`bad_case_analysis.ipynb`获得增强数据集
- 再次运行主函数`main.py`得到新结果

# 实验结果
记号如下
- L: LSTM
- B: BERT
- C: CRF
- D: dynamic merging
- M: multi-head attention layer
1. 模型优化部分

  | ~ | Precision | Recall | F1 score |
  | - | - | - | - |
  | L+C | 0.7028 | 0.7290 | 0.7157 |
  | L+M+C | 0.7136 | 0.7319 | 0.7226 |
  | B+C | 0.7185 | 0.7539 | 0.7358 |
  | B+L+C | 0.7190 | 0.7653 | 0.7414 |
  | D+B+C | 0.7268 | 0.7704 | 0.7479 |
  | D+B+L+C | 0.7301 | **0.7825** | 0.7557 |
  | A(D+B+L+C) | **0.7489** | 0.7744 | **0.7615**|
2. 训练优化部分

  baseline: D_BERT_BiLSTM_CRF
  
  batch_size: 12
  
  seq_len: 256

  | ~ | 显存消耗 | 平均单个epoch耗时 | 可训练参数 | F1 score |
  | - | - | - | - | - |
  | baseline | 10751M | 2分37秒 | 353M | **0.7557** |
  | amp | 9217M | **1分50秒** | 353M | 0.7450 |
  | lora | 7451M | 2分12秒 | **3.6M** | 0.7478 |
  | lora+amp | **6829M** | 2分08秒 | **3.6M** | 0.7437 |
  
