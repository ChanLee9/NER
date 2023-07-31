# 项目背景
数据集来源：<https://www.datafountain.cn/competitions/529>

随着互联网的不断深入普及，越来越多的用户在体验企业提供的产品和服务时，会将其感受和评论分享在互联网上。这些评价和反馈信息对企业针对性地改善产品和服务有极强的指导意义，但互联网的海量信息容量让人工查找并处理评价内容的方案代价高昂。本赛题提供了一个银行业产品评价的场景，探索利用自然语言处理技术来完成评论观点的自动化提取，为行业的进一步发展提高提供参考。

# 项目结构
`.data`: 存放数据集

`.models`: 存放用到的模型

`.pretrained_models`: 存放用到的预训练模型

`.saved_results`: 存放最终实验结果，包括具体数据和可视化结果

`.utils`: 存放用到的工具函数：
  - `k_folds`: k折交叉验证
  - `preprocess`: 预处理类，数据增强
  - `training`: 训练代码
  - `utils`: DataSet和DataLoader

`main.py`: 主函数入口，各种超参数的指定

`run.sh`: 运行脚本

# 实验结果
1. 实验最终结果(test set)

  | ~ | Precision | Recall | F1 score |
  | - | - | - | - |
  | BERT_CRF | 0.8060 | 0.8055 | 0.8016 |
  | BERT_SPAN | **0.8466** | 0.8101 | 0.8110 |
  | GLOBALPOINTERS | 0.8251 | **0.8200** | **0.8199** |

2. 不同数据增强方式的影响(validation F1)

  | ~ | No Augmentation | randomly add chars | randomly delete chars | randomly replace entities | 3 types of augmentaions |
  | - | - | - | - | - | - |
  | BERT_CRF | 0.7556 | 0.7618 | 0.7628 | 0.7754 | 0.8089 |
  | BERT_SPAN | **0.7661** | **0.7809** | 0.7773 | 0.7956 | 0.8128 |
  | GLOBALPOINTERS | 0.7653 | 0.7778 | **0.7846** | **0.8119** | **0.8210** |

3. 在切分数据时，不同长度的影响(validation F1)

由于span不同的时候，使用tokenizer切分的长度不同，因此需要对max_length进行处理，使得切分的长度为32的倍数，即
$$max\_ length = ceiling(span/32)*32$$

  | ~ | SPAN=125 | SPAN=60 | SPAN=30 |
  | - | - | - | - |
  | BERT_CRF | 0.7927 | 0.8089 | 0.8016 |
  | BERT_SPAN | 0.7905 | 0.8128 | 0.8063 |
  | GLOBALPOINTERS | **0.7966** | **0.8210** | **0.8112** |
