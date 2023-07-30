python main.py \
--model BERT_CRF \
--model_name_or_path pretrained_models/bert-base-chinese \
--device cuda:0 \
--lr 2e-4 \
--augmentation_level 4 \
--epochs 5 \
--k_folds 5 \
