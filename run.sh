python main.py \
--model BERT_SPAN \
--model_name_or_path pretrained_models/bert-base-chinese \
--device cuda:0 \
--lr 2e-4 \
--augmentation_level 0 \
--epochs 2 \
--k_folds 2 \
