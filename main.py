from utils.k_folds import k_folds
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='select a model to run')
    parser.add_argument('--device', type=str, required=True, default='cpu', help='which device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataloader')
    parser.add_argument('--use_lora', type=int, help='whether to use lora module')
    parser.add_argument('--use_amp', type=int, help='whether to use amp to acclerate training')
    parser.add_argument('--use_grad_accumulat', type=int, help='whether to use gradient accumulate')
    parser.add_argument('--pretrained_model_path', type=str, default='pretrained_models/bert-large-chinese', help='path to pretrained models')
    parser.add_argument('--label_size', type=int, default=9, help='number of labels')
    parser.add_argument('--max_length', type=int, default=256, help='max length to tokenize sentences')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size in lstm model')
    parser.add_argument('--pad_id', type=int, default=0, help='word pad id')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--train_path', type=str, default='data/train.csv', help='default: data/train.csv, can be changed to data/augmented.csv')
    parser.add_argument('----de', type=str, default='data/test.csv')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--lora_r', type=int, help='choose lora r to implement')
    parser.add_argument('--k_folds', type=int, default=5, help='choose K to split train data, -1 to not use k_folds')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.k_folds == -1:
        pass
    k_folds(args)
