from utils.k_folds import k_folds
from utils.preprocess import DataProcessor
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='select a model to run')
    parser.add_argument('--model_name_or_path', type=str, default='pretrained_models/gpt2', help='path to pretrained models')
    
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--merge_weight', action='store_true', help='whether to merge weights in bert layers')
    parser.add_argument('--device', type=str, required=True, default='cpu', help='which device to use')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    
    parser.add_argument('--data_path', type=str, default="data/train_data_public.csv", help='data path')
    parser.add_argument('--save_path', type=str, default="saved_results", help='where to save model weights and results')
    parser.add_argument('--label_path', type=str, default="data/label.txt", help="path to label set")
    
    parser.add_argument('--test_size', type=float, default=0.2, help='test size when split train data, should be in (0, 1)')
    parser.add_argument('--span', type=int, default=60, help='select a model to run')
    parser.add_argument('--augmentation_level', type=int, default=4, help='select a model to run')
    parser.add_argument('--use_lora', action='store_true', help='whether to use lora module')
    parser.add_argument('--label_size', type=int, default=9, help='number of labels')
    parser.add_argument('--lora_r', type=int, help='choose lora r to implement')
    parser.add_argument('--k_folds', type=int, default=5, help='choose K to split train data, -1 to not use k_folds')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    # preprocess
    data_processer = DataProcessor(args)
    data_processer.data_augmentation()
    data_processer.split_long_texts()
    data = data_processer.raw_data
    
    k_folds(args, data)
