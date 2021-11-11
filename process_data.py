from utils.data_preprocess import h36m_train_extract
from utils.data_preprocess import custom_data_extract
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['3dpw', '3dhp', 'h36m', 'internet'],help='process which dataset?')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'h36m':
        h36m_train_extract(config.H36M_ROOT, training_split=False, extract_img=False)
    elif args.dataset == 'internet':
        custom_data_extract('supp_assets/bilibili')
    else:
        print('Not implemented.')
