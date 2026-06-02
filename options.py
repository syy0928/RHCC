import argparse
parser = argparse.ArgumentParser(description='CHV')

# models
parser.add_argument('--model', type=str, default='swin_tiny+mona', help='vgg16,resnet,vit')
parser.add_argument('--feature_dim', type=int, default=768, help='dim of vgg output(4096,2048,768)')

# dataset
# parser.add_argument('--data_name', type=str, default='nwpu', help='cifar10 or coco or nuswide')
# parser.add_argument('--data_path', type=str, default='D:/shenyuanyaun/Dataset/NWPU-RESISC45', help='dataset path...')
# parser.add_argument('--cluster_num', default='135,90,45', type=str,help='number of clusters')
# parser.add_argument('--R', type=int, default=6300, help='MAP@R')

parser.add_argument('--data_name', type=str, default='ucm', help='cifar10 or coco or nuswide')
parser.add_argument('--data_path', type=str, default='D:/shenyuanyaun/Dataset/UCMD', help='dataset path...')
parser.add_argument('--cluster_num', default='63,42,21', type=str, help='number of clusters')
parser.add_argument('--R', type=int, default=420, help='MAP@R')

# parser.add_argument('--data_name', type=str, default='aid', help='cifar10 or coco or nuswide')
# parser.add_argument('--data_path', type=str, default='D:/Wang_SiJia/DataSet/AID_dataset/AID', help='dataset path...')
# parser.add_argument('--cluster_num', default='90,60,30', type=str,help='number of clusters')
# parser.add_argument('--R', type=int, default=2000, help='MAP@R')

# train
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=60, help='training epoch')
parser.add_argument('--use_gpu', type=bool, default=True, help="use gpu ?")
parser.add_argument('--batch_size', type=int, default=32, help='the batch size for training')
parser.add_argument('--consist', default=False, type=bool, help='softmax temperature')
parser.add_argument('--eval_epochs', type=int, default=1)
parser.add_argument('--start_eval', type=int, default=1, help="the epoch when start to test")
parser.add_argument('--gamma', type=float, default=2.0, help='gamma for Cauchy distribution')
parser.add_argument('--lambda_q', type=float, default=0.01, help='lambda to balance the quantization loss')
parser.add_argument('--workers', type=int, default=4, help='number of data loader workers.')
# parser.add_argument('--add', type=str, default="test", help='additional data')
parser.add_argument('--add', type=str, default="半径参数实验", help='additional data')
# Hashing
parser.add_argument('--hash_bit', type=int, default=128, help='hash bit,it can be 8, 16, 32, 64, 128...')
parser.add_argument('--T', type=float, default=0, help='Threshold for binary')

# Loss
parser.add_argument('--tau1', default=0.3, type=float, help='softmax temperature')
parser.add_argument('--tau2', default=0.3, type=float, help='softmax temperature')
parser.add_argument('--weight1', default=0.1, type=float)
parser.add_argument('--weight2', default=0.2, type=float)
#Loss_origin
parser.add_argument('--tau', default=0.3, type=float, help='softmax temperature')
# parser.add_argument('--tau2', default=0.3, type=float, help='softmax temperature')
# parser.add_argument('--weight1', default=0.1, type=float)

# hyperbolic
parser.add_argument('--hyper_c', type=float, default=0.1, help='balance between hyperbolic space and Euclidean space')
parser.add_argument('--clip_r', type=float, default=2.3, help='feature clip radius')
parser.add_argument('--hyper_dim', type=int, default=128, help='dimension of hyperbolic embeddings')
parser.add_argument('--WOHP', action='store_true', help='disable prototypical contrastive learning')
parser.add_argument('--WOIC', action='store_true', help='disable instance contrastive learning')
parser.add_argument('--radius_per_level', default='0.3,0.2,0.1', type=str, help='number of clusters')
parser.add_argument('--IC', action='store_true', help=' instance-wise contrastive learning without hierarchies')
parser.add_argument('--HIC', action='store_false', default=True, help=' hierarchical instance-wise contrastive learning')
parser.add_argument('--HPC', action='store_false', default=True, help=' prototypical contrastive learning')
