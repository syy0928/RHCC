import math, os, sys
import pickle
import shutil
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import platform
# DEBUG switch
from src.utils.logger import Logger

DEBUG_UTIL = False


def getDatabaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])

    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_database.pkl"

    return path


def getTrainbaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])
    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_trainbase.pkl"
    return path



def adjust_learning_rate(option, optimizer, epoch):
    """带重启的余弦退火学习率调整"""
    T_0 = 10  # 第一个周期的epoch数
    T_mult = 2  # 周期倍增因子
    eta_min = option.lr * 0.01  # 最小学习率为初始的1%

    # 计算当前周期和周期内的位置
    if epoch == 0:
        current_T = T_0
        epoch_in_cycle = 0
    else:
        # 计算当前处于哪个周期
        total_epochs = 0
        current_T = T_0
        cycle_num = 0

        while total_epochs + current_T <= epoch:
            total_epochs += current_T
            current_T *= T_mult
            cycle_num += 1

        epoch_in_cycle = epoch - total_epochs

    # 余弦退火公式
    lr = eta_min + 0.5 * (option.lr - eta_min) * (
            1 + math.cos(math.pi * epoch_in_cycle / current_T)
    )

    # 更新所有参数组的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


'''
def adjust_learning_rate(option, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every  epochs"""
    # for param_group in optimizer.param_groups:
    # param_group['lr'] = lr
    if epoch < 15:
        lr = option.lr
    else:
        lr = option.lr * (0.9 ** (epoch // 20))
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr
    # optimizer.param_groups[2]['lr'] = lr
    # optimizer.param_groups[2]['lr'] = lr

    return lr


'''

def saveStatus(option, state, epoch, MAP, result_all=None):
    # save hash center
    # print("!!!!!!!  type {}".format(hashCenter_pre.))
    # np.save('../data/' + self.option.data_name + '/centers.npy', hashCenter_pre.detach().cpu().numpy())
    if MAP >= state['best_MAP']:
        state['best_MAP'] = MAP
        state['best_epoch'] = epoch
        state['final_result'] = result_all
        # np.save(OctConv_utils.getWeightBestPath(self.option, self.state), centerWeight_train)
    elif epoch >= option.epochs - 1:
        # np.save('../data/' + self.option.data_name + '/finalweight.npy', centerWeight_train)
        pass
    else:
        pass


def calc_ham_dist_2(outputs1, outputs2, option):
    ip = torch.mm(outputs1, outputs2.t())
    mod = torch.mm((outputs1 ** 2).sum(dim=1).reshape(-1, 1), (outputs2 ** 2).sum(dim=1).reshape(1, -1))
    cos = ip / mod.sqrt()
    hash_bit = outputs1.shape[1]
    dist_ham = hash_bit / 2.0 * (1.0 - cos)

    # dist_ham = torch.where(dist_ham < 0., dist_ham + 0.01, dist_ham)

    return dist_ham


def intra_distance(hash_code, label, option):
    """
    :param hash_code: numpy array
    :param label: numpy array
    :return:
    """
    mean_all_class = []
    for i in range(label.shape[1]):
        index = np.argwhere(label[:, i] > 0.5).flatten()
        all_code = hash_code[index, :]
        intra_num = all_code.shape[0]
        center = np.mean(all_code, axis=0).reshape((1, all_code.shape[1]))
        center = center.repeat(intra_num, axis=0)
        dist = calc_ham_dist_2(torch.tensor(all_code).float(), torch.tensor(center).float(), option)[:, 0]
        # dist = dist - torch.triu(dist)
        # mean = dist.sum() / (intra_num * (intra_num - 1) / 2)
        # mean_all_class.append(mean)
        mean = torch.mean(dist)
        mean_all_class.append(mean)
    return np.mean(mean_all_class)


def inter_distance(hash_code, label, option):
    """
    :param hash_code: numpy array
    :param label: numpy array
    :return:
    """
    all_centers = []
    for i in range(label.shape[1]):
        index = np.argwhere(label[:, i] > 0.5).flatten()
        all_code = hash_code[index, :]
        # intra_num = all_code.shape[0]
        center = np.mean(all_code, axis=0)
        all_centers.append(center)

    all_centers = np.array(all_centers)
    inter_num = all_centers.shape[0]
    dist = calc_ham_dist_2(torch.tensor(all_centers).float(), torch.tensor(all_centers).float(), option)
    dist = dist - torch.triu(dist)
    mean = dist.sum() / (inter_num * (inter_num - 1) / 2)

    return mean

'''
def save_checkpoint(option, state, model_dict, is_best, filename='checkpoint.pth.tar'):
    save_model_path = '../data/' + option.data_name + '/models'
    if option.data_name is not None:
        filename_ = filename
        filename = os.path.join(save_model_path, filename_)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
    Logger.info('save models {filename}\n'.format(filename=filename))
    torch.save(model_dict, filename)
    if is_best:
        filename_best = 'model_best.pth.tar'
        if save_model_path is not None:
            filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)
        if save_model_path is not None:
            if state['filename_previous_best'] is not None and os.path.exists(state['filename_previous_best']):
                os.remove(state['filename_previous_best'])
            filename_best = os.path.join(save_model_path,
                                         'model_best_{score:.4f}.pth.tar'.format(score=model_dict['best_MAP']))
            shutil.copyfile(filename, filename_best)
            state['filename_previous_best'] = filename_best
'''
def save_checkpoint(option, state, model_dict, is_best, filename='checkpoint.pth.tar'):
    """
    修正后的保存函数：
    - 普通模型：按传入的filename保存（含data_name/model/cluster_num等）
    - 最优模型：在filename前加"best_"前缀，格式与普通模型一致
    - 自动删除旧的最优模型，避免冗余
    """
    # ------------------- 1. 构建保存路径（确保目录存在） -------------------
    # 基础保存目录：../data/{data_name}/models
    save_model_path = os.path.join('../data', option.data_name, 'models')
    os.makedirs(save_model_path, exist_ok=True)  # 自动创建目录（无需手动判断）

    # ------------------- 2. 普通模型的完整路径 -------------------
    # 拼接：save_model_path + 传入的filename（如 "cifar10_vit(10)_64.pth.tar"）
    normal_model_path = os.path.join(save_model_path, filename)

    # ------------------- 3. 保存普通模型 -------------------
    torch.save(model_dict, normal_model_path)
    Logger.info(f'✅ 普通模型已保存至：{normal_model_path}')

    # ------------------- 4. 处理最优模型（关键：自定义最优模型文件名） -------------------
    if is_best:
        # 最优模型文件名：在传入的filename前加"best_"（如 "best_cifar10_vit(10)_64.pth.tar"）
        best_filename = f"best_{filename}"
        best_model_path = os.path.join(save_model_path, best_filename)

        # ① 复制普通模型为最优模型
        shutil.copyfile(normal_model_path, best_model_path)  # 确保src和dst都传
        Logger.info(f'🏆 最优模型已保存至：{best_model_path}')

        # ② 删除旧的最优模型（避免冗余）
        if 'filename_previous_best' in state and state['filename_previous_best']:
            old_best_path = state['filename_previous_best']
            if os.path.exists(old_best_path) and old_best_path != best_model_path:
                os.remove(old_best_path)
                Logger.info(f'🗑️ 已删除旧最优模型：{old_best_path}')

        # ③ 更新state：记录当前最优模型路径（供下次删除用）
        state['filename_previous_best'] = best_model_path

    return  # 函数结束，不包含任何评估逻辑


def loadHashPool(path, type='testbase'):
    # getDatabaseHashPoolPath()
    file = open(path, 'rb')
    start = True
    if type == 'testbase':
        while True:
            try:
                data = pickle.load(file)
                hashcode_batch = data['output'].cpu()
                hashcode_batch.require_grad = False
                label_batch = data['target'].cpu()
                label_batch.require_grad = False
                if start:
                    hash_pool = hashcode_batch
                    labels = label_batch
                    start = False
                else:
                    hash_pool = torch.cat((hash_pool, hashcode_batch), dim=0)
                    labels = torch.cat((labels, label_batch), dim=0)
            except Exception:
                break
        return hash_pool, labels
    elif type == 'database':
        while True:
            try:
                data = pickle.load(file)
                hashcode_batch = data['output'].cpu()
                label_batch = data['target'].cpu()
                hashcode_batch.require_grad = False
                label_batch.require_grad = False
                # centers = data['center'].cpu()
                if start:
                    hash_pool = hashcode_batch
                    labels = label_batch
                    # centers_all = centers
                    start = False
                else:
                    hash_pool = torch.cat((hash_pool, hashcode_batch), dim=0)
                    labels = torch.cat((labels, label_batch), dim=0)
                    # centers_all = torch.cat((centers_all, centers), dim=0)
            except Exception:
                break
        return hash_pool, labels


def getTestbaseHashPoolPath(option, state):
    time = Logger.getTimeStr(state['start_time'])

    path = "../data/" + option.data_name + "/" + option.data_name + "_" + str(option.hash_bit) + "bit_" + str(
        state['epoch']) + "e_" + time + "_testbase.pkl"
    return path


if __name__ == "__main__":
    option = {}
    state = {}
    code, labels = loadHashPool(option, state,
                                "D:\python\CSQ_NEW\data\\voc\\voc_64bit_21e_[0818-15_53_42]_testbase.pkl")
    code = code.numpy()
    labels = labels.numpy()
    print()

    pass
    #     # voc_adj.pkl path
    #     dir_voc_adj = "./data/voc/voc_adj.pkl"
    #     y = gen_A(20, 0.4, str(dir_voc_adj))
    #     #print(y)
