####################这里的假负样本是根据相似度加权的################################


import sys
import torch
from torch import nn
from torch.nn import BCELoss
import torch.nn.functional as F
import numpy as np
from src.hyptorch.pmath import dist_matrix

torch.set_printoptions(
    threshold=sys.maxsize,  # 取消张量元素数量限制（再大也全打）
    precision=6,  # 小数位数和numpy对齐
    sci_mode=False,  # 禁用科学计数法
    linewidth=sys.maxsize  # 行宽无限，不换行显示...
)

import torch
import torch.nn.functional as F
import numpy as np


def false_negative_mask(center_index_level):
    if isinstance(center_index_level, np.ndarray):
        center_index_level = torch.from_numpy(center_index_level).cuda()
    bsize = center_index_level.shape[0]
    ci_expand = center_index_level.unsqueeze(1).expand(bsize, bsize)
    mask = ci_expand != ci_expand.t()
    return mask


#
# def negative_sample_similarity_weight(sim_processed, neg_mask, sigma=0.5):
#     device = sim_processed.device
#     # 你的处理后≥0相似度，直接高斯加权
#     sim_processed = torch.abs(sim_processed - 1 )
#     neg_weight = torch.exp(- (sim_processed ** 2) / (2 * sigma ** 2))
#     neg_weight = torch.where(neg_mask, neg_weight, torch.tensor(0.0, device=device))
#     return neg_weight
#
#
#
# def weighted_hierarchical_contrastive_loss(x0, x1, x0_e, x1_e, tau1, tau2, center_index, hyper_c,weight1=1, weight2=0.2, sigma=1.5):
#     device = x0.device
#     bsize = x0.shape[0]
#     level = center_index.shape[1]
#     if hyper_c > 0:
#         dist_f = lambda x, y: -dist_matrix(x, y)
#     else:
#         dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
#     dist_e = lambda x, y: x @ y.t()
#     loss_all = torch.tensor(0.0, device=device)
#     for i in range(level):
#         center_index_level = center_index[:, i]
#         mask = false_negative_mask(center_index_level)
#         eye_mask = torch.eye(bsize, device=device) * 1e9
#         logits00 = (0.9 * dist_e(x0_e, x0_e) / tau2 + 0.1 * dist_f(x0, x0) / tau1) - eye_mask
#         logits01 = (0.9 * dist_e(x0_e, x1_e) / tau2 + 0.1 * dist_f(x0, x1) / tau1)
#
#         logits = torch.cat([logits01, logits00], dim=1)          # (bsize, 2*bsize)
#         logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
#         # 正样本掩码（仅用于跨视图部分）：同一原型（包括自身）
#         pos_mask_cross = ~mask                                    # (bsize, bsize)
#         neg_mask_cross = mask                                     # (bsize, bsize)
#         # neg_mask_intra = torch.ones_like(mask)                    # 全 True
#         neg_mask_intra = mask | torch.eye(bsize, device=device).bool()
#         logits_x1 = logits[:, :bsize]   # x0 与 x1 的相似度
#         logits_x0 = logits[:, bsize:]   # x0 与 x0 的相似度
#         neg_weight_x1 = negative_sample_similarity_weight(logits_x1, neg_mask_cross, sigma=sigma)
#         neg_weight_x0 = negative_sample_similarity_weight(logits_x0, neg_mask_intra, sigma=sigma)
#         neg_weight_full = torch.cat([neg_weight_x1, neg_weight_x0], dim=1)  # (bsize, 2*bsize)
#         exp_logits = torch.exp(logits)
#         pos_sum = (exp_logits[:, :bsize] * pos_mask_cross.float()).sum(dim=1)  # (bsize,)
#         neg_sum = (exp_logits * neg_weight_full).sum(dim=1)                    # (bsize,)
#         loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
#         loss = loss.mean()
#         loss_all += (1 + i) * loss
#     return loss_all / level


def false_negative_mask(center_index_level):
    if isinstance(center_index_level, np.ndarray):
        center_index_level = torch.from_numpy(center_index_level).cuda()
    bsize = center_index_level.shape[0]
    ci_expand = center_index_level.unsqueeze(1).expand(bsize, bsize)
    mask = ci_expand != ci_expand.t()
    return mask


def negative_sample_similarity_weight(sim_matrix, neg_mask, sigma=0.5):
    device = sim_matrix.device
    neg_sim = torch.where(neg_mask, sim_matrix, torch.tensor(-1e9, device=device))
    neg_sim_min = neg_sim.min(dim=1, keepdim=True)[0]
    neg_sim_max = neg_sim.max(dim=1, keepdim=True)[0]
    neg_sim_norm = (neg_sim - neg_sim_min) / (neg_sim_max - neg_sim_min + 1e-8)
    neg_weight = torch.exp(-((1 - neg_sim_norm) ** 2) / (2 * sigma ** 2))
    neg_weight = torch.where(neg_mask, neg_weight, torch.tensor(0.0, device=device))
    return neg_weight


def weighted_hierarchical_contrastive_loss(x0, x1, x0_e, x1_e, tau1, tau2, center_index, hyper_c, weight1=1, sigma=1.5):
    device = x0.device
    bsize = x0.shape[0]
    level = center_index.shape[1]
    if hyper_c > 0:
        dist_f = lambda x, y: -dist_matrix(x, y)
    else:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    dist_e = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    loss_all = 0.0
    for i in range(level):
        center_index_level = center_index[:, i]
        mask = false_negative_mask(center_index_level)
        eye_mask = torch.eye(bsize, device=device) * 1e9
        logits00 = ((1 - weight1) * dist_e(x0_e, x0_e) / tau2 + weight1 * dist_f(x0, x0) / tau1) - eye_mask
        logits01 = ((1 - weight1) * dist_e(x0_e, x1_e) / tau2 + weight1 * dist_f(x0, x1) / tau1)
        logits = torch.cat([logits01, logits00], dim=1)  # (bsize, 2*bsize)
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        # 正样本掩码（仅用于跨视图部分）：同一原型（包括自身）
        pos_mask_cross = ~mask  # (bsize, bsize)
        neg_mask_cross = mask  # (bsize, bsize)
        neg_mask_intra = torch.ones_like(mask)  # 全 True
        logits_x1 = logits[:, :bsize]  # x0 与 x1 的相似度
        logits_x0 = logits[:, bsize:]  # x0 与 x0 的相似度
        neg_weight_x1 = negative_sample_similarity_weight(logits_x1, neg_mask_cross, sigma=sigma)
        neg_weight_x0 = negative_sample_similarity_weight(logits_x0, neg_mask_intra, sigma=sigma)
        neg_weight_full = torch.cat([neg_weight_x1, neg_weight_x0], dim=1)  # (bsize, 2*bsize)
        exp_logits = torch.exp(logits)
        pos_sum = (exp_logits[:, :bsize] * pos_mask_cross.float()).sum(dim=1)  # (bsize,)
        neg_sum = (exp_logits * neg_weight_full).sum(dim=1)  # (bsize,)
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        loss = loss.mean()
        loss_all += (1 + i) * loss
    return loss_all / level



def parse_proto(results, num_cluster):
    level = len(results['im2cluster'])
    if level == 0:
        raise ValueError("聚类结果 'im2cluster' 为空")
    num_samples = results['im2cluster'][0].shape[0]
    dim = results['centroids'][0].shape[1]
    center_index = np.zeros((num_samples, level), dtype=np.int64)
    center_index[:, 0] = results['im2cluster'][0]  # 第一层标签
    for l in range(1, level):
        prev_labels = center_index[:, l - 1]
        max_prev_label = np.max(prev_labels)
        if max_prev_label >= len(results['im2cluster'][l]):
            raise ValueError(
                f"在第 {l} 层映射时，上一层的最大标签 {max_prev_label} 超出当前层 im2cluster 的长度 {len(results['im2cluster'][l])}，请检查聚类结果是否正确")
        center_index[:, l] = results['im2cluster'][l][prev_labels]
    center_corresponding = np.zeros((num_samples, level, dim), dtype=np.float32)
    for k in range(level):
        centroids_k = results['centroids'][k]  # (num_centers_k, dim)
        center_corresponding[:, k, :] = centroids_k[center_index[:, k]]
    return center_corresponding, center_index


def contrastive_proto(z_i, z_j, center_corresponding, center_index, results, tau1=0.2, tau2=0.3, hyper_c=0):
    centers = results['centroids']
    level = len(centers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    center_corresponding = torch.tensor(center_corresponding, dtype=torch.float32).to(device)
    if hyper_c == 0:
        dist_f = lambda x, y: -dist_matrix(x, y, hyper_c)
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, hyper_c)
    loss_all = 0
    for l in range(level):
        centers_level = torch.tensor(centers[l], dtype=torch.float32).to(device)
        positive_i = torch.diag(torch.exp(dist_f(z_i, center_corresponding[:, l, :]) / tau1))
        negative_i = torch.sum(torch.exp(dist_f(z_i, centers_level) / tau1) + torch.tensor(1e-5, device=device))
        positive_j = torch.diag(torch.exp(dist_f(z_j, center_corresponding[:, l, :]) / tau1))
        negative_j = torch.sum(
            torch.exp(dist_f(z_j, centers_level) / tau1) + torch.tensor(1e-5, device=device))  # 补充：指定device
        loss_level = -torch.log(positive_i / negative_i) - torch.log(positive_j / negative_j)
        loss_all += (1 + l) * loss_level.mean()
    avg_loss = loss_all / level
    return avg_loss
