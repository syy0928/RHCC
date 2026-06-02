from datetime import datetime
import time
import torch.nn.functional as F
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from src.hyptorch.pmath import project, dist_matrix
from src.utils.logger import Logger

class ds:
    def __init__(self):
        self.output = []
        self.label = []
def partition_arg_topK(matrix, K, axis=0):
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

class LabelMatchs(object):
    def __init__(self, label_match_matrix):
        self.label_match_matrix = label_match_matrix
        self.all_sims = np.sum(label_match_matrix, axis=1)

def calc_label_match_matrix(database_labels, query_labels):
    return LabelMatchs(np.dot(query_labels, database_labels.T) > 0)

def batch_cosine_similarity(db, q, batch_size=200):
    """返回 db 与 q 的余弦相似度矩阵 [db_num, q_num]"""
    db_t = torch.as_tensor(db, dtype=torch.float32)
    q_t = torch.as_tensor(q, dtype=torch.float32)
    num_q = q_t.shape[0]
    sim_list = []
    for i in range(0, num_q, batch_size):
        q_batch = q_t[i:i+batch_size]
        sim_batch = F.cosine_similarity(db_t.unsqueeze(1), q_batch.unsqueeze(0), dim=2)
        sim_list.append(sim_batch)
    return torch.cat(sim_list, dim=1).cpu().numpy()

def batch_hyperbolic_similarity(db, q, c=1.0, batch_size=200):
    db_t = torch.as_tensor(db, dtype=torch.float32)
    q_t = torch.as_tensor(q, dtype=torch.float32)
    num_q = q_t.shape[0]
    sim_list = []
    for i in range(0, num_q, batch_size):
        q_batch = q_t[i:i+batch_size]
        dist = dist_matrix(db_t, q_batch, c)
        sim = -dist
        sim_list.append(sim)
    return torch.cat(sim_list, dim=1).cpu().numpy()


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def compute_similarity(strategy, db_euc, q_euc, db_hyp, q_hyp, **kwargs):

    # 基础参数
    c = kwargs.get('c', 1.0)
    beta = kwargs.get('beta', 1.0)
    batch_size = kwargs.get('batch_size', 200)
    normalize_euc = kwargs.get('normalize_euc', False)
    feature_fusion_mode = kwargs.get('feature_fusion_mode', 'weighted')
    alpha = kwargs.get('alpha', 0.8)

    # 统一转张量（安全兜底）
    def to_tensor(x):
        return torch.as_tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x

    db_euc = to_tensor(db_euc)
    q_euc = to_tensor(q_euc)
    db_hyp = to_tensor(db_hyp)
    q_hyp = to_tensor(q_hyp)

    # 欧式归一化
    if normalize_euc:
        db_euc = F.normalize(db_euc, dim=1)
        q_euc = F.normalize(q_euc, dim=1)

    # 提前计算两种核心相似度（全局复用）
    sim_euc = to_tensor(batch_cosine_similarity(db_euc, q_euc, batch_size))
    sim_hyp = to_tensor(batch_hyperbolic_similarity(db_hyp, q_hyp, c, batch_size))

    # ========================
    # 策略选择
    # ========================
    if strategy == 'euc':
        sim = sim_euc

    elif strategy == 'hyp_cos':
        sim = to_tensor(batch_cosine_similarity(db_hyp, q_hyp, batch_size))

    elif strategy == 'hyp':
        sim = sim_hyp

    elif strategy == 'adaptive_fusion':
        # 修改后的逻辑示例
        # 1. 归一化保持不变
        sim_euc_norm = (sim_euc - sim_euc.min()) / (sim_euc.max() - sim_euc.min() + 1e-8)
        sim_hyp_norm = (sim_hyp - sim_hyp.min()) / (sim_hyp.max() - sim_hyp.min() + 1e-8)

        # 2. 计算权重 (不去均值，保留样本维度)
        # 假设 sim 形状为 [Batch_Size]，我们需要堆叠后做 softmax
        weights_raw = torch.stack([sim_euc_norm / 0.1, sim_hyp_norm / 0.1], dim=1)  # [N, 2]
        weights_softmax = torch.softmax(weights_raw, dim=1)  # [N, 2]

        # 3. 融合 (广播机制自动对齐)
        # weight_euc 现在是形状 [N] 的张量，每个样本权重不同
        weight_euc = weights_softmax[:, 0]
        weight_hyp = weights_softmax[:, 1]

        sim = weight_euc * sim_euc_norm + weight_hyp * sim_hyp_norm
#
# '''
#     elif strategy == 'adaptive_fusion':
#         # 旧版自适应（全局权重）
#         sim_euc_norm = (sim_euc - sim_euc.min()) / (sim_euc.max() - sim_euc.min() + 1e-8)
#         sim_hyp_norm = (sim_hyp - sim_hyp.min()) / (sim_hyp.max() - sim_hyp.min() + 1e-8)
#         weight_euc = torch.softmax(sim_euc_norm / 0.1, dim=0).mean()
#         weight_hyp = torch.softmax(sim_hyp_norm / 0.1, dim=0).mean()
#         weight_sum = weight_euc + weight_hyp + 1e-8
#         sim = (weight_euc / weight_sum) * sim_euc_norm + (weight_hyp / weight_sum) * sim_hyp_norm
# '''
    elif strategy == 'sample_wise_confidence':
        # 1. 归一化
        sim_euc_norm = (sim_euc - sim_euc.min()) / (sim_euc.max() - sim_euc.min() + 1e-8)
        sim_hyp_norm = (sim_hyp - sim_hyp.min()) / (sim_hyp.max() - sim_hyp.min() + 1e-8)
        temperature = 3
        confidence_euc = torch.exp(sim_euc_norm * temperature)
        confidence_hyp = torch.exp(sim_hyp_norm * temperature)
        confidences = torch.stack([confidence_euc, confidence_hyp], dim=1)  # [N, 2]
        weights = torch.softmax(confidences, dim=1)  # [N, 2]
        weight_euc = weights[:, 0]  # 提取欧氏距离的权重 [N]
        weight_hyp = weights[:, 1]  # 提取超球面距离的权重 [N]
        sim = weight_euc * sim_euc_norm + weight_hyp * sim_hyp_norm


    else:
        raise ValueError(f"未知策略: {strategy}")

    return sim


# elif strategy == 'adaptive_fusion':
#     # 全局自适应融合（基于相似度均值）
#     # 1. Min-Max 归一化到 [0,1]
#     sim_euc_norm = (sim_euc - sim_euc.min()) / (sim_euc.max() - sim_euc.min() + 1e-8)
#     sim_hyp_norm = (sim_hyp - sim_hyp.min()) / (sim_hyp.max() - sim_hyp.min() + 1e-8)
#     # 2. 计算两种相似度的全局均值
#     mean_euc = sim_euc_norm.mean()
#     mean_hyp = sim_hyp_norm.mean()
#     # 3. 基于均值差异计算权重（温度参数可调，这里设为 0.5）
#     temperature = kwargs.get('temperature', 0.3)
#     w_euc = torch.exp(mean_euc / temperature) / (
#                 torch.exp(mean_euc / temperature) + torch.exp(mean_hyp / temperature))
#     w_hyp = 1 - w_euc
#     # 4. 加权融合
#     sim = w_euc * sim_euc_norm + w_hyp * sim_hyp_norm

def evaluate_with_strategy(strategy, db_euc, q_euc, db_hyp, q_hyp,
                           db_labels, test_labels, option, **kwargs):
    sim = compute_similarity(strategy, db_euc, q_euc, db_hyp, q_hyp, **kwargs)
    return mean_average_precision(
        database_hash=db_euc,
        test_hash=q_euc,
        database_labels=db_labels,
        test_labels=test_labels,
        option=option,
        sim=sim,
        ids=None
    )


###################################################评估函数#####################################################


def mean_average_precision(database_hash, test_hash, database_labels, test_labels, option, sim=None, ids=None):
    # 二值化哈希码（按需启用）
    # T = option.T
    # database_hash = np.where(database_hash < T, -1, 1)
    # test_hash = np.where(test_hash < T, -1, 1)
    # database_hash = F.normalize(database_hash, dim=1)
    # test_hash = F.normalize(test_hash, dim=1)
    num_test = test_hash.shape[0]  # 查询数量NQ
    num_database = database_hash.shape[0]  # 数据库样本总数（检索结果的最大范围）

    # 计算相似度并排序（按降序，每行对应1个查询的数据库样本排名）
    '''
    if sim is None:

        # sim = np.dot(database_hash, test_hash.T)  # 相似度矩阵[num_database, num_test]
        database_hash_tensor = torch.from_numpy(database_hash).float()
        # print(database_hash_tensor.shape)
        test_hash_tensor = torch.from_numpy(test_hash).float()
        # print(test_hash_tensor.T.shape)
        # 现在可以安全地调用 F.cosine_similarity
        # 注意：test_hash.T 的转置操作也应该在张量上进行
        sim = F.cosine_similarity(database_hash_tensor.unsqueeze(1), test_hash_tensor.unsqueeze(0), dim=2)
    '''
    if sim is None:
        # 转换为张量（保持原有逻辑）
        database_hash_tensor = torch.from_numpy(database_hash).float()
        test_hash_tensor = torch.from_numpy(test_hash).float()

        # ========== 核心修改：分批次计算 ==========
        # 设置批次大小（可根据你的内存/显存调整，越小内存占用越少）
        batch_size = 200  # 建议从100/200开始尝试，根据硬件调整
        num_test = test_hash_tensor.shape[0]  # 测试样本总数
        sim_list = []  # 存储每个批次的相似度结果

        # 循环分批次计算
        for i in range(0, num_test, batch_size):
            # 取当前批次的测试哈希
            test_batch = test_hash_tensor[i:i + batch_size]
            # 计算当前批次与整个数据库的余弦相似度
            sim_batch = F.cosine_similarity(
                database_hash_tensor.unsqueeze(1),  # [num_database, 1, dim]
                test_batch.unsqueeze(0),  # [1, batch_size, dim]
                dim=2  # 在特征维度计算余弦相似度
            )
            sim_list.append(sim_batch)
        # 拼接所有批次结果，得到完整的相似度矩阵 [num_database, num_test]
        sim = torch.cat(sim_list, dim=1)
        # ========== 分批次计算结束 ==========


    if ids is None:
        ids = np.argsort(-sim, axis=0)  # 排序索引[num_database, num_test]，越靠前相似度越高
    # save_retrieval_results_to_excel(ids, output_dir="D:/shenyuanyaun/HSSR/data/nwpu/retrieval/nwpu_0.7551_128_32.xlsx")
    del sim  # 释放内存
    # 初始化指标容器
    map_list = []
    pre_at_k = {'5': [], '10': [], '20': [], '50': []}
    recall_at_k = {'5': [], '10': [], '20': [], '50': []}
    nmrr_list = []
    k_list = [5, 10, 20, 50]

    # 预计算所有查询的NG(q)（真实相似图像数量）和GTM（最大NG(q)）
    ng_list = []
    for i in range(num_test):
        test_label = test_labels[i]
        if np.sum(test_label) == 0:  # 无效标签（无相似图像）
            ng_list.append(0)
            continue
        # 真实相似图像：数据库中与查询标签一致的样本
        relevant_mask = (np.argmax(database_labels, axis=1) == np.argmax(test_label))
        ng_q = np.sum(relevant_mask)  # 当前查询的真实相似图像数量
        ng_list.append(ng_q)
    GTM = max(ng_list) if ng_list else 0  # 所有查询中最大的NG(q)

    # 遍历每个查询计算指标
    for i in tqdm(range(num_test), desc="Calculating Metrics"):
        test_label = test_labels[i]
        if np.sum(test_label) == 0:  # 无效查询，指标均为0
            map_list.append(0.0)
            nmrr_list.append(0.0)
            for k in k_list:
                pre_at_k[str(k)].append(0.0)
                recall_at_k[str(k)].append(0.0)
            continue

        # 基础参数与标签处理
        ng_q = ng_list[i]  # 当前查询的真实相似图像数量
        query_rank = ids[:, i]  # 数据库样本按相似度排序的索引（针对当前查询）
        # 标记排序结果中每个样本是否为真实相似图像
        is_relevant = (np.argmax(database_labels[query_rank], axis=1) == np.argmax(test_label))
        total_relevant = ng_q  # 真实相似图像总数（与ng_q一致）

        # -------------------------- 原有指标计算 --------------------------
        # Precision@k和Recall@k
        for k in k_list:
            # 确保k不超过数据库样本总数（避免索引越界）
            k_valid = min(k, num_database)
            relevant_in_topk = np.sum(is_relevant[:k_valid])
            pre = relevant_in_topk / k_valid if k_valid > 0 else 0.0
            recall = relevant_in_topk / total_relevant if total_relevant > 0 else 0.0
            pre_at_k[str(k)].append(pre)
            recall_at_k[str(k)].append(recall)

        # 平均准确率AP
        cumulative_relevant = np.cumsum(is_relevant)  # 累计相关样本数
        precision_at_rank = cumulative_relevant / np.arange(1, num_database + 1)  # 每个排名的精确率
        ap = np.sum(precision_at_rank * is_relevant) / max(1, cumulative_relevant[-1])  # 避免除零
        map_list.append(ap)

        # -------------------------- ANMRR计算（修正后） --------------------------
        # 1. 计算当前查询的K值
        K = min(4 * ng_q, 2 * GTM)
        K = min(K, num_database)  # 确保不超过数据库总数

        # 2. 收集所有真实相似图像的排名（并修正超过K的rank）
        relevant_ranks = []
        for rank_idx in range(num_database):  # 遍历所有数据库样本的排名
            if is_relevant[rank_idx]:
                raw_rank = rank_idx + 1  # 排名从1开始
                # 核心修正：超过K的rank强制替换为K+1
                corrected_rank = raw_rank if raw_rank <= K else (K + 1)
                relevant_ranks.append(corrected_rank)
                # 收集满NG(q)个后停止（避免冗余）
                if len(relevant_ranks) == ng_q:
                    break

        # 3. 补充未进入前K的真实相似图像排名（文档规定：设为K+1）
        missing_count = ng_q - len(relevant_ranks)
        if missing_count > 0:
            relevant_ranks += [K + 1] * missing_count

        # 4. 计算修正检索排名MRR(q)
        sum_rank_term = np.sum([rank / ng_q for rank in relevant_ranks])
        mrr_q = sum_rank_term - 0.5 - (ng_q / 2)

        # 5. 计算归一化修正检索排名NMRR(q)
        denominator = K + 0.5 - (0.5 * ng_q)
        if denominator == 0:
            nmrr_q = 0.0
        else:
            nmrr_q = mrr_q / denominator
        nmrr_q = max(0.0, min(1.0, nmrr_q))  # 限制范围
        nmrr_list.append(nmrr_q)
    # 计算最终指标（所有查询的均值）
    final_map = np.mean(map_list)
    final_pre = [np.mean(pre_at_k[str(k)]) for k in k_list]
    final_recall = [np.mean(recall_at_k[str(k)]) for k in k_list]
    final_anmrr = np.mean(nmrr_list)

    return (final_map, *final_pre, *final_recall, final_anmrr)


def save_retrieval_results_to_excel(ids, output_dir="G:/Projects/HHCH-main/data/ucm/", top_k=50):
    """
    将检索结果写入Excel表格

    参数:
        ids: 排序索引矩阵[num_database, num_test]，每行是数据库索引，每列对应一个查询
        output_dir: 输出文件夹路径
        top_k: 保留每个查询的前k个检索结果，None表示保留全部
    """
    # 创建输出文件夹
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 获取查询数量和每个查询的检索结果数量
    num_queries = ids.shape[1]  # 列数=查询数量
    max_results = ids.shape[0]  # 行数=每个查询的检索结果总数

    # 确定需要保留的结果数量
    result_count = top_k if top_k is not None else max_results
    if result_count > max_results:
        result_count = max_results

    # 准备数据：每行是一个查询的结果
    data = []
    for query_idx in range(num_queries):
        # 获取当前查询的所有检索结果（按相似度从高到低）
        query_results = ids[:, query_idx]

        # 截取前result_count个结果
        query_results = query_results[:result_count]

        # 构造一行数据：[查询序号, 结果1, 结果2, ..., 结果k]
        row = [query_idx]  # 查询序号从1开始
        row.extend(query_results.tolist())  # 追加检索结果
        data.append(row)

    # 构造列名：第一列是"查询图像序号"，后面是"检索结果1"、"检索结果2"...
    columns = ["查询图像序号"]
    columns.extend([f"检索结果{i + 1}" for i in range(result_count)])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)

    # 生成带时间戳的文件名，避免重复
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ucm_swin_v2_b(0.8124)_321_128_32.xlsx"
    output_path = os.path.join(output_dir, filename)

    # 写入Excel
    try:
        df.to_excel(output_path, index=False)
        print(f"检索结果已成功写入Excel：{output_path}")
        return output_path
    except Exception as e:
        print(f"写入Excel失败：{str(e)}")
        return None


def get_precision_recall_by_Hamming_Radius_optimized(database_output, database_labels, query_output, query_labels,
                                                     radius=2, label_matchs=None, coarse_sign=True, fine_sign=False):
    """
    :param database:
    :param query:
    :param radius:
    :param label_match_matrix: In this optimization, we suppose the test and database lists are fixed, so we only
    calculate the test-db label matching relation once and store it in a matrix with space complexity O(db_size * test_size).
    :return:
    """
    # query_output = query.output
    # database_output = database.output
    # query_labels = query.label
    # database_labels = database.label
    # prevent impact from other measure function
    # 1. 标签清洗：把负标签（可能是异常值）设为0，避免后续计算出错
    query_labels[query_labels < 0] = 0
    database_labels[database_labels < 0] = 0
    # 2. 哈希码二值化：将实数哈希码转为±1的二值编码（汉明距离计算必须用二值码）
    bit_n = query_output.shape[1]  # i.e. K#测试集哈希码位数
    coarse_query_output = np.sign(query_output)#二值化
    coarse_database_output = np.sign(database_output)#二值化
    del query_output, database_output
    # fine_query_output = coarse_query_output if fine_sign else query_output
    # fine_database_output = coarse_database_output if fine_sign else database_output
    fine_query_output = coarse_query_output
    fine_database_output = coarse_database_output

    label_matrix_time = -1
    Logger.info("calculate match matrix")
    if label_matchs is None:
        tmp_time = time.time()
        label_matchs = calc_label_match_matrix(database_labels, query_labels)#点乘
        label_matrix_time = time.time() - tmp_time
        Logger.info("calc label matrix: time: {:.3f}\n".format(label_matrix_time))
    start_time = time.time()
    ips = np.dot(coarse_query_output, coarse_database_output.T)#二值化哈希码乘积，代表相似性
    ips = (bit_n - ips) / 2#汉明距离：不同位的个数
    #步骤1：对每个查询，按汉明距离从小到大排序（近的在前，远的在后）
    # 比如ids[0][0]是第 0 个查询的 “最近邻样本索引”
    ids = np.argsort(ips, 1)#距离排序
    end_time = time.time()
    sort_time = end_time - start_time
    Logger.info("total query: {:d}, sorting time: {:.3f}\n".format(ips.shape[0], sort_time))
    # 步骤2：统计每个查询在“汉明距离≤radius”内的样本总数
    all_nums = np.sum(ips <= radius, axis=1)
    # 步骤3：释放内存（ips已用完，删除避免占用空间）
    del ips
    precX = []  # 存储每个查询的“精度”
    recX = []  # 存储每个查询的“召回率”
    mAPX = []  # 存储每个查询的“AP（平均精度）”
    matchX = []  # 存储每个查询“半径内相似样本数”（辅助记录，非核心指标）
    allX = []  # 存储每个查询“半径内总样本数”（辅助记录，非核心指标）
    iteration = tqdm(range(coarse_query_output.shape[0]), desc="CalMAP")
    for i in iteration:
        # if i % 100 == 0:
        #     tmp_time = time.time()
        #     # print("query map {:d}, time: {:.3f}".format(i, tmp_time - end_time))
        #     end_time = tmp_time
        all_num = all_nums[i]#每个查询在“汉明距离≤radius”内的样本总数
        if all_num != 0:
            idx = ids[i, 0:all_num]#提取第一个查询的结果列表
            if fine_sign:
                imatch = label_matchs.label_match_matrix[i, idx[:]]
            else:
                ips_continue = np.dot(fine_query_output[i, :], fine_database_output[idx, :].T)
                subset_idx = np.argsort(-ips_continue, axis=0)
                idx_continue = idx[subset_idx]
                imatch = label_matchs.label_match_matrix[i, idx_continue]
            match_num = int(np.sum(imatch))
            matchX.append(match_num)
            allX.append(all_num)
            precX.append(float(match_num) / all_num)
            all_sim_num = label_matchs.all_sims[i]
            recX.append(float(match_num) / (all_sim_num + 1e-6))
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
            if match_num != 0:
                mAPX.append(np.sum(Px * imatch) / match_num)
            else:
                mAPX.append(0)
    # print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))
    # print("total time(no label matrix): {:.3f}".format(time.time() - start_time))
    if label_matrix_time > 0:
        pass
        # print("calc label matrix: time: {:.3f}".format(label_matrix_time))
    meanPrecX = 0 if len(precX) == 0 else np.mean(np.array(precX))
    meanRecX = 0 if len(recX) == 0 else np.mean(np.array(recX))
    meanMAPX = 0 if len(mAPX) == 0 else np.mean(np.array(mAPX))
    del fine_database_output, fine_query_output, label_matchs
    return meanPrecX, meanRecX, meanMAPX
