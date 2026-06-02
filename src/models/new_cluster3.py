import os

# 解决OpenMP重复加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 禁用多线程避免冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import torch
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.utils import distance_metric, type_metric
import numpy as np

# 添加项目根路径
sys.path.append("G:/Projects/HHCH-main")
from src.hyptorch.pmath import (
    dist as hyper_dist,
    _mobius_addition_batch,
    artanh,
    project,
    poincare_mean,
    _dist_matrix_pyclustering as pmath_dist_matrix,
    auto_select_c
)

# 格式化numpy输出
np.set_printoptions(
    threshold=sys.maxsize,
    precision=6,
    suppress=True
)

# ========== 安全转换函数 ==========
def safe_float_conversion(value, default=0.2, name="radius"):
    """安全地将输入转换为浮点数，失败时返回默认值并打印警告"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"警告: 无法将 {name}='{value}' 转换为浮点数，使用默认值 {default}")
        return default

def safe_int_conversion(value, default=10, name="num_cluster"):
    """安全地将输入转换为整数，失败时返回默认值并打印警告；若转换后非正数也返回默认值"""
    try:
        val = int(value)
        if val <= 0:
            print(f"警告: {name}='{value}' 转换为 {val}，但值必须为正，使用默认值 {default}")
            return default
        return val
    except (TypeError, ValueError):
        print(f"警告: 无法将 {name}='{value}' 转换为整数，使用默认值 {default}")
        return default

# ========== 保存聚类结果 ==========
def save_clustering_results(results, x, save_dir, epoch):
    """保存聚类结果（im2cluster/centroids/特征）"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到：{save_dir}")

    # 保存各层级im2cluster
    if 'im2cluster' in results:
        for i in range(len(results['im2cluster'])):
            im2cluster_i = results['im2cluster'][i]
            if isinstance(im2cluster_i, np.ndarray):
                im2cluster_i = im2cluster_i.tolist()
            save_path = os.path.join(save_dir, f"im2cluster_epoch_{epoch}_level_{i}.npy")
            np.save(save_path, np.array(im2cluster_i, dtype=object))
            print(f"已保存im2cluster第{i}层：{save_path}")

    # 保存各层级centroids
    if 'centroids' in results:
        for i in range(len(results['centroids'])):
            centroids_i = results['centroids'][i]
            save_path = os.path.join(save_dir, f"centroids_epoch_{epoch}_level_{i}.npy")
            np.save(save_path, centroids_i)
            print(f"已保存centroids第{i}层：{save_path}")

# ========== 生成均匀初始中心 ==========
def generate_uniform_centers(num_centers, target_radius, data_dim, hyper_c=0.0):
    """
    生成指定半径的均匀分布初始聚类中心（高维超球面）
    :param num_centers: 聚类中心数量（必须为正整数）
    :param target_radius: 目标半径（如0.1/0.2/0.3）
    :param data_dim: 特征维度（正整数）
    :param hyper_c: 双曲空间曲率
    :return: 均匀分布的初始中心 (num_centers, data_dim) - float32类型
    """
    # 强制类型转换并检查
    num_centers = int(num_centers)
    data_dim = int(data_dim)
    if num_centers <= 0:
        raise ValueError(f"num_centers 必须为正整数，但得到 {num_centers}")
    if data_dim <= 0:
        raise ValueError(f"data_dim 必须为正整数，但得到 {data_dim}")

    print(f"[DEBUG] generate_uniform_centers: num_centers={num_centers}, data_dim={data_dim}")

    np.random.seed(42)
    centers = np.random.randn(num_centers, data_dim).astype(np.float32)

    # 归一化到单位超球面
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    centers = centers / norms

    # 缩放到目标半径
    centers = centers * target_radius

    # 双曲空间约束
    if hyper_c > 0:
        max_allowed_r = 1.0 / np.sqrt(hyper_c)
        safe_max_r = max_allowed_r * 0.99
        current_max_r = np.max(np.linalg.norm(centers, axis=1))
        if current_max_r > safe_max_r:
            scale_factor = safe_max_r / current_max_r
            centers = centers * scale_factor
            print(
                f"双曲空间约束：中心缩放因子={scale_factor:.4f}，目标半径={target_radius}→实际={target_radius * scale_factor:.4f}")

    # 验证中心范数
    final_norms = np.linalg.norm(centers, axis=1)
    print(f"生成中心范数统计：均值={np.mean(final_norms):.4f}，目标={target_radius}，最大={np.max(final_norms):.4f}")

    # 双曲空间投影
    if hyper_c > 0:
        centers = project(torch.tensor(centers, dtype=torch.float32), c=hyper_c).numpy().astype(np.float32)

    return centers

# ========== K-Means 聚类类 ==========
class K_Means_hyper:
    """基于pyclustering的双曲/欧氏K-Means（输出样本→簇ID的一维数组）"""

    def __init__(self, data, centers, ccore=False, itermax=30, hyper_c=0., metric=None, target_radius=None,
                 level_idx=0):
        # 强制float32类型
        self.data = np.array(data).astype(np.float32) if isinstance(data, list) else data.astype(np.float32)
        self.centers = np.array(centers).astype(np.float32)
        self.itermax = itermax
        self.hyper_c = hyper_c
        self.metric = metric if metric else distance_metric(type_metric.EUCLIDEAN)
        self.target_radius = target_radius
        self._sample2cluster = None  # 存储样本→簇ID的一维数组
        self._cluster_sizes = None   # 存储每个簇的样本数量
        self._total_wce = 0
        self.n_samples = self.data.shape[0]  # 总样本数
        self.n_centers = self.centers.shape[0]  # 簇数量
        self.level_idx = level_idx  # 当前聚类层级索引

    def assign_clusters(self):
        """分配样本到最近中心，返回簇→样本的嵌套列表"""
        n_samples = self.n_samples
        n_centers = self.centers.shape[0]
        clusters = [[] for _ in range(n_centers)]

        # 计算距离矩阵
        if self.hyper_c > 0:
            dists = pmath_dist_matrix(self.data, self.centers, self.hyper_c)
            dists = dists.reshape(n_samples, n_centers).astype(np.float32)
        else:
            dists = self.metric(self.data, self.centers).astype(np.float32)

        # 分配样本到最近中心
        for i in range(n_samples):
            closest_center = np.argmin(dists[i])
            clusters[closest_center].append(i)

        # 处理空聚类
        for i in range(n_centers):
            if len(clusters[i]) == 0:
                random_idx = np.random.choice(n_samples)
                clusters[i] = [random_idx]

        return clusters

    def update_centers(self, clusters):
        """更新聚类中心"""
        new_centers = []
        for i, cluster in enumerate(clusters):
            cluster_points = self.data[cluster]

            if self.hyper_c == 0:
                # 欧氏均值
                center = np.mean(cluster_points, axis=0).astype(np.float32)
            else:
                # 双曲Poincaré均值
                cluster_tensor = torch.tensor(cluster_points, dtype=torch.float32)
                center = poincare_mean(cluster_tensor, dim=0, c=self.hyper_c).numpy().astype(np.float32)

            # 半径约束
            if self.target_radius is not None:
                center_norm = np.linalg.norm(center)
                if center_norm > 1e-8:
                    center = center * self.target_radius / center_norm

            # 双曲空间投影
            if self.hyper_c > 0:
                center = project(torch.tensor(center, dtype=torch.float32), c=self.hyper_c).numpy().astype(np.float32)

            new_centers.append(center)

        return np.array(new_centers).astype(np.float32)

    def calculate_wce(self, clusters):
        """计算加权聚类误差"""
        total_wce = 0
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            cluster_points = self.data[cluster]
            center = self.centers[i:i + 1]

            if self.hyper_c > 0:
                dists = pmath_dist_matrix(cluster_points, center, self.hyper_c).astype(np.float32)
            else:
                dists = self.metric(cluster_points, center).flatten().astype(np.float32)

            total_wce += np.sum(dists)
        return total_wce

    def process(self):
        """完整的K-Means迭代"""
        for _ in range(self.itermax):
            clusters = self.assign_clusters()
            current_wce = self.calculate_wce(clusters)
            new_centers = self.update_centers(clusters)
            center_diff = np.linalg.norm(new_centers - self.centers)
            if center_diff < 1e-6:
                break
            self.centers = new_centers

        # 最终生成样本→簇ID的一维数组
        final_clusters = self.assign_clusters()
        self._sample2cluster = np.zeros(self.n_samples, dtype=np.int32)
        for cluster_id, sample_indices in enumerate(final_clusters):
            self._sample2cluster[sample_indices] = cluster_id

        self._cluster_sizes = np.array([len(cluster) for cluster in final_clusters], dtype=np.int32)
        self._total_wce = self.calculate_wce(final_clusters)

        print(f"\n===== 第{self.level_idx}层各聚类中心的样本数量 =====")
        print(f"  样本数量统计：均值={np.mean(self._cluster_sizes):.2f}，最小值={np.min(self._cluster_sizes)}，最大值={np.max(self._cluster_sizes)}，标准差={np.std(self._cluster_sizes):.2f}")

    def get_centers(self):
        return self.centers

    def get_total_wce(self):
        return self._total_wce

    def get_clusters(self):
        return self._sample2cluster

    def get_cluster_sizes(self):
        return self._cluster_sizes

    def get_cluster_list(self):
        """保留原簇→样本的嵌套列表格式（若有其他逻辑依赖）"""
        clusters = self.assign_clusters()
        return np.array([np.array(c) for c in clusters], dtype=object)

def relabel_to_index(labels):
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    new_labels = np.array([mapping[x] for x in labels], dtype=np.int64)
    return new_labels, mapping



# ========== 主层级聚类函数 ==========
def hierarchical_clustering_K_Means(
        option,
        x,
        num_cluster,
        hyper_c=0.,
        radius_per_level=[0.3, 0.2, 0.1]
):
    """
    层级聚类（安全处理输入类型）
    :param option: 配置对象（未使用，保留接口）
    :param x: 特征数据 (n_samples, n_features)
    :param num_cluster: 每层聚类数量列表（可能为字符串、列表或逗号分隔的字符串）
    :param hyper_c: 双曲曲率
    :param radius_per_level: 每层目标半径列表（可能为字符串、列表或逗号分隔的字符串）
    :return: 包含 'im2cluster', 'centroids', 'density', 'cluster_sizes' 的字典
    """
    import numpy as np
    import os
    save_dir = "D:/shenyuanyaun/HSSR/cluster_results"
    os.makedirs(save_dir, exist_ok=True)

    # ========== 防御性解析 num_cluster ==========
    print("===== DEBUG: hierarchical_clustering_K_Means =====")
    print(f"Raw num_cluster: {num_cluster}, type: {type(num_cluster)}")
    print(f"Raw radius_per_level: {radius_per_level}, type: {type(radius_per_level)}")

    # 处理 num_cluster
    if isinstance(num_cluster, str):
        # 如果是一个字符串，按逗号分割
        cluster_list = [int(x.strip()) for x in num_cluster.split(',') if x.strip()]
    elif isinstance(num_cluster, (list, tuple)):
        # 如果是列表，检查是否包含逗号等异常情况
        # 常见错误：列表中的元素可能是单个字符，例如 ['1','0','0',',','6','0',',','3','0']
        # 尝试将其合并为一个字符串再解析
        if all(isinstance(n, str) for n in num_cluster):
            # 将所有字符串拼接起来，然后按逗号分割
            combined = ''.join(num_cluster)
            cluster_list = [int(x.strip()) for x in combined.split(',') if x.strip()]
        else:
            # 已经是数值列表或混合类型，尝试转换为整数
            cluster_list = []
            for n in num_cluster:
                try:
                    cluster_list.append(int(n))
                except (ValueError, TypeError):
                    print(f"警告: 无法将 {n} 转换为整数，跳过")
            if not cluster_list:
                cluster_list = [10]  # 默认
    else:
        cluster_list = [10]
        print(f"警告: 无法识别的 num_cluster 类型 {type(num_cluster)}，使用默认值 [10]")

    print(f"Parsed num_cluster: {cluster_list}")

    # 处理 radius_per_level
    if isinstance(radius_per_level, str):
        radius_list = [float(x.strip()) for x in radius_per_level.split(',') if x.strip()]
    elif isinstance(radius_per_level, (list, tuple)):
        if all(isinstance(r, str) for r in radius_per_level):
            combined = ''.join(radius_per_level)
            radius_list = [float(x.strip()) for x in combined.split(',') if x.strip()]
        else:
            radius_list = []
            for r in radius_per_level:
                try:
                    radius_list.append(float(r))
                except (ValueError, TypeError):
                    print(f"警告: 无法将 {r} 转换为浮点数，使用默认值 0.2")
                    radius_list.append(0.2)
            if not radius_list:
                radius_list = [0.2]
    else:
        radius_list = [0.2, 0.2, 0.2]
        print(f"警告: 无法识别的 radius_per_level 类型 {type(radius_per_level)}，使用默认值 [0.2,0.2,0.2]")

    # 确保两个列表长度一致，取最小长度
    min_len = min(len(cluster_list), len(radius_list))
    if min_len == 0:
        print("错误: 解析后的 num_cluster 或 radius_per_level 为空，使用默认值")
        cluster_list = [100, 60, 30]
        radius_list = [0.9, 0.5, 0.3]
        min_len = 3

    # 截取前 min_len 个
    num_cluster = cluster_list[:min_len]
    radius_per_level = radius_list[:min_len]
    level = len(num_cluster)

    print(f"Final num_cluster: {num_cluster}")
    print(f"Final radius_per_level: {radius_per_level}")
    print(f"Number of levels: {level}")

    # 以下为原有代码，但使用解析后的 num_cluster 和 radius_per_level
    x = np.array(x).astype(np.float32) if isinstance(x, list) else x.astype(np.float32)
    data_queue = [x]
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'cluster_sizes': []}

    for i in range(level):
        # 安全转换参数（此时应该是数值，但为了保险仍使用安全函数）
        current_num_cluster = safe_int_conversion(num_cluster[i], default=10, name=f"num_cluster[{i}]")
        current_radius = safe_float_conversion(radius_per_level[i], default=0.2, name=f"radius_per_level[{i}]")
        print(f"[DEBUG] level {i}: num_cluster={current_num_cluster}, radius={current_radius}")

        if current_num_cluster <= 0:
            current_num_cluster = 10
            print(f"强制将 num_cluster[{i}] 设置为 {current_num_cluster}")

        current_data = data_queue[-1]
        current_data = np.array(current_data).astype(np.float32) if isinstance(current_data, list) else current_data.astype(np.float32)
        data_dim = int(current_data.shape[1])

        initial_centers = generate_uniform_centers(
            num_centers=current_num_cluster,
            target_radius=current_radius,
            data_dim=data_dim,
            hyper_c=hyper_c
        )

        if hyper_c == 0.:
            print("当前为欧氏空间")
            k_means = K_Means_hyper(
                current_data,
                initial_centers,
                ccore=False,
                itermax=30,
                target_radius=current_radius,
                level_idx=i
            )
            k_means.process()
        else:
            hyper_metric = distance_metric(
                type_metric.USER_DEFINED,
                func=lambda x_np, y_np: pmath_dist_matrix(x_np.astype(np.float32), y_np.astype(np.float32), hyper_c)
            )
            k_means = K_Means_hyper(
                current_data,
                initial_centers,
                ccore=False,
                itermax=30,
                hyper_c=hyper_c,
                metric=hyper_metric,
                target_radius=current_radius,
                level_idx=i
            )
            k_means.process()

        centroids = np.array(k_means.get_centers()).astype(np.float32)
        results['centroids'].append(centroids)
        results['density'].append(k_means.get_total_wce())
        results['im2cluster'].append(k_means.get_clusters())
        results['cluster_sizes'].append(k_means.get_cluster_sizes())

        next_data = np.array(k_means.get_centers()).astype(np.float32) if isinstance(k_means.get_centers(), list) else k_means.get_centers().astype(np.float32)
        data_queue.append(next_data)

    return results

# ========== 辅助函数 ==========
def get_list_shape(lst):
    """获取嵌套列表的形状"""
    shape = []
    current = lst
    while isinstance(current, list) and len(current) > 0:
        shape.append(len(current))
        current = current[0]
    return tuple(shape)

def hierarchical_clustering_HC():
    pass