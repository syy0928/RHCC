import torch
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from pyclustering.utils import distance_metric, type_metric
from sklearn.cluster import KMeans
import numpy as np

from src.hyptorch.pmath import _dist_matrix, _dist_matrix_pyclustering
from src.models.k_means import K_Means_hyper

"""
implementation of different hierarchical clustering methods
"""

"""
pyclustering
nltk
"""


# def hierarchical_clustering_K_Means(option, x, num_cluster):
#     """
#     input:
#     option: configurations
#     x: input data, numpy array
#     num_cluster:[100,50,20]
#     :return: clustering results
#     """
#     # number of levels
#     level = len(num_cluster)
#     data_queue = [x]
#     results = {'im2cluster': [], 'centroids': [], 'density': []}
#     for i in range(level):
#         # perform k means for every level
#         k_means = KMeans(n_clusters=num_cluster[i], max_iter=30, random_state=3407).fit(data_queue[len(data_queue) - 1])
#         results['centroids'].append(k_means.cluster_centers_)
#         results['density'].append(k_means.inertia_)
#         results['im2cluster'].append(k_means.labels_)
#         data_queue.append(k_means.cluster_centers_)
#     return results

def hierarchical_clustering_K_Means(option, x, num_cluster, hyper_c=0.):
    """
    input:
    option: configurations
    x: input data, numpy array
    num_cluster:[100,50,20]
    :return: clustering results
    """
    # number of levels
    level = len(num_cluster)#几个层级
    data_queue = [x]#所有训练数据的64位哈希码
    results = {'im2cluster': [], 'centroids': [], 'density': []}
    for i in range(level):#三个层级
        # perform k means for every level
        #print("开始初始化聚类中心")
        initial_centers = random_center_initializer(data_queue[len(data_queue) - 1], num_cluster[i]).initialize()#从数据中随机选择num_cluster[i]个数据聚类中心
        # 打印形状（簇数量 × 特征维度）
        #print("初始簇中心形状：", len(initial_centers))  # 例如 (100, 64) 表示100个簇中心，每个64维
        if hyper_c == 0.:#未经过双曲映射，即当前是欧氏空间
            #print("当前为欧氏空间")
            k_means = K_Means_hyper(data_queue[len(data_queue) - 1], initial_centers, ccore=False, itermax=30)
            k_means.process()
            """
            分配样本：计算每个样本到所有初始簇中心的欧氏距离，将样本分配给距离最近的簇。
            更新中心：对每个簇，计算簇内所有样本的均值（欧氏空间中的 “中心”），作为新的簇中心。
            判断收敛：若新中心与旧中心的差异（距离）小于某个阈值，或迭代次数达到 30 次，则停止迭代。
            """

        else:
            #双曲距离度量
            #print("当前为双曲空间")
            hyper_metric = distance_metric(type_metric.USER_DEFINED,#距离度量工具类，支持内置距离（欧氏、曼哈顿等）和用户自定义距离
                                           func=lambda x, y: _dist_matrix_pyclustering(x, y, hyper_c))#定义一个双曲距离度量函数
            k_means = K_Means_hyper(data_queue[len(data_queue) - 1], initial_centers, ccore=False, itermax=30,
                                    hyper_c=hyper_c, metric=hyper_metric)#双曲聚类，双曲 K-Means 聚类主类，重写质心更新逻辑，适配双曲空间几何特性
            k_means.process()
        results['centroids'].append(np.array(k_means.get_centers()))#返回当前层级聚类后得到的所有簇中心（每个簇的 “代表性点”）
        results['density'].append(k_means.get_total_wce())#返回当前层级聚类的总加权误差（Weighted Cluster Error），通常是 “所有样本到其所属簇中心的距离平方和”（类似 K-Means 中的损失函数）
        results['im2cluster'].append(k_means.get_clusters())#返回当前层级的 “样本 - 簇” 分配列表，格式为 [簇1样本索引, 簇2样本索引, ..., 簇k样本索引]。
        data_queue.append(k_means.get_centers())#
    # 遍历每个层级，输出对应的簇数量
    for level, centroids in enumerate(results['centroids']):
        # 每个层级的簇数量 = 簇中心数组的第一维长度
        cluster_count = centroids.shape[0]
       # print(f"第 {level + 1} 个层级的簇数量：{cluster_count}")
    return results
    #返回当前训练轮次的聚类结果，包括三个层级的聚类中心，总误差和分配关系


def hierarchical_clustering_HC():
    pass


if __name__ == '__main__':
    # test_data = np.array([[1, 1], [2, 2], [2.5, 2.3], [5.5, 6.6], [5.5, 6.1], [5.4, 6.6], [100, 101],
    #                       [100, 102], [101, 103], [120, 121], [122, 121], [123, 122],
    #                       [1000, 1001], [1001, 1002], [1002, 1003], [1010, 1011], [1010, 1012], [1011, 1012],
    #                       [1200, 1201], [1201, 1202], [1202, 1203], [1210, 1211], [1210, 1212], [1211, 1212]])
    test_data = np.random.uniform(-1, 1, (5000, 16))
    result = hierarchical_clustering_K_Means(None, test_data, [50, 20, 10], hyper_c=0.01)