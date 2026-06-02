import datetime
import os
import sys
import random

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings

from src.Loss.Loss import parse_proto, contrastive_proto, weighted_hierarchical_contrastive_loss

warnings.filterwarnings("ignore")
from src.dataLoader.DataSet_loader import getDataLoader
from src.models.new_cluster3 import hierarchical_clustering_K_Means, save_clustering_results
from src.models.extra_model.SwinTransformer_mona import MainModel
from src.options import parser
from src.utils.logger import Logger
from src.utils.measure_utils import mean_average_precision, evaluate_with_strategy
from src.utils.util import adjust_learning_rate, saveStatus, save_checkpoint

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.autograd.set_detect_anomaly(True)
option = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(
    threshold=sys.maxsize,
    precision=6,
    suppress=True
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_features(option, train_loader, model, epoch):
    Logger.info("computing feature")
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = len(train_loader.dataset)
    feat_dim = option.hyper_dim
    features = torch.zeros(num_samples, feat_dim).to(device)
    # 存储整数标签
    labels = torch.zeros(num_samples, dtype=torch.long).to(device)

    for i, ((feature_1, feature_2, feature_origin), target, index) in enumerate(tqdm(train_loader)):
        with torch.no_grad():
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_origin = feature_origin.to(device)
            target = target.to(device)
            feat, feat_h, p_1, p_2, h_1, h_2 = model((feature_1, feature_2, feature_origin), Train=True)
            features[index] = feat
            if target.dim() == 1 or target.size(1) == 1:
                labels[index] = target.view(-1)
            else:
                labels[index] = target.argmax(dim=1)

    if epoch in [1, 5, 10, 15, 20, 25, 30]:
        p_3_np = features.detach().cpu().numpy()
        # np.save(f"G:/Projects/HHCH-main/src/cluster/New_Cluster/feature_1680_{epoch}", p_3_np)
    return features.detach(), labels.detach().cpu().numpy()


def main(option, state):
    tsne_dir = f"D:/shenyuanyaun/HSSR/data/{option.data_name}/tsne"
    train_loader, test_loader = getDataLoader(option)
    model = MainModel(option)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 修改优化器
    optimizer = torch.optim.Adam([
        {'params': model.parameters()}],
        lr=option.lr
    )

    model.to(device)

    cluster_num_str = "_".join(map(str, option.cluster_num))
    radius = "_".join(map(str, option.radius_per_level))
    excel_dir = f"D:/shenyuanyaun/HSSR/data/{option.data_name}/results"
    os.makedirs(excel_dir, exist_ok=True)
    excel_filename = (
        f"{option.add}_"
        f"tau1_{option.tau1}_tau2_{option.tau2}_"
        f"w1_{option.weight1}_w2_{option.weight2}_{option.data_name}_{option.model}_"
        f"({cluster_num_str})_({radius})_{option.hash_bit}bit_{option.batch_size}.xlsx"
    )
    excel_full_path = os.path.join(excel_dir, excel_filename)
    batch_loss_excel = f"{option.add}_{option.data_name}_batch_loss_{cluster_num_str}_({radius})_{option.hash_bit}bit_{option.batch_size}.xlsx"
    batch_loss_excel_path = os.path.join(excel_dir, batch_loss_excel)

    excel_data = []
    checkpoint_path = ""  # 检查点路径（若有）

    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_hash_dict'])
        start_epoch = checkpoint['epoch'] + 1
        state.update({
            'best_MAP': checkpoint['best_MAP'],
            'best_epoch': checkpoint['epoch'],
            'final_result': checkpoint.get('final_result', None),
            'filename_previous_best': checkpoint.get('filename_previous_best', None),
            'iter': checkpoint['epoch'] * len(train_loader)
        })
    else:
        start_epoch = 1
        state.update({
            'best_MAP': 0.0,
            'best_epoch': 0,
            'final_result': None,
            'filename_previous_best': None,
            'iter': 0
        })

    # 训练循环
    all_batch_losses = []
    for epoch in range(start_epoch, option.epochs + 1):
        lr = adjust_learning_rate(option, optimizer, epoch)
        state['epoch'] = epoch
        Logger.divider(f"Epoch[{epoch}]-lr{lr}")

        # 计算特征并聚类
        if option.IC:
            cluster_results = None
        else:
            all_feat, all_labels = compute_features(option, train_loader, model, epoch)
            all_feat_np = all_feat.cpu().numpy().astype(np.float32)
            cluster_results = hierarchical_clustering_K_Means(
                option, x=all_feat_np,
                num_cluster=option.cluster_num,
                hyper_c=option.hyper_c,
                radius_per_level=option.radius_per_level,
            )
            plot_tsne(
                all_feat_np,
                all_labels,
                f"{tsne_dir}/epoch_{epoch}_gt.png",
                title="t-SNE (Ground Truth)"
            )
            from sklearn.metrics import normalized_mutual_info_score
            def get_top_level_labels(results):
                im2cluster = results['im2cluster']
                num_levels = len(im2cluster)
                if num_levels == 1:
                    return im2cluster[0]
                labels = im2cluster[0]
                for level in range(1, num_levels):
                    labels = im2cluster[level][labels]
                return labels

            top_labels = get_top_level_labels(cluster_results)
            nmi_score = normalized_mutual_info_score(all_labels, top_labels)
            Logger.info(f"Epoch {epoch} 顶层聚类 NMI: {nmi_score:.4f}")
        torch.cuda.empty_cache()
        loss_epoch, batch_losses = train_step(model, optimizer, train_loader, cluster_results, epoch, state, option)
        # all_batch_losses.extend(batch_losses)
        # pd.DataFrame(all_batch_losses).to_excel(batch_loss_excel_path, index=False)
        # Logger.info(f"Epoch[{epoch}] Batch损失已保存至：{batch_loss_excel}")
        Logger.info(f"epoch: {epoch} Loss: {loss_epoch}")
        current_data = {'epoch': epoch, 'lr': lr, 'loss': loss_epoch, 'is_best': None,
                        'nmi': nmi_score if not option.IC else None}
        if epoch % option.eval_epochs == 0 and epoch >= option.start_eval:
            MAP_Rank, pre5, pre10, pre20, pre50, rec5, rec10, rec20, rec50, ANMRR = test_step(
                option, state, model, test_loader, epoch, train=False)
            Logger.info(f"\n====== Epoch {epoch} 测试集评估结果 ======")
            Logger.info(f"MAP_Rank: {MAP_Rank:.4f}, P@5: {pre5:.4f}, P@10: {pre10:.4f}")
            Logger.info(f"P@20: {pre20:.4f}, P@50: {pre50:.4f}, ANMRR: {ANMRR:.4f}")
            Logger.info(f"R@5: {rec5:.4f}, R@10: {rec10:.4f}, R@20: {rec20:.4f}, R@50: {rec50:.4f}")
            saveStatus(option, state, epoch, MAP_Rank,
                       (MAP_Rank, pre5, pre10, pre20, pre50, rec5, rec10, rec20, rec50, ANMRR))
            is_best = MAP_Rank >= state.get('best_MAP', 0.0)
            if is_best:
                state['best_MAP'] = MAP_Rank
                state['best_epoch'] = epoch
            Logger.info(
                f"当前MAP: {MAP_Rank:.4f} | 最优MAP: {state['best_MAP']:.4f} | 是否最优: {is_best} | best_epoch: {state['best_epoch']}")
            current_data.update({
                'MAP_Rank': MAP_Rank, 'P@5': pre5, 'P@10': pre10,
                'P@20': pre20, 'P@50': pre50,
                'R@5': rec5, 'R@10': rec10, 'R@20': rec20, 'R@50': rec50, 'ANMRR': ANMRR
            })
            excel_data.append(current_data)
            pd.DataFrame(excel_data).to_excel(excel_full_path, index=False)
            Logger.info(f"评估数据已保存至Excel：{excel_filename}")
            # 保存模型
            # model_dict = {
            #     'epoch': epoch,
            #     'model_dict': model.state_dict(),
            #     'optimizer_hash_dict': optimizer.state_dict(),
            #     'best_MAP': state['best_MAP'],
            #     'P@5': pre5, 'P@10': pre10, 'R@5': rec5, 'R@10': rec10, 'ANMRR': ANMRR
            # }
            # if epoch == option.epochs:
            #     filename = f"{option.add}_{option.data_name}_c={option.hyper_c}_{option.model}_({option.cluster_num})_({option.radius_per_level})_{option.hash_bit}bit_{option.batch_size}_MAP_{state['best_MAP']:.4f}.pth.tar"
            # else:
            #     filename = f"{option.add}_{option.data_name}_c={option.hyper_c}_{option.model}_({option.cluster_num})_({option.radius_per_level})_{option.hash_bit}bit_{option.batch_size}_MAP_{state['best_MAP']:.4f}.pth.tar"
            # save_checkpoint(option, state, model_dict, is_best, filename=filename)

    # 训练结束打印最终结果
    Logger.info("\n<====== [训练完成] 最终结果 ======>")
    final_result = state.get('final_result', [0.0] * 10)
    Logger.info(f"Hash Pool Radius: {option.R}")
    Logger.info(f"MAP_Rank: {final_result[0]:.4f}, P@5: {final_result[1]:.4f}, P@10: {final_result[2]:.4f}")
    Logger.info(f"P@20: {final_result[3]:.4f}, P@50: {final_result[4]:.4f}, ANMRR: {final_result[9]:.4f}")
    Logger.info(
        f"R@5: {final_result[5]:.4f}, R@10: {final_result[6]:.4f}, R@20: {final_result[7]:.4f}, R@50: {final_result[8]:.4f}")
    Logger.info(f"最终数据已保存至Excel：{excel_filename}")
    Logger.info(f"所有Batch损失已保存至：{batch_loss_excel}")
    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()

def plot_tsne(features, labels, save_path, title="t-SNE"):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os
    import numpy as np

    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()

    labels = np.asarray(labels)

    # ✅ 防止样本太少报错
    perplexity = min(30, max(5, len(features) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init='pca'
    )

    feat_2d = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))

    scatter = plt.scatter(
        feat_2d[:, 0],
        feat_2d[:, 1],
        c=labels,
        s=6,
        cmap='tab20',
        alpha=0.85
    )

    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.yticks([])

    plt.colorbar(scatter)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # =========================
    # ✅ 关键：保存 SVG（论文用）
    # =========================
    svg_path = save_path.replace(".png", ".svg")
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    # =========================
    # ✅ 同时保存高清 PNG（方便查看）
    # =========================
    plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.close()

    print(f"✅ t-SNE saved: {svg_path}")

def train_step(model, optimizer, train_loader, cluster_results, epoch, state, option=None):
    model.train()
    loss_epoch = []
    batch_losses = []
    num_cluster = option.cluster_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not option.IC:
        proto_corresponding, proto_index = parse_proto(cluster_results, num_cluster)
    train_loader = tqdm(train_loader, desc=f"Epoch [{epoch}] Training", leave=True)
    for batch_idx, ((feature_1, feature_2, feature_origin), target, index) in enumerate(train_loader):
        optimizer.zero_grad()
        state['iter'] += 1
        feature_1 = feature_1.to(device)
        feature_2 = feature_2.to(device)
        feature_origin = feature_origin.to(device)
        target = target.to(device)
        index = index.to(device)
        p_3, h_3, p_1, p_2, h_1, h_2 = model((feature_1, feature_2, feature_origin), Train=True)
        quantization_loss = (torch.mean((torch.abs(h_1) - 1.0) ** 2) + torch.mean((torch.abs(h_2) - 1.0) ** 2)) / 2
        instance_contrastive_loss = torch.tensor(0.0, device=device)
        if option.HIC:
            index_cpu = index.cpu().numpy()
            center_index_cpu = proto_index[index_cpu]  # 保持numpy数组，绝对不变形
            instance_contrastive_loss = (weighted_hierarchical_contrastive_loss(
                p_1, p_2, h_1, h_2,
                center_index=center_index_cpu,
                tau1=option.tau1,
                tau2=option.tau2,
                weight1=option.weight1,
                hyper_c=option.hyper_c

            ) + weighted_hierarchical_contrastive_loss(
                p_2, p_1, h_2, h_1,
                center_index=center_index_cpu,
                tau1=option.tau1,
                tau2=option.tau2,
                weight1=option.weight1,
                hyper_c=option.hyper_c

            )) / 2

        proto_contrastive_loss = torch.tensor(0.0, device=device)
        if option.HPC:
            index_cpu = index.cpu().numpy()
            proto_corresponding_batch = torch.tensor(proto_corresponding[index_cpu], device=device)
            proto_index_batch = torch.tensor(proto_index[index_cpu], device=device)
            proto_contrastive_loss = contrastive_proto(
                p_1, p_2,
                center_corresponding=proto_corresponding_batch,
                center_index=proto_index_batch,
                results=cluster_results,
                tau1=option.tau1,
                tau2=option.tau2,
                hyper_c=option.hyper_c
            )
#50*0.002 =
        if epoch <= 25:
            proto_weight = option.weight2
        else:
            proto_weight = option.weight2 / 2
        loss2 = instance_contrastive_loss  # 实例权重1.0
        loss1 = option.lambda_q * quantization_loss  # 量化权重lambda_q
        loss3 = proto_weight * proto_contrastive_loss  # 原型动态权重

        # 总损失：100%等于训练实际值
        total_loss = loss1 + loss2 + loss3
        loss1_val = loss1.item()
        loss2_val = loss2.item()
        loss3_val = loss3.item()
        total_loss_val = total_loss.item()

        # 日志记录（真实训练值）
        batch_loss_info = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'quant_loss': loss1_val,
            'instance_loss': loss2_val,
            'proto_loss': loss3_val,
            'total_loss': total_loss_val,
        }
        batch_losses.append(batch_loss_info)
        loss_epoch.append(total_loss_val)

        # 进度条打印（所见即所得）
        train_loader.set_postfix(
            quant=f"{loss1_val:.4f}",
            inst=f"{loss2_val:.4f}",
            proto=f"{loss3_val:.4f}",
            total=f"{total_loss_val:.4f}",
            refresh=True
        )
        # loss1.backward(retain_graph=True)
        if option.HIC and option.HPC:
            loss2.backward(retain_graph=True)
            loss3.backward()
        elif option.HIC:
            loss2.backward()
        elif option.HPC:
            loss3.backward()
        else:
            pass
        optimizer.step()

    return np.mean(loss_epoch), batch_losses


def predict_hash_code(option, state, model, data_loader, epoch, database_type: str, train=None):
    model.eval()
    data_loader = tqdm(data_loader, desc=f"epoch[{epoch}]==>{database_type}==>Testing:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if train:
        for i, ((input1, input2, input3), target, index) in enumerate(data_loader):
            images = input3.to(device)
            target = target.to(device)
            hyper_code, hash_code = model(images, False)
            if i == 0:
                hash_codes = hash_code
                hyper_codes = hyper_code
                all_label = target
            else:
                hash_codes = torch.cat((hash_codes, hash_code), dim=0)
                hyper_codes = torch.cat((hyper_codes, hyper_code), dim=0)
                all_label = torch.cat((all_label, target), dim=0)
    else:
        for i, (input, target) in enumerate(data_loader):
            images = input.to(device)
            target = target.to(device)
            hyper_code, hash_code = model(images, False)
            if i == 0:
                hash_codes = hash_code
                hyper_codes = hyper_code
                all_label = target
            else:
                hash_codes = torch.cat((hash_codes, hash_code), dim=0)
                hyper_codes = torch.cat((hyper_codes, hyper_code), dim=0)
                all_label = torch.cat((all_label, target), dim=0)

    if option.hyper_c == 0:
        return hyper_codes, hash_codes, all_label
    else:
        return hyper_codes, hash_codes, all_label


def test_step(option, state, model, data_loader, epoch, train=False):
    model.eval()
    if train:
        database_type = "trainbase"
    else:
        database_type = "testbase"
    hyper_code, hash_code, label = predict_hash_code(option, state, model, data_loader, epoch,
                                                     database_type=database_type, train=train)

    # 转换为NumPy计算评估指标
    database_hashcode = hash_code.detach().cpu().numpy().astype('float32')
    database_hypercode = hyper_code.detach().cpu().numpy().astype('float32')
    database_labels = label.detach().cpu().numpy().astype('int8')
    test_hashcode = hash_code.detach().cpu().numpy().astype('float32')
    test_hypercode = hyper_code.detach().cpu().numpy().astype('float32')
    test_labels = label.detach().cpu().numpy().astype('int8')

    Logger.info("===> start calculate MAP!\n")
    # metrics = evaluate_with_strategy(strategy='adaptive_fusion',db_euc=test_hashcode,q_euc=test_hashcode,db_hyp=test_hypercode,q_hyp=test_hypercode,db_labels=test_labels, test_labels=test_labels,option=option)
    metrics = mean_average_precision(
        # database_hypercode, test_hypercode,
        database_hashcode, test_hashcode,
        database_labels, test_labels, option
    )
    return metrics


if __name__ == '__main__':
    option = parser.parse_args()
    option.cluster_num = [int(x) for x in option.cluster_num.split(',')]

    set_seed(42)
    start_time = datetime.datetime.now()
    Logger.info("\t\tstart program\t\t")
    Logger.divider("print option")
    for k, v in vars(option).items():
        Logger.info(f'\t{k}: {v}')
    state = {'start_time': start_time}

    option.radius_per_level = [float(x) for x in option.radius_per_level.split(',')]
    main(option, state)
    end_time = datetime.datetime.now()
    Logger.divider(f"END {Logger.getTimeStr(end_time)}")
