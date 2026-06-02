import torch

from src.Loss.Loss import hierarchical_contrastive_loss, contrastive_loss, contrastive_proto


def compute_losses(p1, p2, h1, h2, option, cluster_results,
                   proto_corresponding, proto_index, index, is_train=True):
    """
    封装损失计算逻辑，可用于训练和测试阶段

    参数:
        p1, p2: 模型输出的双曲映射特征
        h1, h2: 模型输出的哈希码
        option: 配置参数对象
        cluster_results: 聚类结果
        proto_corresponding: 原型对应的特征 (来自parse_proto)
        proto_index: 原型索引 (来自parse_proto)
        index: 当前批次样本的索引
        is_train: 是否为训练模式 (训练模式保留梯度，测试模式不保留)

    返回:
        loss_dict: 包含总损失和各分项损失的字典
    """
    # 确保使用正确的设备
    device = p1.device

    # 量化损失
    quantization_loss = torch.mean((torch.abs(h1) - 1.0) ** 2) + \
                        torch.mean((torch.abs(h2) - 1.0) ** 2)
    quantization_loss = quantization_loss / 2

    # 实例对比损失
    instance_contrastive_loss = torch.tensor(0.0, device=device)
    if option.HIC or option.IC:
        if option.HIC:
            loss1 = hierarchical_contrastive_loss(
                p1, p2,h1,h2, center_index=proto_index[index],
                tau=option.tau, hyper_c=option.hyper_c
            )
            loss2 = hierarchical_contrastive_loss(
                p2, p1,h2,h1, center_index=proto_index[index],
                tau=option.tau, hyper_c=option.hyper_c
            )
        else:  # option.IC
            loss1 = contrastive_loss(
                p1, p2, tau=option.tau, hyper_c=option.hyper_c
            )[0]
            loss2 = contrastive_loss(
                p2, p1, tau=option.tau, hyper_c=option.hyper_c
            )[0]
        instance_contrastive_loss = (loss1 + loss2) / 2

    # 原型对比损失
    proto_contrastive_loss = torch.tensor(0.0, device=device)
    if option.HPC:
        proto_corresponding_batch = proto_corresponding[index]
        proto_index_batch = proto_index[index]
        proto_contrastive_loss = contrastive_proto(
            p1, p2,
            center_corresponding=proto_corresponding_batch,
            center_index=proto_index_batch,
            results=cluster_results,
            tau=option.tau,
            hyper_c=option.hyper_c
        )

    # 总损失计算
    total_loss = 1.0 * instance_contrastive_loss + \
                 option.lambda_q * quantization_loss + \
                 proto_contrastive_loss

    # 构建损失字典
    loss_dict = {
        'total_loss': total_loss,
        'quantization_loss': quantization_loss,
        'instance_contrastive_loss': instance_contrastive_loss,
        'proto_contrastive_loss': proto_contrastive_loss
    }

    return loss_dict
