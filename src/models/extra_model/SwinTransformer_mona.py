import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models import SwinTransformer_mona
from src.hyptorch.nn import ToPoincare, HypLinear

''''''
class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class MainModel(torch.nn.Module):
    def __init__(self, option):
        super(MainModel, self).__init__()
        self.option = option

        # Swin Transformer 主干网络
        self.model = SwinTransformer_mona()
        self.model.init_weights(pretrained="C:/Users/hipeson/.cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 冻结所有Swin层
        self.model.frozen_stages = 4
        self.model._freeze_stages()

        # 冻结所有模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 确保Mona模块可训练
        for name, param in self.model.named_parameters():
            if 'my_module' in name:
                param.requires_grad = True

        self.hash_layer = nn.Sequential(
            nn.Linear(768, 512),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, option.hash_bit),
        )

        self.toPoincare = ToPoincare(
            c=option.hyper_c,
            ball_dim=option.hyper_dim,
            riemannian=False,
            clip_r=option.clip_r

        )

        self.head = nn.Sequential(
            nn.Linear(self.option.hash_bit, self.option.hyper_dim),
            self.toPoincare,
        )

    def get_features(self, x):
        swin_feat_map = self.model(x)[3]
        pooled_feat = self.avg_pool(swin_feat_map)
        swin_feat = torch.flatten(pooled_feat, start_dim=1)
        return swin_feat

    def forward(self, images, Train: bool):

        if Train:
                # 提取特征
                aug_one_fea = self.get_features(images[0])
                aug_two_fea = self.get_features(images[1])
                origin_fea = self.get_features(images[2])
                # 哈希编码
                a1 = torch.tanh(self.hash_layer(aug_one_fea))
                a2 = torch.tanh(self.hash_layer(aug_two_fea))
                a3 = torch.tanh(self.hash_layer(origin_fea))

                h1 = F.normalize(a1, dim=1, p=2)
                h2 = F.normalize(a2, dim=1, p=2)
                h3 = F.normalize(a3, dim=1, p=2)

                if self.option.hyper_c == 0:
                    p1, p2, p3 = h1, h2, h3
                else:

                    # 先映射到半径为1的欧式球面上
                    p1 = self.head(a1)
                    p2 = self.head(a2)
                    p3 = self.head(a3)

                return p3, h3, p1, p2, h1, h2
        else:
                with torch.no_grad():
                    features = self.get_features(images)
                    p = self.head(torch.tanh(self.hash_layer(features)))
                    h = torch.tanh(self.hash_layer(features))

                    # p = self.head(self.hash_layer(features))
                    # h = F.normalize(self.hash_layer(features), p=2, dim=1)

                    return p , h

    def getParams(self):
            return [
                {'params': self.hash_layer.parameters(), 'lr': self.option.lr},
                {'params': self.head.parameters(), 'lr': self.option.lr},
                {'params': self.model.parameters(), 'lr': self.option.lr},
            ]