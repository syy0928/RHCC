import torch
import torchvision

import torch.nn.functional as F
from src.hyptorch.nn import ToPoincare, HypLinear
import torch.nn as nn


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


OVERFLOW_MARGIN = 1e-8
# diagnosis.py
import sys
import torch
import torchvision
import subprocess


#####################################################Vit#########################################
'''
class MainModel(torch.nn.Module):
    def __init__(self, option):
        super(MainModel, self).__init__()
        self.option = option
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 初始化ViT-Base/16模型（不加载默认预训练权重）
        self.backbone = torchvision.models.vit_b_16(pretrained=True)
        self.swin_backbone = torchvision.models.swin_t(pretrained=True)
        self.swin_backbone = self.swin_backbone.features  # 此时 self.swin_backbone 仅输出4维特征图

        # 冻结主干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.swin_backbone.parameters():
            param.requires_grad = False

        # 哈希层（ViT-Base的CLS特征维度为768）
        self.hash_layer = nn.Sequential(
            nn.Linear(768, 512),
            # nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Linear(512, option.hash_bit)
        )

        self.toPoincare = ToPoincare(
            c=option.hyper_c,
            ball_dim=option.hash_bit,
            riemannian=False,
            clip_r=option.clip_r
        )

        self.head = nn.Sequential(
            nn.Linear(self.option.hash_bit, self.option.hyper_dim),
            self.toPoincare
        )

        # 验证特征提取是否正确
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            features = self.get_swin_features(test_input)
            print(f"特征维度: {features.shape}")  # 应输出 torch.Size([1, 768])

    def get_vit_features(self, x):
        # 根据你的ViT层级结构，手动提取CLS特征
        # 1. 图像分块与嵌入（conv_proj）
        x = self.backbone.conv_proj(x)  # 输出: [batch, 768, H/patch_size, W/patch_size]

        # 2. 展平空间维度，转为序列
        x = x.flatten(2).transpose(1, 2)  # 输出: [batch, num_patches, 768]

        # 3. 拼接class_token
        batch_size = x.shape[0]
        class_tokens = self.backbone.class_token.expand(batch_size, -1, -1)  # [batch, 1, 768]
        x = torch.cat([class_tokens, x], dim=1)  # 输出: [batch, num_patches+1, 768]

        # 4. 添加位置编码
        x = x + self.backbone.encoder.pos_embedding

        # 5. 通过Transformer编码器
        x = self.backbone.encoder.layers(x)

        # 6. 最后的LayerNorm
        x = self.backbone.encoder.ln(x)

        # 7. 取CLS token（第0个位置）
        return x[:, 0]  # 输出: [batch, 768]

    def get_swin_features(self, x):
        # 1. Swin输出格式：[B, H, W, C] → 示例：[64,7,7,768]
        swin_feat_map = self.swin_backbone(x)
        # print(f"Swin原始输出: {swin_feat_map.shape}")  # 打印：[64,7,7,768]

        # 2. 修正维度顺序：[B, H, W, C] → [B, C, H, W]（关键！）
        swin_feat_map = swin_feat_map.permute(0, 3, 1, 2)  # 0=B,3=C,1=H,2=W
        # print(f"Swin修正后: {swin_feat_map.shape}")  # 目标：[64,768,7,7]

        # 3. 全局平均池化：压缩H/W维度→[64,768,1,1]
        pooled_feat = self.avg_pool(swin_feat_map)
        # print(f"池化后: {pooled_feat.shape}")  # 应输出 [64,768,1,1]

        # 4. 展平：移除H/W维度→[64,768]
        swin_feat = torch.flatten(pooled_feat, start_dim=1)
        # print(f"Swin最终特征: {swin_feat.shape}")  # 目标：[64,768]

        return swin_feat

    def forward(self, images, Train: bool):
        if Train:
            # 提取三个增强图像的特征
            aug_one_pre = self.get_vit_features(images[0])
            aug_two_pre = self.get_vit_features(images[1])
            origin_image = self.get_vit_features(images[2])

            # 计算哈希码
            h_1 = torch.tanh(self.hash_layer(aug_one_pre))
            h_2 = torch.tanh(self.hash_layer(aug_two_pre))
            h_3 = torch.tanh(self.hash_layer(origin_image))

            # 投影到双曲空间
            if self.option.hyper_c == 0:
                p_1 = h_1
                p_2 = h_2
                p_3 = h_3
            else:

                p_1 = self.head(h_1)
                p_2 = self.head(h_2)
                p_3 = self.head(h_3)

            return p_3, h_3, p_1, p_2, h_1, h_2


        else:
            with torch.no_grad():
                features = self.get_vit_features(images)
                hash_code = torch.tanh(self.hash_layer(features))
                return hash_code

    def getParams(self):
        return [
            {'params': self.hash_layer.parameters(), 'lr': self.option.lr},
            {'params': self.head.parameters(), 'lr': self.option.lr}
        ]
'''

#################################################VGG##########################################################

class MainModel(torch.nn.Module):

    def __init__(self, option):
        super(MainModel, self).__init__()
        self.option = option
        ######################VGG################################
        self.backbone = torchvision.models.vgg19(pretrained=True)
        # 2. 从检查点中提取真正的模型权重（关键步骤！）
        # 根据参数列表，模型权重存储在 'model_dict' 键下
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:6])

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.hash_layer = nn.Sequential(nn.Linear(4096, 512), nn.Dropout(0.1),
                                        nn.ReLU(), nn.Linear(512, option.hash_bit))
        self.toPoincare = ToPoincare(c=option.hyper_c,
                                     ball_dim=option.hash_bit,
                                     riemannian=False,
                                     clip_r=option.clip_r)
        self.head = nn.Sequential(nn.Linear(self.option.hash_bit, self.option.hyper_dim), self.toPoincare)



    def forward(self, images, Train: bool):
        if Train:
            ################## VGG #####################
            aug_one_pre = self.backbone.features(images[0])
            aug_one_pre = aug_one_pre.view(aug_one_pre.size(0), -1)
            aug_one_pre = self.backbone.classifier(aug_one_pre)
            aug_two_pre = self.backbone.features(images[1])
            aug_two_pre = aug_two_pre.view(aug_two_pre.size(0), -1)
            aug_two_pre = self.backbone.classifier(aug_two_pre)

            origin_image = self.backbone.features(images[2])
            origin_image = origin_image.view(origin_image.size(0), -1)
            origin_image = self.backbone.classifier(origin_image)

            h_1 = torch.tanh(self.hash_layer(aug_one_pre))
            h_2 = torch.tanh(self.hash_layer(aug_two_pre))
            h_3 = torch.tanh(self.hash_layer(origin_image))
            #project to Poincare ball (hyperbolic space)
            if self.option.hyper_c == 0:
                p_1 = h_1
                p_2 = h_2
                p_3 = h_3
            else:
                p_1 = self.head(h_1)
                p_2 = self.head(h_2)
                p_3 = self.head(h_3)
            """
            p_3: feature of image without augmentation in the hyperbolic space, for tree construction
            p_1: feature of image with augmentation a in the hyperbolic space
            p_2: feature of image with augmentation b in the hyperbolic space
            h_1: hash codes of image with augmentation a 
            h_2: hash codes of image with augmentation b 
            Note that p_1 \\approx h_1 when c -> 0
            """

            return p_3, h_3, p_1, p_2, h_1, h_2
        #原始图像的双曲空间哈希码，原始图像的哈希码，
        else:
            with torch.no_grad():
                ################## VGG ##################
                images = self.backbone.features(images)
                images = images.view(images.size(0), -1)
                images = self.backbone.classifier(images)

                hash_code = torch.tanh(self.hash_layer(images))#[-1,1]
                hyper_code = self.head(hash_code)
                return hyper_code, hash_code

    def getParams(self):
        return [
            {'params': self.hash_layer.parameters(), 'lr': self.option.lr},
            {'params': self.head.parameters(), 'lr': self.option.lr}
        ]


#################################ResNet50##################################################
'''
class MainModel(torch.nn.Module):
    def __init__(self, option):
        super(MainModel, self).__init__()
        self.option = option
        ###################### 替换为ResNet50 Backbone ################################
        # 1. 加载预训练ResNet50（保持pretrained=True与原VGG19逻辑一致）
        self.backbone = torchvision.models.resnet50(pretrained=True)

        # 2. 改造ResNet50的分类器：移除默认的最终全连接层（原输出1000类）
        #    - ResNet50默认结构：... → 全局平均池化 → fc(2048→1000)
        #    - 改造后：... → 全局平均池化 → 输出2048维特征（直接接入哈希层）
        self.backbone.fc = nn.Identity()  # 用恒等映射替代原fc层，保留2048维特征输出

        # 3. 冻结Backbone权重（与原代码逻辑一致，不训练预训练骨干网络）
        for param in self.backbone.parameters():
            param.requires_grad = False

        ###################### 哈希层适配ResNet50的2048维输入 ######################
        # 原VGG19输出4096维，故哈希层第一层为Linear(4096, 512)
        # 现ResNet50输出2048维，仅修改哈希层第一层输入维度为2048，其余结构完全不变
        self.hash_layer = nn.Sequential(
            nn.Linear(2048, 512),  # 输入维度从4096→2048（适配ResNet50输出）
            # nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Linear(512, option.hash_bit)  # 输出维度仍为哈希码位数，与原逻辑一致
        )

        ###################### 以下部分（双曲投影、head）完全不变 ######################
        self.toPoincare = ToPoincare(
            c=option.hyper_c,
            ball_dim=option.hash_bit,
            riemannian=False,
            clip_r=option.clip_r
        )
        self.head = nn.Sequential(
            nn.Linear(self.option.hash_bit, self.option.hyper_dim),
            self.toPoincare
        )

    def forward(self, images, Train: bool):
        if Train:
            ################## ResNet50特征提取（适配其接口，逻辑与原VGG19一致） #####################
            # ResNet50无需手动调用features/classifier，直接通过backbone()输出2048维特征
            # 输入images[0/1/2]对应三种数据（增强1、增强2、原始），与原逻辑完全一致
            aug_one_pre = self.backbone(images[0])  # 输出2048维特征
            aug_two_pre = self.backbone(images[1])  # 输出2048维特征
            origin_image = self.backbone(images[2])  # 输出2048维特征

            ################## 哈希层、双曲投影等逻辑完全不变 #####################
            # 2048维特征直接接入哈希层，输出哈希码（与原VGG19的4096维输入流程一致）
            h_1 = torch.tanh(self.hash_layer(aug_one_pre))
            h_2 = torch.tanh(self.hash_layer(aug_two_pre))
            h_3 = torch.tanh(self.hash_layer(origin_image))

            # 双曲空间投影（与原逻辑完全一致）
            if self.option.hyper_c == 0:
                p_1 = h_1
                p_2 = h_2
                p_3 = h_3
            else:
                p_1 = self.head(h_1)
                p_2 = self.head(h_2)
                p_3 = self.head(h_3)

            return p_3, h_3, p_1, p_2, h_1, h_2

        else:
            with torch.no_grad():
                ################## 测试模式：ResNet50特征提取+哈希码生成（逻辑不变） ##################
                images = self.backbone(images)  # 输出2048维特征
                hash_code = torch.tanh(self.hash_layer(images))  # 生成[-1,1]哈希码，与原逻辑一致
                return hash_code

    def getParams(self):
        # 训练参数列表完全不变（仅训练hash_layer和head，与原逻辑一致）
        return [
            {'params': self.hash_layer.parameters(), 'lr': self.option.lr},
            {'params': self.head.parameters(), 'lr': self.option.lr}
        ]
'''
if __name__ == '__main__':
    hash_codes = torch.tensor([[-1., 1., -1., 1.], [-1., 1., 1., 1.], [1., -1., -1., 1.], [-1., 1., 1., 1.]])
