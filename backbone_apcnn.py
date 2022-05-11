import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# def nms(dets, thresh):
#     "Dispatch to either CPU or GPU NMS implementations. Accept dets as tensor"
#     return pth_nms(dets, thresh)
'''
x:[B,C,H,W],feature map
score:[B,1,H,W],spatial attention map
thresh:float

'''


def my_crop(x: torch.Tensor, score: torch.Tensor, thresh=0.05):
    assert 0 < thresh < 1

    def crop_by_dim(x, dim: int, thresh=0.05):
        assert dim in [2, 3]
        sum = torch.sum(x, dim=[2, 3])
        score = torch.sum(x, dim=2 if dim == 3 else 3)
        score = score / sum[:, :, None]
        size = score.shape[2]
        for i in range(1, score.shape[2]):
            score[:, :, i] = score[:, :, i] + score[:, :, i - 1]
        low_idx = torch.sum((score <= thresh).int(), dim=2)
        high_idx = size - torch.sum((score >= 1 - thresh).int(), dim=2)
        return low_idx, high_idx

    with torch.no_grad():
        # score = torch.mean(x.abs(),dim=1,keepdim=True)
        top, buttom = crop_by_dim(score, 2, thresh)
        left, right = crop_by_dim(score, 3, thresh)
        batch_idx = torch.arange(left.shape[0], dtype=left.dtype, device=left.device).unsqueeze(1)
        boxes = torch.cat([batch_idx, left, top, right, buttom], 1).float()
    x = roi_align(x, boxes, x.shape[2:])
    return x, boxes


def mean_tensors(tensors: [], shape):
    ret = None
    for tensor in tensors:
        tensor = F.interpolate(tensor, shape, mode='bilinear', align_corners=False)
        ret = tensor if ret is None else tensor + ret
    if ret is not None:
        ret = ret / len(tensors)
    return ret


class PyramidFeatures(nn.Module):
    """Feature pyramid module with top-down feature pathway"""

    def __init__(self, B2_size, B3_size, B4_size, B5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.F5_0 = nn.Conv2d(B5_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.F4_0 = nn.Conv2d(B4_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.F3_0 = nn.Conv2d(B3_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.F5_1 = nn.Sequential(
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        )
        self.F4_1 = nn.Sequential(
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        )
        self.F3_1 = nn.Sequential(
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        )

        self.F5_attention_s = SpatialGate(feature_size)
        self.F5_attention_c = ChannelGate(feature_size)

        self.F4_attention_s = SpatialGate(feature_size)
        self.F4_attention_c = ChannelGate(feature_size)

        self.F3_attention_s = SpatialGate(feature_size)
        self.F3_attention_c = ChannelGate(feature_size)

        self.F5_2 = nn.Sequential(
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        )
        self.F4_2 = nn.Sequential(
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        )
        self.F3_2 = nn.Sequential(
            nn.BatchNorm2d(feature_size),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, inputs):
        B3, B4, B5 = inputs

        F5_0 = self.F5_0(B5)
        F4_0 = self.F4_0(B4)
        F3_0 = self.F3_0(B3)

        F5_1 = self.F5_1(F5_0)

        F5_up = F.interpolate(F5_1, size=F4_0.shape[2:], mode='bilinear', align_corners=False)
        F4_1 = self.F4_1((F5_up + F4_0) / 2)

        F4_up = F.interpolate(F4_1, size=F3_0.shape[2:], mode='bilinear', align_corners=False)
        F3_1 = self.F3_1((F4_up + F3_0) / 2)

        F5_attention_c = self.F5_attention_c(F5_1)
        F5_1 = F5_1 * F5_attention_c
        F5_attention_s = self.F5_attention_s(F5_1)
        F5_1 = F5_1 * F5_attention_s

        F4_attention_c = self.F4_attention_c(F4_1)
        F4_1 = F4_1 * F4_attention_c
        F4_attention_s = self.F4_attention_s(F4_1)
        F4_1 = F4_1 * F4_attention_s

        F3_attention_c = self.F3_attention_c(F3_1)
        F3_1 = F3_1 * F3_attention_c
        F3_attention_s = self.F3_attention_s(F3_1)
        F3_1 = F3_1 * F3_attention_s

        F3_2 = self.F3_2((F3_0 + F3_1) / 2)

        F3_down = F.interpolate(F3_2, size=F4_1.shape[2:], mode='bilinear', align_corners=False)
        F4_2 = self.F4_2((F4_0 + F4_1 + F3_down) / 3)

        F4_down = F.interpolate(F4_2, size=F5_1.shape[2:], mode='bilinear', align_corners=False)
        F5_2 = self.F5_2((F5_0 + F5_1 + F4_down) / 3)

        return [F3_2, F4_2, F5_2, F3_attention_s, F4_attention_s, F5_attention_s]


class SpatialGate(nn.Module):
    """generation spatial attention mask"""

    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x).abs()
        # return torch.sigmoid(x)
        return x


class ChannelGate(nn.Module):
    """generation channel attention mask"""

    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x).abs()
        return x


class APCNN(nn.Module):
    """implementation of AP-CNN on ResNet"""

    def __init__(self, backbone, num_classes):
        super(APCNN, self).__init__()
        final_feat_channel = 64
        self.backbone = backbone
        self.do_flatten = backbone.do_flatten
        self.fpn = PyramidFeatures(self.backbone.fpn_sizes[0], self.backbone.fpn_sizes[1], self.backbone.fpn_sizes[2],
                                   self.backbone.fpn_sizes[3], feature_size=final_feat_channel)
        # self.apn = PyramidAttentions(channel_size=final_feat_channel)
        self.cls_concate = nn.Sequential(
            nn.BatchNorm2d(final_feat_channel * 3),
            nn.Conv2d(final_feat_channel * 3, final_feat_channel, kernel_size=1),
            nn.BatchNorm2d(final_feat_channel),
            nn.ELU(inplace=True),
            nn.Conv2d(final_feat_channel, final_feat_channel, kernel_size=1),
        )
        self.traditional_cls = nn.Sequential(
            nn.BatchNorm1d(final_feat_channel),
            nn.Linear(final_feat_channel, final_feat_channel),
            nn.BatchNorm1d(final_feat_channel),
            nn.ELU(inplace=True),
            nn.Linear(final_feat_channel, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                n = m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def Pool_Concate(self, f3, f4, f5, output_size):
        f3 = F.adaptive_avg_pool2d(f3, output_size)
        f4 = F.adaptive_avg_pool2d(f4, output_size)
        f5 = F.adaptive_avg_pool2d(f5, output_size)
        f_concate = torch.cat([f3, f4, f5], dim=1)
        return f_concate

    def visualize(self, inputs, croped_inputs, boxes, a3, a4, a5):
        print(boxes[0])
        mean = np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float)
        std = np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float)

        img = inputs[0].detach().cpu().numpy()
        img = img * std + mean
        x1, y1, x2, y2 = boxes[0].detach().cpu().numpy()[1:]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img[0, y1, x1:x2] = 1.0
        img[0, y2, x1:x2] = 1.0
        img[0, y1:y2, x1] = 1.0
        img[0, y1:y2, x2] = 1.0

        img[0, y1 - 1, x1:x2] = 1.0
        img[0, y2 - 1, x1:x2] = 1.0
        img[0, y1:y2, x1 - 1] = 1.0
        img[0, y1:y2, x2 - 1] = 1.0

        img[0, y1 + 1, x1:x2] = 1.0
        img[0, y2 + 1, x1:x2] = 1.0
        img[0, y1:y2, x1 + 1] = 1.0
        img[0, y1:y2, x2 + 1] = 1.0

        img[1, y1, x1:x2] = 0.0
        img[1, y2, x1:x2] = 0.0
        img[1, y1:y2, x1] = 0.0
        img[1, y1:y2, x2] = 0.0

        img[1, y1 - 1, x1:x2] = 0.0
        img[1, y2 - 1, x1:x2] = 0.0
        img[1, y1:y2, x1 - 1] = 0.0
        img[1, y1:y2, x2 - 1] = 0.0

        img[1, y1 + 1, x1:x2] = 0.0
        img[1, y2 + 1, x1:x2] = 0.0
        img[1, y1:y2, x1 + 1] = 0.0
        img[1, y1:y2, x2 + 1] = 0.0

        img[2, y1, x1:x2] = 0.0
        img[2, y2, x1:x2] = 0.0
        img[2, y1:y2, x1] = 0.0
        img[2, y1:y2, x2] = 0.0

        img[2, y1 - 1, x1:x2] = 0.0
        img[2, y2 - 1, x1:x2] = 0.0
        img[2, y1:y2, x1 - 1] = 0.0
        img[2, y1:y2, x2 - 1] = 0.0

        img[2, y1 + 1, x1:x2] = 0.0
        img[2, y2 + 1, x1:x2] = 0.0
        img[2, y1:y2, x1 + 1] = 0.0
        img[2, y1:y2, x2 + 1] = 0.0

        img = img.transpose((1, 2, 0))

        crop_img = croped_inputs[0].detach().cpu().numpy()
        crop_img = crop_img * std + mean
        crop_img = crop_img.transpose((1, 2, 0))

        a5 = a5[:1][:1].detach()
        a5 = F.interpolate(a5, size=img.shape[:2], mode='bilinear', align_corners=False)[0][0].cpu().numpy()
        a4 = a4[:1][:1].detach()
        a4 = F.interpolate(a4, size=img.shape[:2], mode='bilinear', align_corners=False)[0][0].cpu().numpy()
        a3 = a3[:1][:1].detach()
        a3 = F.interpolate(a3, size=img.shape[:2], mode='bilinear', align_corners=False)[0][0].cpu().numpy()
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.subplot(2, 3, 2)
        plt.imshow(a3)
        plt.subplot(2, 3, 3)
        plt.imshow(a4)
        plt.subplot(2, 3, 4)
        plt.imshow(a5)
        plt.subplot(2, 3, 5)
        plt.imshow(crop_img)
        plt.show()

    def forward(self, inputs):
        # ResNet backbone with FC removed
        n, c, img_h, img_w = inputs.size()
        x2, x3, x4, backbone_out = self.backbone(inputs)
        if self.do_flatten:
            embedding_HW = (1, 1)
        else:
            embedding_HW = backbone_out.shape[2:]

        # stage I
        f3_att, f4_att, f5_att, a3, a4, a5 = self.fpn([x2, x3, x4])

        # feature concat
        f_concate = self.Pool_Concate(f3_att, f4_att, f5_att, embedding_HW)

        out_concate1 = self.cls_concate(f_concate)
        traditional_cls_score1 = self.traditional_cls(out_concate1.mean(dim=[2, 3]))

        part_x2 = a3 * x2 / torch.mean(a3, dim=[1, 2, 3], keepdim=True)
        part_x3 = a4 * x3 / torch.mean(a4, dim=[1, 2, 3], keepdim=True)
        part_x4 = a5 * x4 / torch.mean(a5, dim=[1, 2, 3], keepdim=True)
        weight_x_raw = self.Pool_Concate(part_x2, part_x3, part_x4, embedding_HW)
        # _, predicted = torch.max(out.data, 1)
        # correct = predicted.eq(targets.data).cpu().sum().item()

        # stage II
        spatial_attention = mean_tensors([a3], inputs.shape[2:])
        crop_inputs, boxes = my_crop(inputs, spatial_attention, 0.05)

        x2, x3, x4, _ = self.backbone(crop_inputs)

        # self.visualize(inputs,crop_inputs,boxes,a3,a4,a5)
        f3_att, f4_att, f5_att, a3, a4, a5 = self.fpn([x2, x3, x4])

        # feature concat
        f_concate = self.Pool_Concate(f3_att, f4_att, f5_att, embedding_HW)

        out_concate2 = self.cls_concate(f_concate)
        traditional_cls_score2 = self.traditional_cls(out_concate2.mean(dim=[2, 3]))

        part_x2 = a3 * x2 / torch.mean(a3, dim=[1, 2, 3], keepdim=True)
        part_x3 = a4 * x3 / torch.mean(a4, dim=[1, 2, 3], keepdim=True)
        part_x4 = a5 * x4 / torch.mean(a5, dim=[1, 2, 3], keepdim=True)
        weight_x_refine = self.Pool_Concate(part_x2, part_x3, part_x4, embedding_HW)
        embeddings = torch.cat([out_concate1, out_concate2, weight_x_raw * 10, weight_x_refine * 10], dim=1)
        if isinstance(self.backbone.final_feat_dim, int):
            embeddings = embeddings.squeeze(3).squeeze(2)
        return embeddings / np.sqrt(embeddings.shape[1]), traditional_cls_score1 + traditional_cls_score2


class DoNothing(nn.Module):
    def __init__(self, backbone, num_classes):
        super(DoNothing, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, inputs):
        x2, x3, x4, embedding = self.backbone(inputs)
        return embedding, torch.zeros((inputs.shape[0], self.num_classes), dtype=embedding.dtype,
                                      device=embedding.device)


if __name__ == '__main__':
    x = torch.arange(64).view((1, 1, 8, 8)).float()
    score = torch.ones((1, 1, 8, 8)).float()
