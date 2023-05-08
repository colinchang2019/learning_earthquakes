# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : loss.py
@ Time    ：2021/6/16 17:20
"""
import torch
from torch import nn
from config import config
from functools import reduce
import numpy as np
from utils.standardrange import *


class WeightedMseMae(nn.Module):
    def __init__(self, weight=config.weight):
        super().__init__()
        self.mse_weight, self.mae_weight = config.mse_mae_weights
        self.normal_loss_global_scale = config.normal_loss_global_scale
        # 区域内的权重, 基于Shi et al. (2015)
        path = config.sample
        self.area = np.load(path)["sequence_array"]
        print(
            "use localization sample in WeightedMseMae loss_function: {}, with weight of {}".format(path,
                                                                                                    config.balancing_weights))

    def forward(self, input, target, ar=None):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target1 = target * 255.0  # transform, restore

        sz = input.size()
        if ar is None:
            ar = self.area
        r = np.array([ar] * (sz[0] * sz[1]))
        r = r.reshape(*sz)
        r = torch.from_numpy(r).to(input.device)

        balancing_weights = config.balancing_weights
        weights = torch.ones_like(input) * balancing_weights[0]
        weights += r * (balancing_weights[1] - balancing_weights[0])
        thresholds = [mag_to_pixel(ele) for ele in config.THRESHOLDS]  # 几个门槛
        for i, threshold in enumerate(thresholds):
            # 这个循环做到是对不同threshold做不同的权重
            weights = weights + (balancing_weights[i + 2] - balancing_weights[i+1]) * (target1 >= threshold).float()
        # weights = weights * r  # 只保留了目标区域

        mse = torch.sum(weights * ((input - target) ** 2), dim=tuple(range(1, len(sz))))
        mae = torch.sum(weights * (torch.abs((input - target))), dim=tuple(range(1, len(sz))))

        return self.normal_loss_global_scale * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))


class weightedMseDD(nn.Module):
    def __init__(self, weight=config.weight, weights_step=config.weights_step):
        super().__init__()
        self.weight = weight  # weight / (1 - weight)是单个元素，有地震和无地震之间的权重之比
        self.weights_step = weights_step  # 区域内的权重
        path = config.sample
        self.area = np.load(path)["sequence_array"]
        print(
            "use localization sample in weightedMseDD loss_function: {}, with weights of {}".format(path, weights_step))

    def forward(self, input, target):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target1 = target * 255.0  # transform, restore
        threshold = config.threshold

        sz = input.size()
        r = np.array([self.area] * (sz[0] * sz[1]))
        r = r.reshape(*sz)
        r = torch.from_numpy(r).to(input.device)

        underThreshold = 1 - r
        aboveThreshold = torch.where(target1 >= threshold, 1, 0)
        mid_threshold = (1 - aboveThreshold) * r  # 表示在r内，但是在above之外

        sumabove = torch.sum(aboveThreshold)
        sumunder = torch.sum(underThreshold)
        summid = torch.sum(mid_threshold)

        weight_above, weight_under = self.weight, 1 - self.weight
        wight_under, weight_mid, weight_above = self.weights_step

        res = torch.sum((input - target) ** 2 * underThreshold) * weight_under
        res += torch.sum((input - target) ** 2 * mid_threshold) * weight_mid
        res += torch.sum((input - target) ** 2 * aboveThreshold) * weight_above
        res /= (sumabove * weight_above + summid * weight_mid + sumunder * weight_under)
        return res


class weightedMseD(nn.Module):
    def __init__(self, weight=config.weight):
        super().__init__()
        self.weight = weight  # weight / (1 - weight)是单个元素，有地震和无地震之间的权重之比
        # 区域内的权重
        path = config.sample
        self.area = np.load(path)["sequence_array"]
        print(
            "use localization sample in weightedMseD loss_function: {}, with weight of {}".format(path, weight))

    def forward(self, input, target):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target1 = target * 255.0  # transform, restore
        threshold = config.threshold
        sz = input.size()
        r = np.array([self.area] * (sz[0] * sz[1]))
        r = r.reshape(*sz)
        r = torch.from_numpy(r).to(input.device)

        underThreshold = 1 - r
        aboveThreshold = r

        sumabove = torch.sum(aboveThreshold)
        sumunder = torch.sum(underThreshold)

        weight_above, weight_under = self.weight, 1 - self.weight

        res = torch.sum((input - target) ** 2 * underThreshold) * weight_under
        res += torch.sum((input - target) ** 2 * aboveThreshold) * weight_above
        res /= (sumabove * weight_above + sumunder * weight_under)
        return res


class weightedMse(nn.Module):
    def __init__(self, weight=config.weight):
        super().__init__()
        self.weight = weight  # weight / (1 - weight)是单个元素，有地震和无地震之间的权重之比
        print("use normal weightedMse, with weight of {} and threshold of {} ".format(weight, config.threshold))

    def forward(self, input, target):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:
        """
        target1 = target * 255.0  # transform, restore
        threshold = config.threshold

        underThreshold = torch.where(target1 < threshold, 1, 0)
        aboveThreshold = torch.where(target1 >= threshold, 1, 0)

        sumabove = torch.sum(aboveThreshold)
        sumunder = torch.sum(underThreshold)

        weight_above, weight_under = self.weight, 1 - self.weight

        res = torch.sum((input - target) ** 2 * underThreshold) * weight_under
        res += torch.sum((input - target) ** 2 * aboveThreshold) * weight_above
        res /= (sumabove * weight_above + sumunder * weight_under)
        return res


if __name__=="__main__":
    loss = weightedMseD()
    loss1 = nn.MSELoss()

    input = torch.rand(2, 1, config.height, config.width)
    target = torch.rand(2, 1, config.height, config.width)
    print(loss(input, target))
    print(loss1(input, target))