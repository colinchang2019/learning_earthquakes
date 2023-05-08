# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : run_rotate.py
@ Time    ：2022/4/18 22:21
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import time
from config import config
from utils.standardrange import *

# from model import EncoderForecaster
# from modelForN import EncoderForecaster
from modelForModify import EncoderForecaster
from utils.earlyStopping import EarlyStopping
from utils.loss import weightedMse, weightedMseD, WeightedMseMae
from utils.dataset import MultiDataset
from utils.rotate import rotate_sequence
from utils.assesment import *
from utils.dataset import MultiDataset, SingleDataset

LR, STEP_SIZE, GAMMA = config.pre_parameters
# LR = 0.001
# STEP_SIZE = 10000
# GAMMA = 0.9

# 固定随机数种子，保证结果可重复

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False

setup_seed(2019)

def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)


import logging

# 1.显示创建
logging.basicConfig(filename='logger.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 2.定义logger,设定setLevel，FileHandler，setFormatter
logger = logging.getLogger(__name__)  # 定义一次就可以，其他地方需要调用logger,只需要直接使用logger就行了
logger.setLevel(level=logging.INFO)  # 定义过滤级别

pathlog = "log_train.txt"
filehandler = logging.FileHandler(pathlog)  # Handler用于将日志记录发送至合适的目的地，如文件、终端等
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)

console = logging.StreamHandler()  # 日志信息显示在终端terminal
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(console)

logger.info("Start log")

logger.info("Parametes:　LR: {}, STEP_SIZE:{}, GAMMA: {}".format(LR, STEP_SIZE, GAMMA))


def main(path="C:/Users/chesley/Pictures/npe/second", case=0, n_train=15076, n_test=2542, dt=100, st=0, ste=0, rotate=0):
    """
    :param path: "C:/Users/chesley/Pictures/ne7/second" as default
    :param case:
    :param n_train:15076 for train
    :param n_test: 2542 for test
    :param dt: 1000 or 3000
    :return:
    """
    logger.info("Run in file: {}".format(path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if case == 0:
        case_str = "_n_"
    elif case == 1:
        case_str = "_r_"
    elif case == 2:
        case_str = "_r3_"
    else:
        case_str = "_r4_"

    # create dataloader
    print("Laoding dataset to torch.")
    path_fa = path + "_half_"  # "C:/Users/chesley/Pictures/ne7/second_half_" as default
    trainDataset = SingleDataset(path_fa, n_train, dt, case_str, "train", st)  # 合并训练数据和目标数据
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=_init_fn
    )
    testDataset = SingleDataset(path_fa, n_test, dt, case_str, "test", ste)
    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.num_workers
    )
    print("Dataset prepared.")

    torch.cuda.empty_cache()  # 清理显存

    model = EncoderForecaster(config).to(device)
    # model = EFN(config).to(device)  # 建立模型

    # loss_fn = nn.MSELoss()  # 定义均方差作为损失函数
    # loss_fn = weightedMse()
    # loss_fn = weightedMseD()
    # loss_fn = weightedMseDD()
    loss_fn = WeightedMseMae()
    loss_fn.to(device)

    loss_fn_p = precision()
    loss_fn_r = recall()

    optimiser = optim.Adam(params=model.parameters(), lr=LR)  # 定义优化方法
    # weight_decay=0.01 参数设置能让优化器自动带有L2正则

    # prepare for loss_fn
    sample = np.load(config.sample)["sequence_array"]

    print("Start Train.")
    num_epochs = config.num_epochs
    total_step = len(trainDataloader)
    loss_List = []
    loss_test_list = []
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=STEP_SIZE, gamma=GAMMA)  # StepLR
    """
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.8, patience=10,
                                                              verbose=False, threshold=0.0001,
                                                              threshold_mode='rel', cooldown=0,
                                                              min_lr=0, eps=1e-08)  # ReduceLROnPlateau
    """

    # save_path for model in different case
    if case == 0:
        path_m = path + "_n_conv2d.pth"
    elif case == 1:
        path_m = path + "_r_conv2d.pth"
    elif case == 2:
        path_m = path + "_r3_conv2d.pth"
    else:
        path_m = path + "_r4_conv2d.pth"

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=config.patience, verbose=False, path=path_m)

    if os.path.exists(path_m):
        print(path_m)
        model.load_state_dict(torch.load(path_m)["state_dict"])

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        totalLoss = 0  # 计算训练集的平均loss
        for i, (images, target) in enumerate(trainDataloader):
            sam = sample.copy()
            if rotate:
                # print("start rotate!")
                sz = images.shape
                images = images.reshape((-1, sz[3], sz[4]))
                target = target.reshape((-1, sz[3], sz[4]))
                x = np.append(images, target, axis=0)
                x = np.append(x, sam, axis=0)
                x = rotate_sequence(x)

                images, target, sam = x[:sz[0] * sz[1] * sz[2]], x[sz[0] * sz[1] * sz[2]: -1], x[-1]
                images = images.reshape(*sz)
                # print(images.shape)
                target = target.reshape((-1, 1, sz[-2], sz[-1]))
                sam = sam.reshape(-1, sz[-2], sz[-1])
                # print(images.shape, target.shape, sam.shape)

                images = torch.from_numpy(images)
                target = torch.from_numpy(target)
                # print("rotation is finished")
            images = images.to(device)
            target = target.to(device)
            # print("batch 数据集形状", images.shape, target.shape)
            pred = model(images)
            loss = loss_fn(pred, target, sam)
            # loss = loss_fn(pred, target)

            # l_weight = 0.1  # 0.0001, 超参数，惩罚系数。
            # l1_penalty = l_weight * sum([p.abs().sum() for p in model.parameters()])
            # l2_penalty = l_weight * sum([p.square().sum() for p in model.parameters()])
            # loss_with_penalty = loss + l1_penalty  # l2_penalty  #

            # 反向传播
            optimiser.zero_grad()
            loss.backward()
            # loss_with_penalty.backward()  # 加入L1正则化
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)  # 控制梯度爆炸，Shi et al.(2015)
            optimiser.step()

            lr_now = optimiser.state_dict()['param_groups'][0]['lr']

            # lr_scheduler.step(loss)
            lr_scheduler.step()

            # 计算平均loss
            totalLoss = totalLoss + loss.item()

            # 打印结果
            if i % 30 == 0:
                tem = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch + 1, num_epochs,
                                                                                          i + 1, total_step,
                                                                                          totalLoss / (i + 1), lr_now))
        loss_List.append(totalLoss / (i + 1))
        logger.info(
            "Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch + 1, num_epochs, i + 1,
                                                                                total_step, totalLoss / (i + 1),
                                                                                lr_now))

        # 每一次epoch都对测试集进行测试
        model.eval()
        with torch.no_grad():
            loss_t = 0
            for j, (images, target) in enumerate(testDataloader):
                images, target = images.to(device), target.to(device)
                preds = model(images)
                loss_test = loss_fn(preds, target)
                loss_t += loss_test.item()
                loss_prec = loss_fn_p(preds, target, threshold=100).item()
                loss_reca = loss_fn_r(preds, target, threshold=100).item()
        logger.info(
            "Loss in Test dataset: {}, precision: {}, recall: {}".format(loss_t / (j + 1), loss_prec, loss_reca))
        # logger.info("Loss in Test dataset: {}".format(loss_t/(j+1)))

        checkpoint = {
            "state_dict": model.state_dict(),
            "opt_state_dict": optimiser.state_dict(),
            "epoch": epoch
        }
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping.checkpoint = checkpoint
        early_stopping(loss_t / (j + 1), model)

        loss_test_list.append(loss_t / (j + 1))
        print("_" * 10)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    df = pd.DataFrame(data=np.array(loss_List), columns=["loss_train"])
    df["loss_test"] = np.array(loss_test_list)

    if case == 0:
        pathdf = path + "_n_conv2d_loss.xlsx"
    elif case == 1:
        pathdf = path + "_r_conv2d_loss.xlsx"
    elif case == 2:
        pathdf = path + "_r3_conv2d_loss.xlsx"
    else:
        pathdf = path + "_r4_conv2d_loss.xlsx"

    df.to_excel(pathdf)

    plt.figure(figsize=(8, 8))
    plt.plot(loss_List, color='red', linewidth=1.5, linestyle='-', label="loss_train")
    plt.plot(loss_test_list, color='black', linewidth=1.5, linestyle='-', label="loss_test")
    plt.legend(loc="upper right")
    if case == 0:
        pathpic = path + "_n_conv2d_loss.jpg"
    elif case == 1:
        pathpic = path + "_r_conv2d_loss.jpg"
    elif case == 2:
        pathpic = path + "_r3_conv2d_loss.jpg"
    else:
        pathpic = path + "_r4_conv2d_loss.jpg"
    plt.savefig(pathpic, dpi=100)
    plt.close()


if __name__ == "__main__":
    data_i = 4
    name = "me4"
    path = "/hy-tmp/import_datasets/" + name + "/second"
    # path = "C:/Users/chesley/Pictures/ne" + str(data_i) + "/second"
    torch.cuda.empty_cache()
    case = 1  # 1
    train = True
    if train:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        main(path, case, n_train=2000, n_test=1106, dt=500, st=6000)  # 13727 3118 for me1; 15076, 2542 for ne9
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device1 = torch.device("cpu")

        if case == 0:
            pathtx1 = path + "_n_test_x.npz"
            pathty1 = path + "_n_test_y.npz"
        elif case == 1:
            pathtx1 = path + "_r_test_x.npz"
            pathty1 = path + "_r_test_y.npz"
        elif case == 2:
            pathtx1 = path + "_r3_test_x.npz"
            pathty1 = path + "_r3_test_y.npz"
        else:
            pathtx1 = path + "_r4_test_x.npz"
            pathty1 = path + "_r4_test_y.npz"
        test_x = np.load(pathtx1)["sequence_array"]
        test_y = np.load(pathty1)["sequence_array"]
        test_x = torch.from_numpy(test_x).to(device).float()
        test_y = torch.from_numpy(test_y).to(device).float()
        print(test_x.dtype)

        model = EncoderForecaster(config).to(device)

        if case == 0:
            path_m = path + "_n_conv2d.pth"
        elif case == 1:
            path_m = path + "_r_conv2d.pth"
        elif case == 2:
            path_m = path + "_r3_conv2d.pth"
        else:
            path_m = path + "_r4_conv2d.pth"
        print(path_m)
        model.load_state_dict(torch.load(path_m)["state_dict"])

        test_x = test_x[-2:]
        test_y = test_y[-2:]
        y_pre = model(test_x)

        order = ["energy", "energyLg", "lgLgEng"]
        # y_pre, test_y = otherToengLg(y_pre, log=order[data_i-1])
        loss_fn = nn.MSELoss()  # 定义均方差作为损失函数
        loss = loss_fn(y_pre, test_y)
        print(y_pre.shape)
        print("loss: ", loss)

        y_pre = y_pre.cpu().detach().numpy()
        print(np.unique(y_pre))
        # y_pre = y_pre.cpu().detach().numpy()
        # plt.imshow(y_pre[0][0])

        test_y = test_y.cpu().detach().numpy()
        plt.imshow(test_y[0][0])
        plt.show()