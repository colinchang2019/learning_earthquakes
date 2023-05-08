# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : precisionAccessForMulti.py
@ Time    ：2021/10/12 10:40
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from config import config as configC
from utils.loss import weightedMse, weightedMseD
from modelForModify import EncoderForecaster
from utils.assesment import *
from utils.dataset import MultiDataset
from utils.spacelayer import spacelayer


def validate(model, config, path, case, los, threholldList, pathout, n_test=2700,st=0, dt=500, mrecall=False, mprecision=False, d=1):
    """
    :param model:
    :param config:
    :param path: for dataset and pth file
    :param case: for two kind of dataset
    :param los: for loss in training
    :return: [(loss, precision, recall, accuracy)]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("run file in: ", path)

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
    testDataset = MultiDataset(path_fa, n_test, dt, case_str, "test", st)
    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.num_workers
    )
    print("Dataset prepared.")

    model.eval()
    loss_fn = nn.MSELoss()
    if los == "w":
        loss_fn = weightedMse()
    elif los == "D":
        loss_fn = weightedMseD()
    loss_fn_p = precision()
    loss_fn_r = recall()
    loss_fn_a = accuracy()
    loss_fn_tpr = TPR()
    loss_fn_fpr = FPR()

    loss_fn_csi = CSI()
    loss_fn_far = FAR()
    loss_fn_pod = POD()
    loss_fn_rscore = RScore()
    loss_fn_mcc = mcc()


    with torch.no_grad():
        loss_t, loss_p, loss_r, loss_a, loss_rscore = 0, 0, 0, 0, 0
        n = len(testDataloader)
        for j, (images, target) in enumerate(testDataloader):
            images, target = images.to(device), target.to(device)
            tem = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print("Time: {}, process {}/{}".format(tem, j + 1, n))
            with autocast():
                preds = model(images)
                loss_test = loss_fn(preds, target)
                loss_t += loss_test.item()
                print(torch.max(preds * 255.0))
                if mrecall:
                    preds = preds.cpu().detach().numpy()
                    preds = spacelayer(preds, d=d)
                    preds = torch.from_numpy(preds).to(device)

                if mprecision:
                    target = target.cpu().detach().numpy()
                    target = spacelayer(target, d=d)
                    target = torch.from_numpy(target).to(device)

                #preds[:, :, config.layerAccess[1][-1] + 1:, :] = 0
                #preds[:, :, :config.layerAccess[0][0], :] = 0
                #target[:, :, config.layerAccess[1][-1] + 1:, :] = 0
                #target[:, :, :config.layerAccess[0][0], :] = 0

                loss_test = np.array([[loss_fn_p(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_p = loss_test
                else:
                    loss_p = np.concatenate((loss_p, loss_test), axis=0)

                loss_test = np.array([[loss_fn_r(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_r = loss_test
                else:
                    loss_r = np.concatenate((loss_r, loss_test), axis=0)

                loss_test = np.array([[loss_fn_a(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_a = loss_test
                else:
                    loss_a = np.concatenate((loss_a, loss_test), axis=0)
                """
                loss_test = np.array([[loss_fn_tpr(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_tpr = loss_test
                else:
                    loss_tpr = np.concatenate((loss_tpr, loss_test), axis=0)

                loss_test = np.array([[loss_fn_fpr(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_fpr = loss_test
                else:
                    loss_fpr = np.concatenate((loss_fpr, loss_test), axis=0)
                
                loss_test = np.array([[loss_fn_csi(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_csi = loss_test
                else:
                    loss_csi = np.concatenate((loss_fpr, loss_test), axis=0)

                loss_test = np.array([[loss_fn_far(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_far = loss_test
                else:
                    loss_far = np.concatenate((loss_fpr, loss_test), axis=0)

                loss_test = np.array([[loss_fn_pod(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_pod = loss_test
                else:
                    loss_pod = np.concatenate((loss_fpr, loss_test), axis=0)

                loss_test = np.array([[loss_fn_rscore(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_rscore = loss_test
                else:
                    loss_rscore = np.concatenate((loss_fpr, loss_test), axis=0)
                """
                loss_test = np.array([[loss_fn_rscore(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_rscore = loss_test
                else:
                    loss_rscore = np.concatenate((loss_rscore, loss_test), axis=0)

                loss_test = np.array([[loss_fn_mcc(preds, target, threshold=x).item() for x in threholldList]])
                if j == 0:
                    loss_mcc = loss_test
                else:
                    loss_mcc = np.concatenate((loss_mcc, loss_test), axis=0)

    print("Loss in Test dataset: {}".format(loss_t / (j + 1)))
    n = j + 1
    loss_p = np.mean(loss_p, axis=0)
    loss_r = np.mean(loss_r, axis=0)
    loss_a = np.mean(loss_a, axis=0)
    #loss_tpr = np.mean(loss_tpr, axis=0)
    #loss_fpr = np.mean(loss_fpr, axis=0)

    #loss_csi = np.mean(loss_csi, axis=0)
    #loss_far = np.mean(loss_far, axis=0)
    #loss_pod = np.mean(loss_pod, axis=0)
    loss_rscore = np.mean(loss_rscore, axis=0)
    loss_mcc = np.mean(loss_mcc, axis=0)

    full = np.array([loss_p, loss_r, loss_a, loss_rscore, loss_mcc])  # , loss_tpr, loss_fpr, loss_csi, loss_far, loss_pod, loss_rscore
    fullt = full.T
    df = pd.DataFrame(fullt)
    df.columns = ["precision", "recall", "accuracy", "rscore", "mcc"]  # , "tpr", "fpr", "csi", "far", "pod", "rscore"
    df["threshold"] = thresholdList
    # df["mag"] = logTomag(thresholdList)
    df.to_csv(pathout)

    """
    plt.scatter(loss_fpr, loss_tpr, label="roc_cnn_r")

    plt.legend()

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()
    """
    return loss_t / n


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # path = 'C:/Users/chesley/Pictures/npe2/second'
    # path = "C:/Users/chesley/Pictures/npp/second"
    # path_m = "C:/Users/chesley/Pictures/npe2/" + "save_cnn-4/second_n_conv2d.pth"
    # path_m = "C:/Users/chesley/Pictures/npp/" + "save_s2cnn-7/second_n_so3.pth"
    # path_m = "C:/Users/chesley/Pictures/npp/" + "save_s2cnn-8/second_r_so3.pth"
    # model = SO3EncoderForcaster(config=configS).to(device)
    path = "/hy-tmp/import_datasets/p9d/second"  # p915single/casen
    # 0.001
    case = 0
    if case == 0:
        path_m = path + "_n_conv2d.pth"
        pathout = path + "_n_conv2d_roc_new.csv"
    else:
        path_m = path + "_r_conv2d.pth"
        pathout = path + "_r_conv2d_roc_new.csv"
    model = EncoderForecaster(configC).to(device)
    model.load_state_dict(torch.load(path_m)["state_dict"])
    # thresholdLiist = np.array([i for i in range(101, 201, 1)]) / 10
    thresholdList = np.array([i for i in range(0, 255, 50)])  # / 100  range(0, 25501, 50)
    # thresholdLiist = np.array([i for i in range(11, 101, 1)])
    # pathout = "C:/Users/chesley/Pictures/npe2/" + "save_cnn-4/second_n_conv2d_roc.csv"
    # pathout = "C:/Users/chesley/Pictures/npp/" + "save_s2cnn-7/second_n_so3_roc.csv"
    # pathout = "C:/Users/chesley/Pictures/npp/" + "save_s2cnn-8/second_r_so3_roc.csv"
    res = validate(model, config=configC, path=path, case=0, los="D", threholldList=thresholdList, pathout=pathout,
                   n_test=806, st=0, dt=1, mrecall=True, mprecision=False, d=3)
    print(res)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Finish")