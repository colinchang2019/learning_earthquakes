# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : spacelayer.py
@ Time    ：2022/9/19 10:47
"""
import numpy as np
import torch
from config import config
import matplotlib.pyplot as plt


def spacelayer(x, d=3):
    number, _0, height, width = x.shape
    x = x.reshape((-1, height, width))
    xnew = x.copy()
    dy = 2 * d  # config.dy  #
    ss, ns = config.layerAccess
    xnew = spacemini(x, ss[-1], ns[0], xnew, d, dy)
    ds = [d + _ for _ in range(1, len(ss))]
    for i, dx in enumerate(ds):
        xnew = spacemini(x, ss[-i-1], ss[-i-2], xnew, dx, dy)
        xnew = spacemini(x, ns[i], ns[i+1], xnew, dx, dy)
    # xnew[:, ns[-1]:, :] = x[:, ns[-1]:, :]
    # xnew[:, :ss[0], :] = x[:, :ss[0], :]
    #xnew = spacemini(x, 0, ss[0], xnew, 3, 6)
    #xnew = spacemini(x, ns[-1], 3839, xnew, 3, 6)

    xnew = xnew.reshape((number, _0, height, width))
    return xnew


def spacemini(x, idxmin, idxmax, xnew, dx, dy):
    _01, height, width = x.shape
    x1 = np.zeros_like(x)
    print(x1.shape, dx, dy, np.max(x1))
    x1[:, idxmin:idxmax, :] = x[:, idxmin:idxmax, :]
    indexs = np.argwhere(x1 > 0.0)
    indexs = sorted(indexs, key=lambda _1: x1[_1[0], _1[1], _1[2]])  # 排序一下
    n = height

    for _1 in indexs:
        _n, x_, y_ = _1[0], _1[1], _1[2]
        left_x, left_y = x_ - dx if x_ - dx >= idxmin else idxmin, y_ - dy if y_ - dy >= 0 else 0
        right_x, right_y = x_ + dx + 1 if x_ + dx + 1 <= n else n, \
                           y_ + dy + 1 if y_ + dy + 1 <= n else n
        xnew[_n, left_x:right_x, left_y:right_y] = x[_1[0], _1[1], _1[2]]
    """
        for _1 in indexs:
        _n, x_, y_ = _1[0], _1[1], _1[2]
        left_x, left_y = x_ - dx if x_ - dx >= 0 else 0, y_ - dy if y_ - dy >= 0 else 0
        right_x, right_y = x_ + dx + 1 if x_ + dx + 1 <= n else n, \
                           y_ + dy + 1 if y_ + dy + 1 <= n else n
        xnew[_n, left_x:right_x, left_y:right_y] = x[_1[0], _1[1], _1[2]]
    """
    return xnew


def test(path, case, log, _c=90):
    torch.cuda.empty_cache()  # 清理显存
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    case_str = ["_n", "_r", "_r3", "_r4"]
    st = "_half_" + str(_c)
    pathtx1 = path + st + case_str[case] + "_test_x.npz"
    pathty1 = path + st + case_str[case] + "_test_y.npz"
    test_x = np.load(pathtx1)["sequence_array"]
    test_y = np.load(pathty1)["sequence_array"]
    print(test_x.shape, test_y.shape)

    test_y = spacelayer(test_y, d=3)
    res = test_y[0][0]

    res = 255.0 - res * 255.0

    # reverse back
    res[:, :] = res[::-1, :]

    plt.imshow(res, cmap="gray")

    plt.axis("off")
    pathfig = "C:/Users/chesley/Pictures/remag_b/view/sample_space_layer" + log + ".jpg"
    plt.savefig(pathfig, dpi=600, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    path = "C:/Users/chesley/Pictures/remag_b/p915single/casen/second"
    test(path, case=0, log="mag4_p915single_reald3", _c=805)