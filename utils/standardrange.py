# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : standardrange.py
@ Time    ：2021/8/25 19:43
"""
import numpy as np
from config import config


def pre_to_255(df, log):
    """
    :param df: np.array or pd.DataFrame
    :param log: "energy",
    :return: from pre range to 1-255
    """
    lgengmin, lgengmax = 8.39, 18.46
    engmin, engmax = np.power(10, lgengmin), np.power(10, lgengmax)
    lglgengmin, lglgengmax = np.log10(lgengmin), np.log10(lgengmax)

    if log == "energy":
        df = (df - engmin) / (engmax - engmin) * (255.0 - 1.0) + 1.0
    elif log == "energyLg":
        df = (df - lgengmin) / (lgengmax - lgengmin) * (255.0 - 1.0) + 1.0
    elif log == "lgLgEng":
        df = (df - lglgengmin) / (lglgengmax - lglgengmin) * (255.0 - 1.0) + 1.0
    return df


def from_255_to_pre(df, log):
    """
    :param df:
    :param log:
    :return:
    """
    lgengmin, lgengmax = 8.39, 18.46
    engmin, engmax = np.power(10, lgengmin), np.power(10, lgengmax)
    lglgengmin, lglgengmax = np.log10(lgengmin), np.log10(lgengmax)

    df = (df - 1.0) / (255.0 - 1.0)
    if log == "energy":
        df = df * (engmax - engmin) + engmin
    elif log == "energyLg":
        df = df * (lgengmax - lgengmin) + lgengmin
    elif log == "lgLgEng":
        df = df * (lglgengmax - lglgengmin) + lglgengmin
    return df


def energyLgToOther(df, log):
    """
    :param df: np.array of energy, range in [1, 255]
    :param log:
    :return:
    """
    df = from_255_to_pre(df, "energyLg")
    if log == "energy":
        df = np.power(10, df)
        df = pre_to_255(df, log)
    elif log == "energyLg":
        df = pre_to_255(df, log)
    elif log == "lgLgEng":
        df = np.log10(df)
        df = pre_to_255(df, log)
    return df


def otherToengLg(df, log):
    """
    :param df: np.array of energy, range in [1, 255]
    :param log:
    :return:
    """
    if log == "energy":
        df = from_255_to_pre(df, log)
        df = np.log10(df)
    elif log == "energyLg":
        df = from_255_to_pre(df, log)
    elif log == "lgLgEng":
        df = from_255_to_pre(df, log)
        df = np.power(10, df)
    df = pre_to_255(df, "energyLg")
    return df

def logTomag(df, st_va=config.start_value):
    lgengmin, lgengmax = config.start_mag, 10  # for mag at 4, it is 8.39,
    lgLgNewmin, lgLgNewmax = np.log10(0.5), np.log10(lgengmax - (config.start_mag - 0.5))
    """
    print("The start of mag in figures is: ", st_va)
    df["lgMagNew"] = np.log10(df["MagNew"] - 3.5)
    df["lgMagNewTransform"] = (df["lgMagNew"] - lgLgNewmin) / (lgLgNewmax - lgLgNewmin) * (255.0 - st_va) + st_va
    """
    res = (df - st_va) / (255.0 - st_va) * (lgLgNewmax - lgLgNewmin) + lgLgNewmin  # 转换回df["lgMagNew"]
    res = np.power(10, res) + (config.start_mag - 0.5)  # MagNew

    res = np.where(df >= st_va, res, 0)
    return res

def mag_to_pixel(x, st_va=config.start_value):
    if x < config.start_mag:
        return 0
    lgengmin, lgengmax = config.start_mag, 10  # for mag at 4, it is 8.39,
    lgLgNewmin, lgLgNewmax = np.log10(0.5), np.log10(lgengmax - (config.start_mag - 0.5))
    x = np.log10(x - (config.start_mag - 0.5))
    x = (x - lgLgNewmin) / (lgLgNewmax - lgLgNewmin) * (255.0 - st_va) + st_va
    return x


def pixel_to_mag(x, st_va=config.start_value, _np=False):
    if not _np:
        if x < st_va:
            return 0
    lgengmin, lgengmax = config.start_mag, 10  # for mag at 4, it is 8.39,
    lgLgNewmin, lgLgNewmax = np.log10(0.5), np.log10(lgengmax - (config.start_mag - 0.5))
    x = (x - st_va) / (255.0 - st_va) * (lgLgNewmax - lgLgNewmin) + lgLgNewmin
    x = np.power(10, x) + (config.start_mag - 0.5)

    if _np:
        x = np.where(x >= 0, x, 0)

    return x
