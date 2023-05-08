# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : config.py
@ Time    ：2021/5/7 20:36
"""
# from model import ConvLstm
# odelForModify import ConvLstm
from model_parallel import ConvLstm
from collections import OrderedDict
import torch
import numpy as np

class Config():
    def __init__(self, seq_in, seq_out):
        """
        多显卡训练
        :param seq_in:
        :param seq_out:
        """
        # (type, activation, in_ch, out_ch, kernel_size, padding, stride)
        self.sequences_in = seq_in
        self.sequences_out = seq_out
        self.b = 1920  # 960  # 960  # 19CX  # 480
        self.width = 2 * self.b  # 3840  # 1920  # 960  # 1000  # 3000  # 160  # 480 # 3000  #3840  #
        self.height = 2 * self.b  # 960  # 3840  # 960  # 1000  # 3000  # 160  # 480 # 3000  #3840  #

        self.pic = seq_in  # 预测图片的个数
        self.dtday = 30
        self.percent = 0.7  # 0.6

        self.batch = 1  # 20  # 3  # 200  # 16  # 16  # 500  # 2  #
        self.batch1 = 500
        self.batch2 = 200

        self.time_start = "1638_01"  # "1638_01"
        self.time_end = "2021_04"
        self.path = "C:/Users/chesley/Pictures/npz/sequence_array.npz"

        self.num_workers = 0  # 多线程/ windows必须设置为0
        self.num_epochs = 40  # 20  # 3  # 40  # 训练次数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare for earlystopping
        self.patience = 3  # 3  # 7

        # prepare for weighted loss
        self.weight = 0.9998  # 0.8  # 0.95  # 0.65
        self.weights_step = (0.0002, 0.3, 0.6998)
        self.threshold = 0.5  # 0  # 海岸线背景是0-10, 因此设置为15

        self.half = False

        self.start_mag = 4.0
        self.start_value = 100.0  # 地震在背景图片中的起始点
        self.target_csv = "csd/"

        # prepare for weightedMseMae in Shi et al. (2015)
        """
        self.sample = "C:/Users/chesley/Pictures/earthquake_zone/fault/fault_" + str(
            self.height) + "_0" + "_judge.npz"

        self.sample = "C:/Users/chesley/Pictures/earthquake_zone/sample_" + str(
            self.height) + "_1" + "_collection_mag" + str(int(self.start_mag)) + ".npz"

        self.sample = "C:/Users/chesley/Pictures/earthquake_zone/mix/mix_" + str(
            self.height) + "_0" + "_judge_mag4.npz"
        """
        self.sample = "/hy-tmp/import_datasets/earthquake_zone/sample_" + str(
            self.height) + "_6" + "_collection_mag" + str(int(self.start_mag)) + ".npz"

        # self.layerAccess = ([207, 270, 393], [1526, 1649, 1712])
        self.layerAccess = ([427, 502, 611, 786, 1133], [2706, 3053, 3228, 3337, 3412])
        self.dx = 1
        self.dy = 6  # 3 # 6  #

        # self.layerAccess = ([427, 502, 611, 786, 1133], [2706, 3053, 3228, 3337, 3412])
        # ([427, 502, 611, 786, 1133], [2706, 3053, 3228, 3337, 3412])
        # ([296, 329, 372, 427, 502, 611, 786, 1133], [2706, 3053, 3228, 3337, 3412, 3467, 3510, 3543])
        # self.layerAccess = ([225, 491], [1428, 1694])

        self.balancing_weights = [0.000177210579159563, 0.01,
                                  1.0]  # 0.00017721057915956307  # [0.00011001237265819627, 0.001, 1.0]  #  # [0.00023837162682711961, 0.002, 1.0]  # [0.00040179490616810184, 0.004, 1.0]  # [0.00040179490616810184, 0.004, 1.0, 10, 30, 50, 100]
        self.THRESHOLDS = np.array([4])  # np.array([4, 5, 6, 7, 8])  # int(self.start_mag)
        self.mse_mae_weights = (1.0, 1.0)
        self.normal_loss_global_scale = 1 / 1875  # 300000  #

        self.pre_parameters = (0.0003, 20000, 0.7)  # (0.001, 10000, 0.9)

        # save for n in shi et al. (2015)  # used in p915single
        self.encoder = [[OrderedDict({'conv1_leaky_1': [2, 8, 7, 5, 3]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 4, 2]}),
                         OrderedDict({'conv3_leaky_1': [192, 192, 3, 1, 1]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3)]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [192, 192, 3, 1, 1]}),
                         OrderedDict({'deconv2_leaky_1': [192, 64, 4, 4, 0]}),
                         OrderedDict({'deconv3_relu_1': [64, 8, 7, 5, 1],
                                      'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                  Sequences=self.sequences_out, kernel_size=3)]]
        """
        # cnn-10;
        # suppose the width=160
        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 4, 2, 1]}),
                         OrderedDict({'conv3_leaky_1': [192, 256, 4, 2, 1]}),
                         OrderedDict({'conv4_leaky_1': [256, 512, 3, 1, 1]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=512, hidden_channels=512,
                                  Sequences=self.sequences_in, kernel_size=3)
                         ]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [512, 256, 3, 1, 1]}),
                         OrderedDict({'deconv2_leaky_1': [256, 192, 4, 2, 1]}),
                         OrderedDict({'deconv3_leaky_1': [192, 64, 4, 2, 1]}),
                         OrderedDict({'deconv4_relu_1': [64, 8, 7, 5, 1],
                                      # 'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=512, hidden_channels=512,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                  Sequences=self.sequences_out, kernel_size=3)]]
        """


        """
        # cnn-7;
        # suppose the width=960
        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
                         OrderedDict({'conv3_leaky_1': [192, 256, 3, 2, 1]}),
                         OrderedDict({'conv4_leaky_1': [256, 512, 3, 2, 1]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=512, hidden_channels=512,
                                  Sequences=self.sequences_in, kernel_size=3)
                         ]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [512, 256, 4, 2, 1]}),
                         OrderedDict({'deconv2_leaky_1': [256, 192, 4, 2, 1]}),
                         OrderedDict({'deconv3_leaky_1': [192, 64, 5, 3, 1]}),
                         OrderedDict({'deconv4_relu_1': [64, 8, 7, 5, 1],
                                      # 'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=512, hidden_channels=512,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                  Sequences=self.sequences_out, kernel_size=3)]]
        """

        """
        # cnn-5;
        # suppose the width=1000
        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 3]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
                         OrderedDict({'conv3_leaky_1': [192, 256, 3, 2, 1]}),
                         OrderedDict({'conv4_leaky_1': [256, 256, 3, 2, 1]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_in, kernel_size=3)
                         ]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
                         OrderedDict({'deconv2_leaky_1': [256, 192, 4, 2, 1]}),
                         OrderedDict({'deconv3_leaky_1': [192, 64, 4, 2, 1]}),
                         OrderedDict({'deconv4_relu_1': [64, 8, 7, 5, 1],
                                      # 'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=256, hidden_channels=256,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                  Sequences=self.sequences_out, kernel_size=3)]]
        """

        """
        # save for n in cnn-4 model
        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 3]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 4, 2]}),
                         OrderedDict({'conv3_leaky_1': [192, 192, 3, 1, 1]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                       Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_in, kernel_size=3)]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [192, 192, 3, 1, 1]}),
                         OrderedDict({'deconv2_leaky_1': [192, 64, 4, 4, 0]}),
                         OrderedDict({'deconv3_leaky_1': [64, 1, 7, 5, 1],
                                      # 'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      # 'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                       Sequences=self.sequences_out, kernel_size=3)]]
        """

        """
        # save for r in original model
        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 3]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 4, 2]}),
                         OrderedDict({'conv3_leaky_1': [192, 192, 3, 1, 1]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                       Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_in, kernel_size=3)]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [192, 192, 3, 1, 1]}),
                         OrderedDict({'deconv2_leaky_1': [192, 64, 4, 4, 0]}),
                         OrderedDict({'deconv3_relu_1': [64, 8, 7, 5, 1],
                                      # 'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                       Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                       Sequences=self.sequences_out, kernel_size=3)]]

        """

        """

        # save for r in cnn-8
        self.encoder = [[OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 3]}),
                         OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 2]}),
                         OrderedDict({'conv3_leaky_1': [192, 512, 5, 4, 2]})],
                        [ConvLstm(input_channels=8, hidden_channels=64,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_in, kernel_size=3),
                         ConvLstm(input_channels=512, hidden_channels=512,
                                  Sequences=self.sequences_in, kernel_size=3)]]
        self.decoder = [[OrderedDict({'deconv1_leaky_1': [512, 192, 4, 4, 0]}),
                         OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
                         OrderedDict({'deconv3_relu_1': [64, 8, 7, 5, 1],
                                      # 'conv3_leaky_2': [8, 8, 3, 1, 1],
                                      'conv3_3': [8, 1, 1, 1, 0]
                                      })],
                        [ConvLstm(input_channels=512, hidden_channels=512,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=192, hidden_channels=192,
                                  Sequences=self.sequences_out, kernel_size=3),
                         ConvLstm(input_channels=64, hidden_channels=64,
                                  Sequences=self.sequences_out, kernel_size=3)]]
        """


config = Config(seq_in=9, seq_out=1)
# print(config.encoder)
# print(config.decoder)
# print(config.sequences)
# print(config.time_start, config.time_end)
print(config.batch, config.sequences_in, config.sequences_out, config.percent)