import os
import numpy as np
from scipy.stats import wasserstein_distance
import tensorflow as tf
from scipy import integrate, signal

from config import DefaultConfig
from statsmodels.tsa.seasonal import seasonal_decompose  #SEATS分解

config = DefaultConfig()
window = signal.windows.gaussian(5, std=1)
window_1 = signal.windows.gaussian(3, std=1)


def gaussian_filter(x):
    # print(window)
    filter_x = np.zeros_like(x)
    filter_x[0] = (x[0] * window_1[1] + x[1] * window_1[2]) / (window_1[1] + window_1[2])
    filter_x[1] = (x[0] * window[1] + x[1] * window[2] + x[2] * window[3] + x[3] * window[4]) / \
                  (window[1] + window[2] + window[3] + window[4])
    filter_x[-2] = (x[-4] * window[0] + x[-3] * window[1] + x[-2] * window[2] + x[-1] * window[3]) / \
                   (window[0] + window[1] + window[2] + window[3])
    filter_x[-1] = (x[-2] * window_1[0] + x[-1] * window_1[1]) / (window_1[0] + window_1[1])
    for i in range(2, len(x) - 2):
        front_1 = x[i - 2]
        front_2 = x[i - 1]
        back_1 = x[i + 1]
        back_2 = x[i + 2]
        filter_x[i] = (front_1 * window[0] + front_2 * window[1] + x[i] * window[2] + back_1 * window[3]
                       + back_2 * window[4]) / np.sum(window)
    return filter_x


def seats(x, y):
    timeseries = np.hstack((y, x))
    seats = seasonal_decompose(timeseries, model="additive", period=288, extrapolate_trend="freq")
    return seats.trend, seats.resid, seats.seasonal


def cal_normal_pdf(res, std=0.1):
    # return 1 / (std * (2. * np.pi) ** 0.5) * tf.math.exp(- (sample - mean) ** 2 / (2 * std ** 2))

    # remove_noise = lambda x: 1 if x >= 0.6 else x
    # cut_noise = lambda x: 1 if x >= 0.80 else x
    temp_score = tf.math.exp(- res ** 2 / (2 * std ** 2)).numpy()
    # temp_score = np.array([cut_noise(x) for x in temp_score])
    # for i in range(1, len(temp_score) - 1):
    #     front = temp_score[i - 1]
    #     back = temp_score[i + 1]
    #     if front >= 0.8 and back >= 0.8 and temp_score[i] >= 0.6:
    #         temp_score[i] = 1
    return temp_score


def distance_score(ori, rec):
    trend, res, seasonal = seats(ori, rec)
    cut_noise = lambda x: 0 if -0.05 < x < 0.05 else x
    res_cutted = np.array([cut_noise(x) for x in res[288:]])
    new_ori = gaussian_filter(trend[288:] + seasonal[288:] + res_cutted)
    # new_ori = trend[288:] + seasonal[288:] + res_cutted

    # score = dis.minkowski(new_ori, rec, p=2)

    # score = wasserstein_distance_function(new_ori, rec)

    # plot_ex(new_ori, rec, machine_id, kpi, day)

    # l2_kpi_1 = np.linalg.norm(new_ori, ord=2)
    # l2_kpi_2 = np.linalg.norm(rec, ord=2)
    # # get the cross-correlation of each shift `s`
    # CC = np.correlate(new_ori, rec, mode='full')
    # # return the SBD between kpi_1 and kpi_2
    # score =  1 - (np.max(CC) / (l2_kpi_1 * l2_kpi_2))

    new_res = new_ori - rec
    gaussian_score = 1 - cal_normal_pdf(new_res)
    score = integrate.trapz(gaussian_score, [i for i in range(config.input_size)])

    return score
