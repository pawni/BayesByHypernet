import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import time
from observations import cifar10

from sklearn.calibration import calibration_curve

try:
    import cPickle as pickle
except Exception as e:
    import pickle


(x_train, y_train), (x_test, y_test) = cifar10('/data/np716/cifar_10/')

x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)

x_train_first = x_train[y_train < 5]
y_train_first = y_train[y_train < 5]

x_test_first = x_test[y_test < 5]
y_test_first = y_test[y_test < 5]

x_test_outlier = x_test[y_test >= 5]


# helper for rotation
def rotate(img, angle):
    img = scipy.ndimage.rotate(img.reshape((28, 28)), angle, reshape=False)
    return img.reshape((-1))


max_ent = np.sum(-1 * (np.ones(5) / 5.) * np.log((np.ones(5) / 5.)), -1)


def get_pred_df(data, session, ops, mode):
    cols = ['prob', 'Prediction', 'sample_idx', 'unit']
    df = pd.DataFrame(columns=cols)

    probs = get_probs(data, session, ops, mode)

    for sample_idx in range(probs.shape[1]):  # per data sample
        for class_idx in range(10):  # per class ...
            data = zip(
                probs[:, sample_idx, class_idx],
                [class_idx] * len(probs),
                [sample_idx] * len(probs),
                list(range(len(probs)))
            )
            new_df = pd.DataFrame(columns=cols, data=data)
            df = pd.concat([df, new_df])

    return df


def get_probs(data_inp, session, ops, mode, is_eval=True):
    if mode == 'ensemble':
        probs = np.stack([
            session.run(prob, feed_dict={
                ops['x']: data_inp, ops['is_eval']: is_eval})
            for prob in ops['probs']])
    elif mode == 'map' or mode == 'mle':
        probs = session.run(ops['probs'], feed_dict={
            ops['x']: data_inp, ops['is_eval']: is_eval})
        probs = probs[np.newaxis, :]
    else:
        probs = np.zeros((100, len(data_inp), 5))  # ensemble, data, classes
        batch_size = 1000
        for b in range(len(data_inp) // batch_size):
            start = b * batch_size
            end = start + batch_size
            for i in range(100):
                probs[i, start:end] += session.run(
                    ops['probs'], feed_dict={
                        ops['x']: data_inp[start:end], ops['is_eval']: is_eval})
        end = (len(data_inp) // batch_size) * batch_size
        if end < len(data_inp):
            start = end
            for i in range(100):
                probs[i, start:] += session.run(
                    ops['probs'], feed_dict={
                        ops['x']: data_inp[start:], ops['is_eval']: is_eval})
    return probs


def build_adv_examples(images, labels, eps, session, ops, mode):
    feed_dict = {ops['x']: images, ops['y']: labels, ops['adv_eps']: eps,
                 ops['is_eval']: True}

    if mode == 'ensemble':
        adv_data = np.mean(np.stack([
            session.run(ad, feed_dict=feed_dict)
            for ad in ops['adv_data']]), 0)
    elif mode == 'map' or mode == 'mle':
        adv_data = session.run(ops['adv_data'], feed_dict=feed_dict)
    else:
        adv_data = session.run(ops['adv_data'], feed_dict=feed_dict) / 100
        for i in range(99):
            adv_data += session.run(ops['adv_data'], feed_dict=feed_dict) / 100
    return adv_data


def calc_entropy(probs):  # shape = [sample, classes]
    return np.sum(-1 * probs * np.log(np.maximum(probs, 1e-5)), -1)


def calc_ent_auc(ent):
    hist, bin_edges = np.histogram(ent, density=True,
                                   bins=np.arange(0, max_ent, max_ent / 500))
    c_hist = np.cumsum(hist * np.diff(bin_edges))

    return np.sum(np.diff(bin_edges) * c_hist)


def build_result_dict(session, ops, mode):
    result_dict = {}

    # calc test acc:
    probs = get_probs(x_test_first, session, ops, mode)
    mean_probs = probs.mean(0)
    test_acc = np.mean(np.argmax(mean_probs, -1) == y_test_first)
    test_entropy = calc_entropy(mean_probs)
    test_ent_auc = calc_ent_auc(test_entropy)

    test_cal_pos, test_cal_bins = calibration_curve(
        np.ones(len(mean_probs)),
        mean_probs[np.arange(len(mean_probs)), y_test_first],
        normalize=False, n_bins=50)

    result_dict['test_acc'] = test_acc
    result_dict['test_ent_auc'] = test_ent_auc
    result_dict['test_entropy'] = test_entropy
    result_dict['test_cal_pos'] = test_cal_pos
    result_dict['test_cal_bins'] = test_cal_bins

    # not mnist entropy
    probs = get_probs(x_test_outlier, session, ops, mode)
    mean_probs = probs.mean(0)
    outlier_entropy = calc_entropy(mean_probs)
    outlier_ent_auc = calc_ent_auc(outlier_entropy)

    result_dict['outlier_entropy'] = outlier_entropy
    result_dict['outlier_ent_auc'] = outlier_ent_auc

    # build adv examples and test performance
    adv_df = pd.DataFrame(columns=['eps', 'acc', 'ent', 'ent_auc'])
    result_dict['adv_examples'] = {}
    for eps in np.linspace(0., 0.4, num=9):
        adv_data = build_adv_examples(x_test_first[:100], y_test_first[:100],
                                      eps, session, ops, mode)
        result_dict['adv_examples'][eps] = adv_data

        adv_data = np.pad(adv_data, ((0, 0), (4, 4), (4, 4), (0, 0)),
                          'constant')

        adv_probs = get_probs(adv_data, session, ops, mode)
        mean_adv_probs = adv_probs.mean(0)
        adv_acc = np.mean(np.argmax(mean_adv_probs, -1) == y_test_first[:100])
        adv_entropy = calc_entropy(mean_adv_probs)
        adv_ent_auc = calc_ent_auc(adv_entropy)
        adv_df.loc[len(adv_df)] = [eps, adv_acc, adv_entropy.mean(),
                                   adv_ent_auc]

    result_dict['adv_df'] = adv_df

    return result_dict
