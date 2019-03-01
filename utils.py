import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import time

from sklearn.calibration import calibration_curve

try:
    import cPickle as pickle
except Exception as e:
    import pickle


# read mnist data
mnist = input_data.read_data_sets('/vol/biomedic/users/np716/data/mnist')  # put data path
# here
not_mnist = input_data.read_data_sets(
    '/vol/biomedic/users/np716/data/notMNIST_real/notMNIST-to-MNIST-master/')  # put data
# path here

swelling_sets = []
for s in np.arange(3, 12):
    swelling_sets.append(input_data.read_data_sets(
        '/vol/biomedic/users/np716/data/morphomnist/swelling_r{}_s3/'.format(s)))

# helper for rotation
def rotate(img, angle):
    img = scipy.ndimage.rotate(img.reshape((28, 28)), angle, reshape=False)
    return img.reshape((-1))


# generate set of rotated three's
rot_three_img = np.array(
    [rotate(mnist.test.images[270], rot * 10) for rot in range(10)])
rot_pos_three_img = np.array(
    [rotate(mnist.test.images[270], rot * -10) for rot in range(10)])

# generate set of rotated one's
rot_one_img = np.array(
    [rotate(mnist.test.images[202], rot * 10) for rot in range(10)])
rot_pos_one_img = np.array(
    [rotate(mnist.test.images[202], rot * -10) for rot in range(10)])

# generate set of rotated six's
rot_six_img = np.array(
    [rotate(mnist.test.images[217], rot * 10) for rot in range(19)])
rot_pos_six_img = np.array(
    [rotate(mnist.test.images[217], rot * -10) for rot in range(19)])

# generate mixup
mixup_three_eight_img = np.array([(l / 10.) * mnist.test.images[391]
                                  + (1 - l / 10.) * mnist.test.images[270]
                                  for l in range(11)])

max_ent = np.sum(-1 * (np.ones(10) / 10.) * np.log((np.ones(10) / 10.)), -1)


def get_pred_df(data, session, ops, mode):
    cols = ['prob', 'Prediction', 'sample_idx', 'unit']
    df = pd.DataFrame(columns=cols)

    probs = get_probs(data, session, ops, mode)

    for sample_idx in range(probs.shape[1]):  # per data sample
        for class_idx in range(10):  # per class ...
            data = list(zip(
                probs[:, sample_idx, class_idx],
                [class_idx] * len(probs),
                [sample_idx] * len(probs),
                list(range(len(probs)))
            ))
            new_df = pd.DataFrame(columns=cols, data=data)
            df = pd.concat([df, new_df])

    return df


def get_probs(data_inp, session, ops, mode):
    if mode == 'ensemble':
        probs = np.stack([
            session.run(prob, feed_dict={ops['x']: data_inp})
            for prob in ops['probs']
        ])
    elif mode == 'map' or mode == 'mle':
        probs = session.run(ops['probs'], feed_dict={ops['x']: data_inp})
        probs = probs[np.newaxis, :]
    else:
        probs = np.zeros((100, len(data_inp), 10))  # ensemble, data, classes
        batch_size = 1000
        for b in range(len(data_inp) // batch_size):
            start = b * batch_size
            end = start + batch_size
            for i in range(100):
                probs[i, start:end] += session.run(
                    ops['probs'], feed_dict={ops['x']: data_inp[start:end]})
        end = (len(data_inp) // batch_size) * batch_size
        if end < len(data_inp):
            start = end
            for i in range(100):
                probs[i, start:] += session.run(
                    ops['probs'], feed_dict={ops['x']: data_inp[start:]})
    return probs


def build_adv_examples(images, labels, eps, session, ops, mode):
    feed_dict = {ops['x']: images, ops['y']: labels, ops['adv_eps']: eps}

    if mode == 'ensemble':
        adv_data = np.mean(session.run(ops['adv_data'], feed_dict=feed_dict), 0)
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
    probs = get_probs(mnist.test.images, session, ops, mode)
    mean_probs = probs.mean(0)
    test_acc = np.mean(np.argmax(mean_probs, -1) == mnist.test.labels)
    test_entropy = calc_entropy(mean_probs)
    test_ent_auc = calc_ent_auc(test_entropy)

    test_cal_pos, test_cal_bins = calibration_curve(
        np.ones(len(mean_probs)),
        mean_probs[np.arange(len(mean_probs)), mnist.test.labels],
        normalize=False, n_bins=50)

    result_dict['mean_probs'] = mean_probs

    result_dict['test_acc'] = test_acc
    result_dict['test_ent_auc'] = test_ent_auc
    result_dict['test_entropy'] = test_entropy
    result_dict['test_cal_pos'] = test_cal_pos
    result_dict['test_cal_bins'] = test_cal_bins

    # not mnist entropy
    probs = get_probs(not_mnist.test.images, session, ops, mode)
    mean_probs = probs.mean(0)
    not_mnist_entropy = calc_entropy(mean_probs)
    not_mnist_ent_auc = calc_ent_auc(not_mnist_entropy)

    result_dict['not_mnist_mean_probs'] = mean_probs
    result_dict['not_mnist_entropy'] = not_mnist_entropy
    result_dict['not_mnist_ent_auc'] = not_mnist_ent_auc

    # build adv examples and test performance
    adv_df = pd.DataFrame(columns=['eps', 'acc', 'ent', 'ent_auc'])
    result_dict['adv_examples'] = {}
    for eps in np.linspace(0., 0.4, num=9):
        adv_data = build_adv_examples(mnist.test.images[:100],
                                      mnist.test.labels[:100],
                                      eps, session, ops, mode)
        result_dict['adv_examples'][eps] = adv_data
        adv_probs = get_probs(adv_data, session, ops, mode)
        mean_adv_probs = adv_probs.mean(0)
        adv_acc = np.mean(
            np.argmax(mean_adv_probs, -1) == mnist.test.labels[:100])
        adv_entropy = calc_entropy(mean_adv_probs)
        adv_ent_auc = calc_ent_auc(adv_entropy)
        adv_df.loc[len(adv_df)] = [eps, adv_acc, adv_entropy.mean(),
                                   adv_ent_auc]

    result_dict['adv_df'] = adv_df

    swelling_df = pd.DataFrame(columns=['swelling', 'acc', 'ent', 'ent_auc'])
    for i, s in enumerate(np.arange(3, 12)):
        cur_data = swelling_sets[i]
        sw_probs = get_probs(cur_data.test.images, session, ops, mode)
        mean_sw_probs = sw_probs.mean(0)
        sw_acc = np.mean(
            np.argmax(mean_sw_probs, -1) == mnist.test.labels)
        sw_entropy = calc_entropy(mean_sw_probs)
        sw_ent_auc = calc_ent_auc(sw_entropy)
        swelling_df.loc[len(swelling_df)] = [s, sw_acc, sw_entropy.mean(),
                                             sw_ent_auc]
    result_dict['swelling_df'] = swelling_df

    # run predictions after training
    # need:
    rot_three_df = get_pred_df(rot_three_img, session, ops, mode)
    rot_three_df['Angle'] = rot_three_df['sample_idx'] * 10
    result_dict['rot_three_df'] = rot_three_df

    rot_pos_three_df = get_pred_df(rot_pos_three_img, session, ops, mode)
    rot_pos_three_df['Angle'] = rot_pos_three_df['sample_idx'] * 10
    result_dict['rot_pos_three_df'] = rot_pos_three_df

    rot_one_df = get_pred_df(rot_one_img, session, ops, mode)
    rot_one_df['Angle'] = rot_one_df['sample_idx'] * 10
    result_dict['rot_one_df'] = rot_one_df

    rot_pos_one_df = get_pred_df(rot_pos_one_img, session, ops, mode)
    rot_pos_one_df['Angle'] = rot_pos_one_df['sample_idx'] * 10
    result_dict['rot_pos_one_df'] = rot_pos_one_df

    rot_six_df = get_pred_df(rot_six_img, session, ops, mode)
    rot_six_df['Angle'] = rot_six_df['sample_idx'] * 10
    result_dict['rot_six_df'] = rot_six_df

    rot_pos_six_df = get_pred_df(rot_pos_six_img, session, ops, mode)
    rot_pos_six_df['Angle'] = rot_pos_six_df['sample_idx'] * 10
    result_dict['rot_pos_six_df'] = rot_pos_six_df

    mixup_three_eight_df = get_pred_df(mixup_three_eight_img, session, ops,
                                       mode)
    mixup_three_eight_df['Mixup factor'] = mixup_three_eight_df[
                                               'sample_idx'] / 10.
    result_dict['mixup_three_eight_df'] = mixup_three_eight_df

    return result_dict
