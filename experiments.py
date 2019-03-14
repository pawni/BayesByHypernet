from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils import mnist, not_mnist, build_result_dict
import time
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
from base_layers import BBHDiscriminator

try:
    import cPickle as pickle
except Exception as e:
    import pickle


def weight_summaries(weights):
    weight_mean, weight_var = tf.nn.moments(weights, 1)
    tf.summary.histogram('weights/w_var', weight_var)
    tf.summary.histogram('weights/w_mean', weight_mean)
    tf.summary.scalar('weights/w_var', tf.reduce_mean(weight_var))
    tf.summary.scalar('weights/w_mean', tf.reduce_mean(weight_mean))
    tf.summary.scalar('weights/w_var_min', tf.reduce_min(weight_var))
    tf.summary.scalar('weights/w_var_max', tf.reduce_max(weight_var))


def analysis(ops, session, save_path, num_samples, mod='bayesian'):
    print('Running analysis ...')
    results = build_result_dict(session, ops, mod)
    print('{}'.format(results['test_acc']))
    print('MNIST {} notMNIST {}'.format(
        results['test_ent_auc'], results['not_mnist_ent_auc']))

    if not (mod == 'map' or mod == 'dropout' or mod == 'ensemble'):
        all_weights = ops['all_weights']
        num_run_samples = 1000 // num_samples
        weights = np.zeros([num_run_samples * num_samples, 25])

        for i in range(num_run_samples):
            idx = i * num_samples
            end = idx + num_samples

            weights[idx:end] = np.swapaxes(session.run(all_weights[:25]), 0, 1)

        fig, ax = plt.subplots(5, 5, figsize=(40, 40))

        for i in range(5):
            for j in range(5):
                sns.distplot(weights[:, i * 5 + j], ax=ax[i, j])
                ax[i, j].set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weights.png'))
        plt.savefig(os.path.join(save_path, 'weights.pdf'))
        results['weight_samples'] = weights

        weights = np.zeros([num_run_samples * num_samples,
                            all_weights.get_shape().as_list()[0]])

        for i in range(num_run_samples):
            idx = i * num_samples
            end = idx + num_samples

            weights[idx:end] = np.swapaxes(session.run(all_weights), 0, 1)

        plt.figure()
        w_std = np.std(weights, 0)
        sns.distplot(w_std)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weights_std.png'))
        plt.savefig(os.path.join(save_path, 'weights_std.pdf'))

        results['w_std'] = w_std
        results['w_mean'] = np.mean(weights, 0)

        weights = weights[:, :5 * 5 * 20]

        w_df = pd.DataFrame(data=weights)
        corr_df = w_df.corr()
        plt.figure(figsize=(50, 50))
        sns.heatmap(corr_df)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_path, 'corr.png'))
        plt.savefig(os.path.join(save_path, 'corr.pdf'))
    elif mod == 'ensemble':
        num_samples = len(ops['tot_loss'])
        weights = zip(
            *[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                'ens{}'.format(i))
              for i in range(num_samples)])
        weights = tf.concat([tf.reshape(tf.stack(w), [num_samples, -1])
                             for w in weights], 1)

        all_weights = session.run(weights)

        fig, ax = plt.subplots(5, 5, figsize=(40, 40))

        for i in range(5):
            for j in range(5):
                sns.distplot(all_weights[:, i * 5 + j], ax=ax[i, j])
                ax[i, j].set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weights.png'))
        plt.savefig(os.path.join(save_path, 'weights.pdf'))
        results['weight_samples'] = all_weights

        plt.figure()
        w_std = np.std(all_weights, 0)
        sns.distplot(w_std)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'weights_std.png'))
        plt.savefig(os.path.join(save_path, 'weights_std.pdf'))

        results['w_std'] = w_std
        results['w_mean'] = np.mean(all_weights, 0)

        all_weights = all_weights[:, :5 * 5 * 20]

        w_df = pd.DataFrame(data=all_weights)
        corr_df = w_df.corr()
        plt.figure(figsize=(50, 50))
        sns.heatmap(corr_df)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_path, 'corr.png'))
        plt.savefig(os.path.join(save_path, 'corr.pdf'))

    saver = tf.train.Saver()
    saver.save(session, os.path.join(save_path, 'model.ckpt'))

    return results


def run_klapprox_experiment(ops, config):
    save_path = os.path.join(config['logdir'], config['experiment'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed = config['seed']
    subsample_gen_weights = config['random_weights']
    num_samples = config['num_samples']
    annealing = config['annealing']
    lr = config['learning_rate']
    annealing_epoch_start = config['annealing_epoch_start']
    annealing_epoch_length = config['annealing_epoch_length']
    prior_scale = config['prior_scale']
    opt_type = config.get('optimiser', 'rms')
    full_kernel = config.get('full_kernel', False)

    batch_size = 100
    epochs = config['epochs']
    batches_per_epoch = len(mnist.train.labels) // batch_size

    tf.set_random_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    anneal = tf.placeholder_with_default(1., [])

    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if len(tf.get_collection('weight_samples')) > 1:
        gen_weights = tf.concat(
            [tf.transpose(t, [1, 0])
             for t in tf.get_collection('weight_samples')], 0)
    else:
        gen_weights = tf.transpose(
            tf.get_collection('weight_samples')[0], [1, 0])

    all_weights = gen_weights
    ops['all_weights'] = all_weights

    scaling = 1. / len(mnist.train.labels)

    # if subsample_gen_weights > 0:
    #    rand_indizes = tf.random_shuffle(
    #         tf.range(tf.shape(gen_weights)[0])[:subsample_gen_weights])
    #     gen_weights = tf.gather(gen_weights, rand_indizes)
    #     scaling = (scaling * tf.cast(tf.shape(all_weights)[0], tf.float32)
    #                / subsample_gen_weights)

    weight_summaries(gen_weights)

    prior = tf.distributions.Normal(loc=0., scale=prior_scale)
    prior_samples = prior.sample(tf.shape(gen_weights))

    wp_distances = tf.square(
        tf.expand_dims(prior_samples, 2)
        - tf.expand_dims(gen_weights, 1))
    # [weights, samples, samples]

    ww_distances = tf.square(
        tf.expand_dims(gen_weights, 2)
        - tf.expand_dims(gen_weights, 1))

    if full_kernel:
        wp_distances = tf.sqrt(tf.reduce_sum(wp_distances, 0) + 1e-8)
        wp_dist = tf.reduce_min(wp_distances, 0)

        ww_distances = tf.sqrt(
            tf.reduce_sum(ww_distances, 0) + 1e-8) + tf.eye(num_samples) * 1e10
        ww_dist = tf.reduce_min(ww_distances, 0)

        # mean over samples
        kl = tf.cast(tf.shape(gen_weights)[0], tf.float32) * tf.reduce_mean(
            tf.log(wp_dist / (ww_dist + 1e-8) + 1e-8)
            + tf.log(float(num_samples) / (num_samples - 1)))
    else:
        wp_distances = tf.sqrt(wp_distances + 1e-8)
        wp_dist = tf.reduce_min(wp_distances, 1)

        ww_distances = tf.sqrt(ww_distances + 1e-8) + tf.expand_dims(
            tf.eye(num_samples) * 1e10, 0)
        ww_dist = tf.reduce_min(ww_distances, 1)

        # sum over weights, mean over samples
        kl = tf.reduce_sum(tf.reduce_mean(
            tf.log(wp_dist / (ww_dist + 1e-8) + 1e-8)
            + tf.log(float(num_samples) / (num_samples - 1)), 1))

    if opt_type == 'rms':
        opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-5)
    elif opt_type == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
    else:
        raise Exception('dont know opt {}'.format(opt_type))

    loss = ops['loss']
    if annealing:
        loss += anneal * scaling * kl
    else:
        loss += scaling * kl

    train_op = opt.minimize(loss)

    prior_matching_loss = kl
    prior_matching_loss_scaled = scaling * kl

    for i in range(10):
        tf.summary.histogram(
            'weights/w{}'.format(i), gen_weights[i])

    tf.summary.scalar('training/ce', ops['loss'])
    tf.summary.scalar('training/prior_matching', prior_matching_loss)
    tf.summary.scalar('training/prior_matching_scaled_anneal',
                      prior_matching_loss_scaled * anneal)
    tf.summary.scalar('training/prior_matching_scaled',
                      prior_matching_loss_scaled)

    tf.summary.scalar('stats/acc', ops['acc'])
    tf.summary.scalar('training/annealing', anneal)
    tf.summary.scalar('training/tot_loss', loss)

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    numerics = tf.add_check_numerics_ops()

    with tf.Session() as s:
        sum_writer = tf.summary.FileWriter(save_path)

        # initialise the weights
        s.run(init)
        start_time = time.time()

        annealing_length = annealing_epoch_length * batches_per_epoch
        start_annealing = annealing_epoch_start * batches_per_epoch

        with trange(epochs * batches_per_epoch) as pbar:  # run ~50 epochs
            for i in pbar:
                # get batch from dataset
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                if i < start_annealing:
                    annealing = 0.
                elif i < start_annealing + annealing_length:
                    annealing = (i - start_annealing) / float(annealing_length)
                else:
                    annealing = 1.

                feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys,
                             anneal: annealing}

                fetches = {'loss_g': loss, 'ce': ops['loss'],
                           'kl': kl,
                           'acc': ops['acc'], 'g_optimiser': train_op,
                           'numerics': numerics, 'summary': summary_op}

                fetched = s.run(fetches, feed_dict=feed_dict)

                if i % 100 == 0:
                    sum_writer.add_summary(fetched['summary'], i)

                pbar.set_postfix(acc=fetched['acc'],
                                 g_loss=fetched['loss_g'],
                                 kl=fetched['kl'],
                                 l_loss=fetched['ce'])

        end_time = time.time()
        run_time = end_time - start_time
        print('Finished run in {} s'.format(run_time))

        results = analysis(ops, s, save_path, num_samples)
        results['run_time'] = run_time
        results['config'] = config

        with open(os.path.join(save_path, "res.txt"), "a") as f:
            print('Had config:\n{}\n'.format(config), file=f)
            print('Finished run in {} s'.format(run_time), file=f)
            print('{}'.format(results['test_acc']), file=f)
            print('MNIST {} notMNIST {}'.format(
                results['test_ent_auc'], results['not_mnist_ent_auc']), file=f)

        with open(os.path.join(save_path, "dump.pickle"), "wb") as f:
            pickle.dump(results, f)


def run_disc_experiment(ops, config):
    save_path = os.path.join(config['logdir'], config['experiment'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed = config['seed']
    subsample_gen_weights = config['random_weights']
    num_samples = config['num_samples']
    annealing = config['annealing']
    lr = config['learning_rate']
    annealing_epoch_start = config['annealing_epoch_start']
    annealing_epoch_length = config['annealing_epoch_length']
    disc_units = config['disc_units']
    disc_pretrain = config['disc_pretrain']
    disc_train = config['disc_train']
    prior_scale = config['prior_scale']
    opt_type = config.get('optimiser', 'rms')

    batch_size = 100
    epochs = config['epochs']
    batches_per_epoch = len(mnist.train.labels) // batch_size

    tf.set_random_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    anneal = tf.placeholder_with_default(1., [])

    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if len(tf.get_collection('weight_samples')) > 1:
        gen_weights = tf.concat(
            [tf.transpose(t, [1, 0])
             for t in tf.get_collection('weight_samples')], 0)
    else:
        gen_weights = tf.transpose(
            tf.get_collection('weight_samples')[0], [1, 0])

    all_weights = gen_weights
    ops['all_weights'] = all_weights

    scaling = 1. / len(mnist.train.labels)
    if subsample_gen_weights > 0:
        rand_indizes = tf.random_shuffle(
            tf.range(tf.shape(gen_weights)[0])[:subsample_gen_weights])
        gen_weights = tf.gather(gen_weights, rand_indizes)

        scaling = (scaling * tf.cast(tf.shape(all_weights)[0], tf.float32)
                   / subsample_gen_weights)

    weight_summaries(gen_weights)
    gen_weights = tf.reshape(gen_weights, [-1, 1])

    prior = tf.distributions.Normal(loc=0., scale=prior_scale)
    prior_samples = prior.sample(tf.shape(gen_weights))

    discriminator = BBHDiscriminator(
        input_dim=1, units=disc_units)

    prior_logits = discriminator(prior_samples)
    gen_logits = discriminator(gen_weights)

    loss_gen = - tf.reduce_mean(
        tf.log(tf.clip_by_value(tf.nn.sigmoid(gen_logits), 1e-8, 1.0),
               name='log_g_d'))
    loss_prior = - tf.reduce_mean(
        tf.log(tf.clip_by_value(1 - tf.nn.sigmoid(prior_logits), 1e-8, 1.0),
               name='log_n_d'))

    disc_loss = loss_gen + loss_prior

    tf.summary.scalar(
        'stats/disc_prob_gen', tf.reduce_mean(tf.nn.sigmoid(gen_logits)))
    tf.summary.scalar(
        'stats/disc_prob_prior', tf.reduce_mean(tf.nn.sigmoid(prior_logits)))

    disc_prior_acc = tf.reduce_mean(
        tf.cast(tf.nn.sigmoid(prior_logits) <= 0.5, tf.float32))
    disc_gen_acc = tf.reduce_mean(
        tf.cast(tf.nn.sigmoid(gen_logits) > 0.5, tf.float32))

    tf.summary.scalar('stats/d_n_acc', disc_prior_acc)
    tf.summary.scalar('stats/d_g_acc', disc_gen_acc)

    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  'implicit_discriminator')

    if opt_type == 'rms':
        opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-5)
    elif opt_type == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
    else:
        raise Exception('dont know opt {}'.format(opt_type))

    loss = ops['loss']
    gen_logits = tf.reshape(gen_logits, [-1, num_samples])
    kl = tf.reduce_sum(tf.reduce_mean(gen_logits, 1))
    if annealing:
        loss += anneal * scaling * kl
    else:
        loss += scaling * kl

    def optimise(opt, loss, vars):
        gvs = opt.compute_gradients(loss, var_list=vars)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in
                      gvs]
        train_op = opt.apply_gradients(capped_gvs)
        return train_op

    train_op = optimise(opt, loss, net_vars)
    train_op_disc = optimise(opt, disc_loss, disc_vars)

    prior_matching_loss = kl
    prior_matching_loss_scaled = scaling * kl

    for i in range(10):
        tf.summary.histogram(
            'weights/w{}'.format(i), all_weights[i])

    tf.summary.scalar('training/ce', ops['loss'])
    tf.summary.scalar('training/prior_matching', prior_matching_loss)
    tf.summary.scalar('training/prior_matching_scaled_anneal',
                      prior_matching_loss_scaled * anneal)
    tf.summary.scalar('training/prior_matching_scaled',
                      prior_matching_loss_scaled)

    tf.summary.scalar('stats/acc', ops['acc'])
    tf.summary.scalar('training/annealing', anneal)
    tf.summary.scalar('training/tot_loss', loss)

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    numerics = tf.add_check_numerics_ops()

    with tf.Session() as s:
        sum_writer = tf.summary.FileWriter(save_path)

        # initialise the weights
        s.run(init)
        start_time = time.time()

        annealing_length = annealing_epoch_length * batches_per_epoch
        start_annealing = annealing_epoch_start * batches_per_epoch

        with trange(epochs * batches_per_epoch) as pbar:
            for _ in range(disc_pretrain):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                s.run(train_op_disc, feed_dict={ops['x']: batch_xs})

            for i in pbar:
                # get batch from dataset
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                if i < start_annealing:
                    annealing = 0.
                elif i < start_annealing + annealing_length:
                    annealing = (i - start_annealing) / float(annealing_length)
                else:
                    annealing = 1.

                feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys,
                             anneal: annealing}

                for _ in range(disc_train - 1):
                    # run discriminator to give better gradients
                    s.run(train_op_disc, feed_dict=feed_dict)

                fetches = {'loss_g': loss, 'ce': ops['loss'],
                           'loss_d': disc_loss,
                           'kl': kl, 'd_n_acc': disc_prior_acc,
                           'd_g_acc': disc_gen_acc, 'disc_train': train_op_disc,
                           'acc': ops['acc'], 'g_optimiser': train_op,
                           'numerics': numerics, 'summary': summary_op}

                fetched = s.run(fetches, feed_dict=feed_dict)
                if i % 100 == 0:
                    sum_writer.add_summary(fetched['summary'], i)

                pbar.set_postfix(acc=fetched['acc'], d_loss=fetched['loss_d'],
                                 g_loss=fetched['loss_g'],
                                 d_g_loss=fetched['kl'],
                                 l_loss=fetched['ce'],
                                 d_n_acc=fetched['d_n_acc'],
                                 d_g_acc=fetched['d_g_acc'])

        end_time = time.time()
        run_time = end_time - start_time
        print('Finished run in {} s'.format(run_time))

        results = analysis(ops, s, save_path, num_samples)
        results['run_time'] = run_time

        results['config'] = config

        with open(os.path.join(save_path, "res.txt"), "a") as f:
            print('Had config:\n{}\n'.format(config), file=f)
            print('Finished run in {} s'.format(run_time), file=f)
            print('{}'.format(results['test_acc']), file=f)
            print('MNIST {} notMNIST {}'.format(
                results['test_ent_auc'], results['not_mnist_ent_auc']), file=f)

        with open(os.path.join(save_path, "dump.pickle"), "wb") as f:
            pickle.dump(results, f)


def run_analytical_experiment(ops, config):
    save_path = os.path.join(config['logdir'], config['experiment'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed = config['seed']
    subsample_gen_weights = config['random_weights']
    num_samples = config['num_samples']
    annealing = config['annealing']
    lr = config['learning_rate']
    annealing_epoch_start = config['annealing_epoch_start']
    annealing_epoch_length = config['annealing_epoch_length']
    opt_type = config.get('optimiser', 'rms')

    batch_size = 100
    epochs = config['epochs']
    batches_per_epoch = len(mnist.train.labels) // batch_size

    tf.set_random_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    anneal = tf.placeholder_with_default(1., [])

    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if len(tf.get_collection('weight_samples')) > 1:
        gen_weights = tf.concat(
            [tf.transpose(t, [1, 0])
             for t in tf.get_collection('weight_samples')], 0)
    else:
        gen_weights = tf.transpose(
            tf.get_collection('weight_samples')[0], [1, 0])

    all_weights = gen_weights
    ops['all_weights'] = all_weights

    scaling = 1. / len(mnist.train.labels)
    if subsample_gen_weights > 0:
        rand_indizes = tf.random_shuffle(
            tf.range(tf.shape(gen_weights)[0])[:subsample_gen_weights])
        gen_weights = tf.gather(gen_weights, rand_indizes)

    weight_summaries(gen_weights)

    kl = tf.add_n(tf.get_collection('kl_term'))

    if opt_type == 'rms':
        opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-5)
    elif opt_type == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
    else:
        raise Exception('dont know opt {}'.format(opt_type))

    loss = ops['loss']
    if annealing:
        loss += anneal * scaling * kl
    else:
        loss += scaling * kl

    train_op = opt.minimize(loss)

    prior_matching_loss = kl
    prior_matching_loss_scaled = scaling * kl

    for i in range(10):
        tf.summary.histogram(
            'weights/w{}'.format(i), gen_weights[i])

    tf.summary.scalar('training/ce', ops['loss'])
    tf.summary.scalar('training/prior_matching', prior_matching_loss)
    tf.summary.scalar('training/prior_matching_scaled_anneal',
                      prior_matching_loss_scaled * anneal)
    tf.summary.scalar('training/prior_matching_scaled',
                      prior_matching_loss_scaled)

    tf.summary.scalar('stats/acc', ops['acc'])
    tf.summary.scalar('training/annealing', anneal)
    tf.summary.scalar('training/tot_loss', loss)

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    numerics = tf.add_check_numerics_ops()

    with tf.Session() as s:
        sum_writer = tf.summary.FileWriter(save_path)

        # initialise the weights
        s.run(init)
        start_time = time.time()

        annealing_length = annealing_epoch_length * batches_per_epoch
        start_annealing = annealing_epoch_start * batches_per_epoch

        with trange(epochs * batches_per_epoch) as pbar:  # run ~50 epochs
            for i in pbar:
                # get batch from dataset
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                if i < start_annealing:
                    annealing = 0.
                elif i < start_annealing + annealing_length:
                    annealing = (i - start_annealing) / float(annealing_length)
                else:
                    annealing = 1.

                feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys,
                             anneal: annealing}

                fetches = {'loss_g': loss, 'ce': ops['loss'],
                           'kl': kl,
                           'acc': ops['acc'], 'g_optimiser': train_op,
                           'numerics': numerics, 'summary': summary_op}

                fetched = s.run(fetches, feed_dict=feed_dict)
                if i % 100 == 0:
                    sum_writer.add_summary(fetched['summary'], i)

                pbar.set_postfix(acc=fetched['acc'],
                                 g_loss=fetched['loss_g'],
                                 kl=fetched['kl'],
                                 l_loss=fetched['ce'])

        end_time = time.time()
        run_time = end_time - start_time
        print('Finished run in {} s'.format(run_time))

        results = analysis(ops, s, save_path, num_samples)
        results['run_time'] = run_time

        results['config'] = config

        with open(os.path.join(save_path, "res.txt"), "a") as f:
            print('Had config:\n{}\n'.format(config), file=f)
            print('Finished run in {} s'.format(run_time), file=f)
            print('{}'.format(results['test_acc']), file=f)
            print('MNIST {} notMNIST {}'.format(
                results['test_ent_auc'], results['not_mnist_ent_auc']), file=f)

        with open(os.path.join(save_path, "dump.pickle"), "wb") as f:
            pickle.dump(results, f)


def run_l2_experiment(ops, config):
    save_path = os.path.join(config['logdir'], config['experiment'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed = config['seed']
    annealing = config['annealing']
    lr = config['learning_rate']
    annealing_epoch_start = config['annealing_epoch_start']
    annealing_epoch_length = config['annealing_epoch_length']
    opt_type = config.get('optimiser', 'rms')

    batch_size = 100
    epochs = config['epochs']
    batches_per_epoch = len(mnist.train.labels) // batch_size

    tf.set_random_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    anneal = tf.placeholder_with_default(1., [])

    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    kl = tf.losses.get_regularization_loss()

    if opt_type == 'rms':
        opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-5)
    elif opt_type == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
    else:
        raise Exception('dont know opt {}'.format(opt_type))

    scaling = 1. / len(mnist.train.labels)

    loss = ops['loss']
    if annealing:
        loss += anneal * scaling * kl
    else:
        loss += scaling * kl

    train_op = opt.minimize(loss)

    prior_matching_loss = kl
    prior_matching_loss_scaled = kl * scaling

    tf.summary.scalar('training/ce', ops['loss'])
    tf.summary.scalar('training/prior_matching', prior_matching_loss)
    tf.summary.scalar('training/prior_matching_scaled',
                      prior_matching_loss_scaled)
    tf.summary.scalar('training/prior_matching_scaled_anneal',
                      prior_matching_loss_scaled * anneal)

    tf.summary.scalar('stats/acc', ops['acc'])
    tf.summary.scalar('training/annealing', anneal)
    tf.summary.scalar('training/tot_loss', loss)

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    numerics = tf.add_check_numerics_ops()

    with tf.Session() as s:
        sum_writer = tf.summary.FileWriter(save_path)

        # initialise the weights
        s.run(init)
        start_time = time.time()

        annealing_length = annealing_epoch_length * batches_per_epoch
        start_annealing = annealing_epoch_start * batches_per_epoch

        with trange(epochs * batches_per_epoch) as pbar:  # run ~50 epochs
            for i in pbar:
                # get batch from dataset
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                if i < start_annealing:
                    annealing = 0.
                elif i < start_annealing + annealing_length:
                    annealing = (i - start_annealing) / float(annealing_length)
                else:
                    annealing = 1.

                feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys,
                             anneal: annealing}

                fetches = {'loss_g': loss, 'ce': ops['loss'],
                           'kl': kl,
                           'acc': ops['acc'], 'g_optimiser': train_op,
                           'numerics': numerics, 'summary': summary_op}

                fetched = s.run(fetches, feed_dict=feed_dict)
                if i % 100 == 0:
                    sum_writer.add_summary(fetched['summary'], i)

                pbar.set_postfix(acc=fetched['acc'],
                                 g_loss=fetched['loss_g'],
                                 kl=fetched['kl'],
                                 l_loss=fetched['ce'])

        end_time = time.time()
        run_time = end_time - start_time
        print('Finished run in {} s'.format(run_time))

        results = analysis(ops, s, save_path, 1, mod=config['mod'])
        results['run_time'] = run_time

        results['config'] = config

        with open(os.path.join(save_path, "res.txt"), "a") as f:
            print('Had config:\n{}\n'.format(config), file=f)
            print('Finished run in {} s'.format(run_time), file=f)
            print('{}'.format(results['test_acc']), file=f)
            print('MNIST {} notMNIST {}'.format(
                results['test_ent_auc'], results['not_mnist_ent_auc']), file=f)

        with open(os.path.join(save_path, "dump.pickle"), "wb") as f:
            pickle.dump(results, f)


def run_ensemble_experiment(ops, config):
    save_path = os.path.join(config['logdir'], config['experiment'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed = config['seed']
    lr = config['learning_rate']
    opt_type = config.get('optimiser', 'rms')

    batch_size = 100
    epochs = config['epochs']
    batches_per_epoch = len(mnist.train.labels) // batch_size

    tf.set_random_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    anneal = tf.placeholder_with_default(1., [])

    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    kl = tf.losses.get_regularization_loss()

    if opt_type == 'rms':
        opt = tf.train.RMSPropOptimizer(lr, epsilon=1e-5)
    elif opt_type == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
    else:
        raise Exception('dont know opt {}'.format(opt_type))

    scaling = 1. / len(mnist.train.labels)

    optimiser = [opt.minimize(tot_loss) for tot_loss in ops['tot_loss']]

    init = tf.global_variables_initializer()

    with tf.Session() as s:

        # initialise the weights
        s.run(init)
        start_time = time.time()

        with trange(epochs * batches_per_epoch) as pbar:  # run ~50 epochs
            for i in pbar:
                # get batch from dataset
                ce = 0
                b_acc = 0
                for loss, acc, opt in zip(ops['loss'], ops['acc'], optimiser):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys}
                    l_loss, l_acc, _ = s.run([loss, acc, opt],
                                             feed_dict=feed_dict)
                    ce += l_loss / 10
                    b_acc += l_acc / 10
                pbar.set_postfix(acc=b_acc, ce=ce)

        end_time = time.time()
        run_time = end_time - start_time
        print('Finished run in {} s'.format(run_time))

        results = analysis(ops, s, save_path, 1, mod=config['mod'])
        results['run_time'] = run_time

        results['config'] = config

        with open(os.path.join(save_path, "res.txt"), "a") as f:
            print('Had config:\n{}\n'.format(config), file=f)
            print('Finished run in {} s'.format(run_time), file=f)
            print('{}'.format(results['test_acc']), file=f)
            print('MNIST {} notMNIST {}'.format(
                results['test_ent_auc'], results['not_mnist_ent_auc']), file=f)

        with open(os.path.join(save_path, "dump.pickle"), "wb") as f:
            pickle.dump(results, f)
