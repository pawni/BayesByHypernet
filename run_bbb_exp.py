import experiments
import networks

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', '-p', help='tb directory',
                        default='/vol/biomedic2/np716/bbh_nips/mnist/bbb/')
    parser.add_argument('--experiment', '-x', help='tb directory',
                        default='test')
    parser.add_argument('--seed', '-s', help='seed',
                        default=42, type=int)
    parser.add_argument('--epochs', '-e', help='tb directory',
                        default=5, type=int)
    parser.add_argument('--output_mc', '-m', help='', default=False,
                        action='store_true')
    parser.add_argument('--annealing', '-a', help='', default=False,
                        action='store_true')
    parser.add_argument('--random_weights', '-r', help='', default=0, type=int)
    parser.add_argument('--lr', '-d', help='', default=0.001, type=float)
    parser.add_argument('--prior_scale', help='', default=1.,
                        type=float)
    parser.add_argument('--cuda', '-c', default='0')
    parser.add_argument('--opt', '-o', help='', default='adam',
                        choices=['adam', 'rms'])
    parser.add_argument('--align_noise', help='', default=False,
                        action='store_true')
    args = parser.parse_args()

    import os
    import tensorflow as tf

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    config = {}

    config['logdir'] = os.path.join(args.logdir, args.experiment)

    config['seed'] = args.seed
    config['random_weights'] = args.random_weights
    config['num_samples'] = 5
    config['annealing'] = args.annealing
    config['learning_rate'] = args.lr
    config['annealing_epoch_start'] = 5
    config['annealing_epoch_length'] = 15
    config['disc_units'] = [64, 64]
    config['disc_pretrain'] = 100
    config['disc_train'] = 1
    config['epochs'] = args.epochs
    config['prior_scale'] = args.prior_scale
    config['optimiser'] = 'adam'
    config['mod'] = 'bbb'

    tf.reset_default_graph()
    config['experiment'] = 'bbb_analytical'
    config['args'] = str(args)

    ops = networks.get_bbb_mnist({}, init_var=-9.,
                                 prior_scale=args.prior_scale,
                                 aligned_noise=args.align_noise)
    experiments.run_analytical_experiment(ops, config)

    tf.reset_default_graph()
    config['experiment'] = 'bbb_klapprox'
    config['args'] = str(args)
    ops = networks.get_bbb_mnist({}, init_var=-9.,
                                 prior_scale=args.prior_scale,
                                 aligned_noise=args.align_noise)
    experiments.run_klapprox_experiment(ops, config)

    tf.reset_default_graph()
    config['experiment'] = 'bbb_disc'
    config['args'] = str(args)
    ops = networks.get_bbb_mnist({}, init_var=-9.,
                                 prior_scale=args.prior_scale,
                                 aligned_noise=args.align_noise)
    experiments.run_disc_experiment(ops, config)

    if config['random_weights'] > 0:
        config['random_weights'] = 0
        tf.reset_default_graph()
        config['experiment'] = 'bbb_klapprox_r0'
        config['args'] = str(args)
        ops = networks.get_bbb_mnist({}, init_var=-9.,
                                     prior_scale=args.prior_scale,
                                     aligned_noise=args.align_noise)
        experiments.run_klapprox_experiment(ops, config)
