import experiments
import networks

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', '-p', help='tb directory',
                        default='/vol/biomedic2/np716/bbh_nips/mnist/dropout/')
    parser.add_argument('--experiment', '-x', help='tb directory',
                        default='test')
    parser.add_argument('--seed', '-s', help='seed',
                        default=42, type=int)
    parser.add_argument('--epochs', '-e', help='tb directory',
                        default=5, type=int)
    parser.add_argument('--annealing', '-a', help='', default=False,
                        action='store_true')
    parser.add_argument('--lr', help='', default=0.001, type=float)
    parser.add_argument('--keep_prob', help='', default=0.5, type=float)
    parser.add_argument('--prior_scale', help='', default=1.,
                        type=float)
    parser.add_argument('--cuda', '-c', default='0')
    parser.add_argument('--opt', '-o', help='', default='adam',
                        choices=['adam', 'rms'])
    args = parser.parse_args()

    import os
    import tensorflow as tf

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    config = {}

    config['logdir'] = os.path.join(args.logdir, args.experiment)

    config['seed'] = args.seed
    config['annealing'] = args.annealing
    config['learning_rate'] = args.lr
    config['annealing_epoch_start'] = 5
    config['annealing_epoch_length'] = 15
    config['epochs'] = args.epochs
    config['prior_scale'] = args.prior_scale
    config['keep_prob'] = args.keep_prob
    config['optimiser'] = 'adam'

    tf.reset_default_graph()
    config['experiment'] = 'dropout_l2'
    config['mod'] = 'dropout'
    config['args'] = str(args)

    ops = networks.get_dropout_mnist({},  prior_scale=args.prior_scale,
                                     keep_prob=args.keep_prob)
    experiments.run_l2_experiment(ops, config)
