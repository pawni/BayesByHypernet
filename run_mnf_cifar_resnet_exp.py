import experiments_cifar as experiments
import networks

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', '-p', help='tb directory',
                        default='/vol/biomedic2/np716/bbh_nips/cifar_resnet'
                                '/mnf/')
    parser.add_argument('--experiment', '-x', help='tb directory',
                        default='test')
    parser.add_argument('--seed', '-s', help='seed',
                        default=42, type=int)
    parser.add_argument('--epochs', '-e', help='tb directory',
                        default=5, type=int)
    parser.add_argument('--annealing', '-a', help='', default=False,
                        action='store_true')
    parser.add_argument('--random_weights', '-r', help='', default=0, type=int)
    parser.add_argument('--lr', '-d', help='', default=0.001, type=float)
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
    config['random_weights'] = args.random_weights
    config['num_samples'] = 5
    config['annealing'] = args.annealing
    config['learning_rate'] = args.lr
    config['annealing_epoch_start'] = 20
    config['annealing_epoch_length'] = 15
    config['epochs'] = args.epochs
    config['optimiser'] = 'adam'

    tf.reset_default_graph()
    config['experiment'] = 'mnf_analytical'
    config['mod'] = 'mnf'
    config['args'] = str(args)

    ops = networks.get_mnf_cifar_resnet({}, learn_p=False, thres_var=0.3)
    experiments.run_analytical_experiment(ops, config)
