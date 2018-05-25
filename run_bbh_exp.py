import experiments
import networks

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', '-p', help='tb directory',
                        default='/vol/biomedic2/np716/bbh_nips/mnist/bbh/')
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
    parser.add_argument('--lr', '-d', help='', default=0.0001, type=float)
    parser.add_argument('--prior_scale', help='', default=1.,
                        type=float)
    parser.add_argument('--noise_shape', help='', default=1,
                        type=int)
    parser.add_argument('--layer_wise_gen', help='', default=False,
                        action='store_true')
    parser.add_argument('--slice_last_dim', help='', default=False,
                        action='store_true')
    parser.add_argument('--force_zero_mean', help='', default=False,
                        action='store_true')
    parser.add_argument('--kl_method', '-k', help='', default='approx',
                        choices=['approx', 'disc'])
    parser.add_argument('--opt', '-o', help='', default='adam',
                        choices=['adam', 'rms'])
    parser.add_argument('--independent_noise', help='', default=False,
                        action='store_true')

    parser.add_argument('--cuda', '-c', default='0')
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
    config['disc_units'] = [64, 64, 64]
    config['h_units'] = [64, 256, 512]
    config['disc_pretrain'] = 100
    config['disc_train'] = 1
    config['epochs'] = args.epochs
    config['prior_scale'] = args.prior_scale
    config['args'] = str(args)

    print(args)
    print('##########################')
    if args.kl_method == 'approx':
        tf.reset_default_graph()
        config['experiment'] = 'bbh_klapprox'
        ops = networks.get_bbh_mnist(
            {}, num_samples=5, sample_output=args.output_mc,
            noise_shape=args.noise_shape, layer_wise=args.layer_wise_gen,
            slice_last_dim=args.slice_last_dim, num_slices=1,
            force_zero_mean=args.force_zero_mean,
            aligned_noise=not args.independent_noise,
            h_units=config['h_units'])
        experiments.run_klapprox_experiment(ops, config)
    elif args.kl_method == 'disc':
        tf.reset_default_graph()
        config['experiment'] = 'bbh_disc'
        ops = networks.get_bbh_mnist(
            {}, num_samples=5, sample_output=args.output_mc,
            noise_shape=args.noise_shape, layer_wise=args.layer_wise_gen,
            slice_last_dim=args.slice_last_dim, num_slices=1,
            force_zero_mean=args.force_zero_mean,
            aligned_noise=not args.independent_noise,
            h_units=config['h_units'])
        experiments.run_disc_experiment(ops, config)
    else:
        print('didnt know what to do')
