from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np


def outer(x, y):
    return tf.matmul(tf.expand_dims(x, 1), tf.transpose(tf.expand_dims(y, 1)))


class Layer(object):
    def __init__(self, *args, **kwargs):
        self._build(*args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        return self.call(x, *args, **kwargs)

    def call(self, x, *args, **kwargs):
        raise NotImplementedError('Not implemented in abstract class')


with tf.variable_scope('implict_hypernet') as vs:
    hypernet_vs = vs

with tf.variable_scope('') as vs:
    none_vs = vs


class BBHDiscriminator(object):
    def __init__(self, input_dim=1, units=[20, 20]):
        self.layers = []
        with tf.variable_scope(none_vs):
            with tf.variable_scope('implicit_discriminator'):
                for unit in units:
                    layer = tf.layers.Dense(unit, activation=tf.nn.relu)
                    layer.build((None, input_dim))
                    self.layers.append(layer)

                    input_dim = unit

                layer = tf.layers.Dense(1)
                layer.build((None, input_dim))
                self.layers.append(layer)

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)

        return x
    

class BBHLayer(Layer):
    share_noise = True
    
    def _get_weight(self, name, size, units=[16, 32], use_bias=True,
                    noise_shape=1, num_samples=5, num_slices=1,
                    activation_func=lambda x: tf.maximum(0.1 * x, x)):
        slice_size = size[-1]
        assert slice_size % num_slices == 0

        gen_size = slice_size // num_slices

        cond = tf.eye(num_slices)

        with tf.variable_scope(hypernet_vs):
            with tf.variable_scope(name):
                flat_size = np.prod(size[:-1]) * gen_size
                
                if self.share_noise:
                    bbh_noise_col = tf.get_collection('bbh_noise')
                    if len(bbh_noise_col) == 1:
                        z = bbh_noise_col[0]
                    else:
                        z = tf.random_normal((num_samples, noise_shape))
                        tf.add_to_collection('bbh_noise', z)
                else:
                    z = tf.random_normal((num_samples, noise_shape))

                z = tf.stack([
                    tf.concat([
                        tf.tile(tf.expand_dims(z[s_dim], 0), [num_slices, 1]),
                        cond], 1) for s_dim in range(num_samples)])
                # [noise,cond, ..]

                for unit in units:
                    z = tf.layers.dense(inputs=z, units=unit, use_bias=use_bias)

                    z = activation_func(z)

                z = tf.layers.dense(inputs=z, units=flat_size,
                                    use_bias=use_bias)

                w = tf.reshape(z, [num_samples, -1])

                tf.add_to_collection('gen_weights', w)
                tf.add_to_collection('weight_samples', w)

                return tf.reshape(w, [num_samples, ] + list(size))


class BBHDynLayer(Layer):
    share_noise = False
    
    def _get_weight(self, name, size, units=[16, 32], use_bias=False,
                    noise_shape=1,
                    activation_func=lambda x: tf.maximum(0.1 * x, x)):
        with tf.variable_scope(hypernet_vs):
            with tf.variable_scope(name):
                flat_size = np.prod(size)
                
                if self.share_noise:
                    bbh_noise_col = tf.get_collection('bbh_noise')
                    if len(bbh_noise_col) == 1:
                        z = bbh_noise_col[0]
                    else:
                        z = tf.random_normal((1, 1))
                        tf.add_to_collection('bbh_noise', z)
                else:
                    z = tf.random_normal((1, 1))
                
                layers = []
                for unit in units:
                    layers.append(tf.layers.Dense(activation=activation_func, units=unit, use_bias=use_bias))

                layers.append(tf.layers.Dense(units=flat_size, use_bias=use_bias))
                
                def weight(z, cond):
                    cond = tf.reshape(cond, (1, -1))
                    z = tf.concat([z, cond], 1)
                    
                    for layer in layers:
                        z = layer(z)
                        
                    tf.add_to_collection('gen_weights', z)
                    tf.add_to_collection('weight_samples', z)
                        
                    return tf.reshape(z, size)
                
                return lambda x: weight(z, x)


class BBBLayer(Layer):
    share_noise = False

    def _get_weight(self, name, size, init_var=-9, prior_scale=1.):
        with tf.variable_scope(name):
            loc = tf.get_variable(
                'loc', size, tf.float32,
                tf.variance_scaling_initializer())

            log_scale_sq = tf.get_variable(
                'log_scale_sq', size, tf.float32,
                tf.truncated_normal_initializer(init_var, 0.05))

            if self.share_noise:
                bbb_noise_col = tf.get_collection('bbb_noise')
                if len(bbb_noise_col) == 1:
                    z = bbb_noise_col[0]
                else:
                    z = tf.random_normal((1, ))
                    tf.add_to_collection('bbb_noise', z)
            else:
                z = tf.random_normal(size)

            w = z * tf.sqrt(tf.exp(log_scale_sq)) + loc
            tf.add_to_collection('gen_weights', w)

            weight_samples = tf.stack([
                tf.random_normal(size) * tf.sqrt(tf.exp(log_scale_sq)) + loc
                for _ in range(5)])
            weight_samples = tf.reshape(weight_samples, [5, -1])
            tf.add_to_collection('weight_samples', weight_samples)

            # kl = -0.5 * tf.reduce_sum(1 + log_scale_sq - tf.square(loc)
            #                           - tf.exp(log_scale_sq))
            kl = -0.5 * tf.reduce_sum(
                1. + log_scale_sq - 2. * tf.log(prior_scale)
                - ((tf.exp(log_scale_sq) + tf.square(loc))
                   / (tf.square(prior_scale)))
            )

        tf.add_to_collection('bbb_locs', loc)
        tf.add_to_collection('bbb_log_scale_sq', log_scale_sq)
        tf.add_to_collection('bbb_kl', kl)
        tf.add_to_collection('kl_term', kl)

        return w


class MaskedNVPFlow(object):
    """
    copied from https://github.com/AMLab-Amsterdam/MNF_VBNN/
    """
    def __init__(self, name, incoming, n_flows=2, n_hidden=0, dim_h=10,
                 scope=None, nonlin=tf.nn.tanh, **kwargs):
        self.incoming = incoming
        self.n_flows = n_flows
        self.n_hidden = n_hidden
        self.name = name
        self.dim_h = dim_h
        self.params = []
        self.nonlin = nonlin
        self.scope = scope
        self.build()

    def build_mnn(self, fid, param_list):
        dimin = self.incoming
        with tf.variable_scope(self.scope):
            w = tf.get_variable(
                'w{}_{}_{}'.format(0, self.name, fid),
                (dimin, self.dim_h), tf.float32,
                tf.variance_scaling_initializer())

            b = tf.get_variable(
                'b{}_{}_{}'.format(0, self.name, fid),
                (self.dim_h, ), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            param_list.append([(w, b)])
            for l in xrange(self.n_hidden):
                wh = tf.get_variable(
                    'w{}_{}_{}'.format(l + 1, self.name, fid),
                    (self.dim_h, self.dim_h), tf.float32,
                    tf.uniform_unit_scaling_initializer())

                bh = tf.get_variable(
                    'b{}_{}_{}'.format(l + 1, self.name, fid),
                    (self.dim_h), tf.float32,
                    tf.zeros_initializer())
                param_list[-1].append((wh, bh))

            wout = tf.get_variable(
                'w{}_{}_{}'.format(self.n_hidden + 1, self.name, fid),
                (self.dim_h, dimin), tf.float32,
                tf.variance_scaling_initializer())

            bout = tf.get_variable(
                'b{}_{}_{}'.format(self.n_hidden + 1, self.name, fid),
                (dimin, ), tf.float32,
                tf.zeros_initializer())

            wout2 = tf.get_variable(
                'w{}_{}_{}_sigma'.format(self.n_hidden + 1, self.name, fid),
                (self.dim_h, dimin), tf.float32,
                tf.variance_scaling_initializer())

            bout2 = tf.get_variable(
                'b{}_{}_{}_sigma'.format(self.n_hidden + 1, self.name, fid),
                (dimin, ), tf.float32,
                tf.constant_initializer(2.))

            param_list[-1].append((wout, bout, wout2, bout2))

    def build(self):
        for flow in xrange(self.n_flows):
            self.build_mnn('muf_{}'.format(flow), self.params)

    def ff(self, x, weights):
        inputs = [x]
        for j in xrange(len(weights[:-1])):
            h = tf.matmul(inputs[-1], weights[j][0]) + weights[j][1]
            inputs.append(self.nonlin(h))
        wmu, bmu, wsigma, bsigma = weights[-1]
        mean = tf.matmul(inputs[-1], wmu) + bmu
        sigma = tf.matmul(inputs[-1], wsigma) + bsigma
        return mean, sigma

    def random_bernoulli(self, shape, p=0.5):
        if isinstance(shape, (list, tuple)):
            shape = tf.stack(shape)
        return tf.where(tf.random_uniform(shape) < p, tf.ones(shape),
                        tf.zeros(shape))

    def get_output_for(self, z, sample=True):
        logdets = tf.zeros((tf.shape(z)[0],))
        for flow in xrange(self.n_flows):
            mask = self.random_bernoulli(tf.shape(z), p=0.5) if sample else 0.5
            ggmu, ggsigma = self.ff(mask * z, self.params[flow])
            gate = tf.nn.sigmoid(ggsigma)
            logdets += tf.reduce_sum((1 - mask) * tf.log(gate), axis=1)
            z = (1 - mask) * (z * gate + (1 - gate) * ggmu) + mask * z

        return z, logdets


class PlanarFlow(object):
    """
    copied from https://github.com/AMLab-Amsterdam/MNF_VBNN/
    """
    def __init__(self, name, incoming, n_flows=2, scope=None,
                 **kwargs):
        self.incoming = incoming
        self.n_flows = n_flows
        self.sigma = 0.01
        self.params = []
        self.name = name
        self.scope = scope
        self.build()

    def build(self):
        with tf.variable_scope(self.scope):
            for flow in xrange(self.n_flows):
                w = tf.get_variable(
                    'w_{}_{}'.format(flow, self.name),
                    (self.incoming, 1), tf.float32,
                    tf.variance_scaling_initializer())

                u = tf.get_variable(
                    'u_{}_{}'.format(flow, self.name),
                    (self.incoming, 1), tf.float32,
                    tf.truncated_normal_initializer(0., 0.05))

                b = tf.get_variable(
                    'b_{}_{}'.format(flow, self.name),
                    (1, ), tf.float32,
                    tf.zeros_initializer())

                self.params.append([w, u, b])

    def get_output_for(self, z, **kwargs):
        logdets = tf.zeros((tf.shape(z)[0],))
        for flow in xrange(self.n_flows):
            w, u, b = self.params[flow]
            uw = tf.reduce_sum(u * w)
            muw = -1 + tf.nn.softplus(uw)  # = -1 + T.log(1 + T.exp(uw))
            u_hat = u + (muw - uw) * w / tf.reduce_sum(w ** 2)
            if len(z.get_shape()) == 1:
                zwb = z * w + b
            else:
                zwb = tf.matmul(z, w) + b
            psi = tf.matmul(1 - tf.nn.tanh(zwb) ** 2, tf.transpose(w))  # tanh(x)dx = 1 - tanh(x)**2
            psi_u = tf.matmul(psi, u_hat)
            logdets += tf.squeeze(tf.log(tf.abs(1 + psi_u)))
            zadd = tf.matmul(tf.nn.tanh(zwb), tf.transpose(u_hat))
            z += zadd
        return z, logdets
