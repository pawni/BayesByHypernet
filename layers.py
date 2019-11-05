from __future__ import absolute_import, print_function, division

import tensorflow as tf
import numpy as np

from base_layers import *


class BBHDenseLayer(BBHLayer):
    def _build(self, name, input_dim, output_dim, use_bias=True,
               h_units=[16, 32],
               h_use_bias=True, h_noise_shape=1,
               num_samples=5, num_slices=1,
               aligned_noise=True,
               h_activation_func=lambda x: tf.maximum(0.1 * x, x)):
        self.share_noise = aligned_noise
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = self._get_weight(
                '{}/w'.format(name), (input_dim, output_dim), units=h_units,
                use_bias=h_use_bias, noise_shape=h_noise_shape,
                num_samples=num_samples, num_slices=num_slices,
                activation_func=h_activation_func)
            if self.use_bias:
                self.b = self._get_weight(
                    '{}/b'.format(name), (output_dim, ), units=h_units,
                    use_bias=h_use_bias, noise_shape=h_noise_shape,
                    num_samples=num_samples, num_slices=1,
                    activation_func=h_activation_func)

    def call(self, x, sample=0):
        x = tf.matmul(x, self.w[sample])

        if self.use_bias:
            x = x + self.b[sample]

        return x


class BBHConvLayer(BBHLayer):
    def _build(self, name, input_filter, output_filter, kernel_size,
               padding='SAME', strides=(1, 1, 1, 1), use_bias=True,
               num_samples=5, num_slices=1,
               aligned_noise=True,
               h_units=[16, 32], h_use_bias=True, h_noise_shape=1,
               h_activation_func=lambda x: tf.maximum(0.1 * x, x)):
        self.share_noise = aligned_noise
        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = self._get_weight(
                '{}/w'.format(name), (kernel_size, kernel_size, input_filter, output_filter),
                units=h_units, use_bias=h_use_bias, noise_shape=h_noise_shape,
                num_samples=num_samples, num_slices=num_slices,
                activation_func=h_activation_func)
            if self.use_bias:
                self.b = self._get_weight(
                    '{}/b'.format(name), (output_filter, ), units=h_units,
                    use_bias=h_use_bias, noise_shape=h_noise_shape,
                    num_samples=num_samples, num_slices=1,
                    activation_func=h_activation_func)

    def call(self, x, sample=0):
        x = tf.nn.conv2d(x, self.w[sample], self.strides, self.padding,
                         use_cudnn_on_gpu=True)
        if self.use_bias:
            x = x + self.b[sample]

        return x
    
class BBHDynDenseLayer(BBHDynLayer):
    def _build(self, name, input_dim, output_dim, use_bias=True,
               h_units=[16, 32],
               h_use_bias=False, h_noise_shape=1,
               h_activation_func=lambda x: tf.maximum(0.1 * x, x)):
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = self._get_weight(
                '{}/w'.format(name), (input_dim, output_dim), units=h_units,
                use_bias=h_use_bias, noise_shape=h_noise_shape,
                activation_func=h_activation_func)
            if self.use_bias:
                self.b = self._get_weight(
                    '{}/b'.format(name), (output_dim, ), units=h_units,
                    use_bias=h_use_bias, noise_shape=h_noise_shape,
                    activation_func=h_activation_func)

    def call(self, x, *args, **kwargs):
        cond = tf.reduce_mean(x, [0])
        
        cond = tf.concat(tf.nn.moments(x, [0]), 0)
        
        x = tf.matmul(x, self.w(cond))

        if self.use_bias:
            x = x + self.b(cond)

        return x

class BBHDynConvLayer(BBHDynLayer):
    def _build(self, name, input_filter, output_filter, kernel_size,
               padding='SAME', strides=(1, 1, 1, 1), use_bias=True,
               h_units=[16, 32], h_use_bias=False, h_noise_shape=1,
               h_activation_func=lambda x: tf.maximum(0.1 * x, x)):

        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = self._get_weight(
                '{}/w'.format(name), (kernel_size, kernel_size, input_filter, output_filter),
                units=h_units, use_bias=h_use_bias, noise_shape=h_noise_shape,
                activation_func=h_activation_func)
            if self.use_bias:
                self.b = self._get_weight(
                    '{}/b'.format(name), (output_filter, ), units=h_units,
                    use_bias=h_use_bias, noise_shape=h_noise_shape,
                    activation_func=h_activation_func)


    def call(self, x, *args, **kwargs):
        cond = tf.reduce_mean(x, [0, 1, 2])
        cond = tf.concat(tf.nn.moments(x, [0, 1, 2]), 0)
        
        x = tf.nn.conv2d(x, self.w(cond), self.strides, self.padding,
                         use_cudnn_on_gpu=True)
        if self.use_bias:
            x = x + self.b(cond)

        return x

class BBHNormDenseLayer(BBHLayer):
    def _build(self, name, input_dim, output_dim, use_bias=True,
               h_units=[16, 32],
               h_use_bias=False, h_noise_shape=1,
               h_activation_func=lambda x: tf.maximum(0.1 * x, x)):
        self.use_bias = use_bias
        with tf.variable_scope(name):
            w = tf.get_variable(
                'w', (input_dim, output_dim),
                tf.float32,
                tf.truncated_normal_initializer(0, 0.05))
            
            w_norm = self._get_weight(
                '{}/w_norm'.format(name), (1, output_dim),
                units=h_units, use_bias=h_use_bias, noise_shape=h_noise_shape,
                activation_func=h_activation_func)
            
            self.w = w / tf.sqrt(tf.reduce_sum(tf.square(w), axis=[0], keep_dims=True)) * w_norm
            
            if self.use_bias:
                self.b = self._get_weight(
                    '{}/b'.format(name), (output_dim, ), units=h_units,
                    use_bias=h_use_bias, noise_shape=h_noise_shape,
                    activation_func=h_activation_func)

    def call(self, x, *args, **kwargs):
        x = tf.matmul(x, self.w)

        if self.use_bias:
            x = x + self.b

        return x

class BBHNormConvLayer(BBHLayer):
    def _build(self, name, input_filter, output_filter, kernel_size,
               padding='SAME', strides=(1, 1, 1, 1), use_bias=True,
               h_units=[16, 32], h_use_bias=False, h_noise_shape=1,
               h_activation_func=lambda x: tf.maximum(0.1 * x, x)):

        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias
        
        with tf.variable_scope(name):
            w = tf.get_variable(
                'w', (kernel_size, kernel_size, input_filter, output_filter),
                tf.float32,
                tf.truncated_normal_initializer(0, 0.05))
            
            w_norm = self._get_weight(
                '{}/w_norm'.format(name), (1, 1, 1, output_filter),
                units=h_units, use_bias=h_use_bias, noise_shape=h_noise_shape,
                activation_func=h_activation_func)
            
            self.w = w / tf.sqrt(tf.reduce_sum(tf.square(w), axis=[0, 1, 2], keep_dims=True)) * w_norm
            
            if self.use_bias:
                self.b = self._get_weight(
                    '{}/b'.format(name), (output_filter, ), units=h_units,
                    use_bias=h_use_bias, noise_shape=h_noise_shape,
                    activation_func=h_activation_func)


    def call(self, x, *args, **kwargs):
        x = tf.nn.conv2d(x, self.w, self.strides, self.padding,
                         use_cudnn_on_gpu=True)
        if self.use_bias:
            x = x + self.b

        return x
    
    
class BBBDenseLayer(BBBLayer):
    def _build(self, name, input_dim, output_dim, use_bias=True, init_var=-9,
               prior_scale=1., aligned_noise=False):
        self.use_bias = use_bias
        self.share_noise = aligned_noise
        with tf.variable_scope(name):
            self.w = self._get_weight(
                'w', (input_dim, output_dim), init_var=init_var,
                prior_scale=prior_scale)
            if self.use_bias:
                self.b = self._get_weight(
                    'b', (output_dim,), init_var=init_var,
                    prior_scale=prior_scale)

    def call(self, x, *args, **kwargs):
        x = tf.matmul(x, self.w)
        if self.use_bias:
            x = x + self.b

        return x


class BBBConvLayer(BBBLayer):
    def _build(self, name, input_filter, output_filter, kernel_size,
               padding='SAME', strides=(1, 1, 1, 1), use_bias=True,
               aligned_noise=False,
               init_var=-9, prior_scale=1.):

        self.share_noise = aligned_noise
        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = self._get_weight(
                'w', (kernel_size, kernel_size, input_filter, output_filter),
                init_var=init_var, prior_scale=prior_scale)
            if self.use_bias:
                self.b = self._get_weight(
                    'b', (output_filter,), init_var=init_var,
                    prior_scale=prior_scale)

    def call(self, x, *args, **kwargs):
        x = tf.nn.conv2d(x, self.w, self.strides, self.padding,
                         use_cudnn_on_gpu=True)
        if self.use_bias:
            x = x + self.b

        return x


class VanillaDenseLayer(Layer):
    def _build(self, name, input_dim, output_dim, use_bias=True):
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = tf.get_variable(
                'w', (input_dim, output_dim), tf.float32,
                tf.variance_scaling_initializer())
            tf.add_to_collection('l2', tf.reduce_sum(tf.square(self.w)))
            if self.use_bias:
                self.b = tf.get_variable(
                    'b', (output_dim, ), tf.float32,
                    tf.zeros_initializer())
                tf.add_to_collection('l2', tf.reduce_sum(tf.square(self.b)))

    def call(self, x, *args, **kwargs):
        x = tf.matmul(x, self.w)
        if self.use_bias:
            x = x + self.b

        return x


class VanillaConvLayer(Layer):
    def _build(self, name, input_filter, output_filter, kernel_size,
               padding='SAME', strides=(1, 1, 1, 1), use_bias=True):
        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias
        with tf.variable_scope(name):
            self.w = tf.get_variable(
                'w', (kernel_size, kernel_size, input_filter, output_filter),
                tf.float32, tf.variance_scaling_initializer())
            tf.add_to_collection('l2', tf.reduce_sum(self.w ** 2))
            if self.use_bias:
                self.b = tf.get_variable(
                    'b', (output_filter,), tf.float32,
                    tf.zeros_initializer())
                tf.add_to_collection('l2', tf.reduce_sum(self.b ** 2))

    def call(self, x, *args, **kwargs):
        x = tf.nn.conv2d(x, self.w, self.strides, self.padding,
                         use_cudnn_on_gpu=True)
        if self.use_bias:
            x = x + self.b

        return x

########
#
# MNF layers adapted from https://github.com/AMLab-Amsterdam/MNF_VBNN
#


class MNFDenseLayer(Layer):
    def _build(self, name, input_dim, output_dim, learn_p=False,
               thres_var=1., init_var=-9, use_bias=True):

        self.thres_var = thres_var
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        flow_dim_h = 50
        with tf.variable_scope(name):
            self.w_loc = tf.get_variable(
                    'w_loc', (input_dim, output_dim), tf.float32,
                    tf.variance_scaling_initializer())

            self.w_log_scale_sq = tf.get_variable(
                'w_log_scale_sq', (input_dim, output_dim), tf.float32,
                tf.truncated_normal_initializer(init_var, 0.05))

            self.b_loc = tf.get_variable(
                'b_loc', (1, output_dim), tf.float32,
                tf.truncated_normal_initializer(0, 0.05))

            self.b_log_scale_sq = tf.get_variable(
                'b_log_scale_sq', (1, output_dim), tf.float32,
                tf.truncated_normal_initializer(init_var, 0.05))

            self.qzero_mean = tf.get_variable(
                'qzero_mean', (input_dim, ), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.qzero = tf.get_variable(
                'qzero', (input_dim,), tf.float32,
                tf.truncated_normal_initializer(np.log(0.1), 1e-6))

            self.rsr_M = tf.get_variable(
                'var_r_aux', (input_dim,), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.apvar_M = tf.get_variable(
                'apvar_r_aux', (input_dim,), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.rsri_M = tf.get_variable(
                'var_r_auxi', (input_dim,), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.pvar = tf.get_variable(
                'prior_var_r_p', (input_dim,), tf.float32,
                tf.truncated_normal_initializer(1., 1e-6),
                trainable=learn_p)

            self.pvar_bias = tf.get_variable(
                'prior_var_r_p_bias', (1,), tf.float32,
                tf.truncated_normal_initializer(1., 1e-6),
                trainable=learn_p)

            if input_dim == 1:
                self.flow_r = PlanarFlow(name + '_fr', input_dim,
                                         n_flows=2, # fixed to 2
                                         scope=name)
            else:
                self.flow_r = MaskedNVPFlow(name + '_fr', input_dim,
                                            n_flows=2, # fixed to 2
                                            n_hidden=0,
                                            dim_h=2 * flow_dim_h,
                                            scope=name)

            if input_dim == 1:
                self.flow_q = PlanarFlow(name + '_fq', input_dim,
                                         n_flows=2, # fixed to 2
                                         scope=name)
            else:
                self.flow_q = MaskedNVPFlow(name + '_fq', input_dim,
                                            n_flows=2,  # fixed to 2
                                            n_hidden=0,
                                            dim_h=flow_dim_h,
                                            scope=name)

        tf.add_to_collection('mnf_kl', -1. * self.kldiv())
        tf.add_to_collection('kl_term', -1. * self.kldiv())
        weight_samples = tf.stack([self.get_weight() for _ in range(5)])
        weight_samples = tf.reshape(weight_samples, [5, -1])
        tf.add_to_collection('weight_samples', weight_samples)

    def sample_z(self, size_M=1):

        qm0 = tf.exp(self.qzero)
        isample_M = tf.tile(tf.expand_dims(self.qzero_mean, 0), [size_M, 1])
        eps = tf.random_normal(tf.stack((size_M, self.input_dim)))
        sample_M = isample_M + tf.sqrt(qm0) * eps

        sample_M, logdets = self.flow_q.get_output_for(sample_M)

        return sample_M, logdets

    def kldiv(self):
        M, logdets = self.sample_z()
        logdets = logdets[0]
        M = tf.squeeze(M)

        std_mg = tf.exp(self.w_log_scale_sq)
        qm0 = tf.exp(self.qzero)
        if len(M.get_shape()) == 0:
            Mexp = M
        else:
            Mexp = tf.expand_dims(M, 1)

        Mtilde = Mexp * self.w_loc
        Vtilde = tf.square(std_mg)

        iUp = outer(tf.exp(self.pvar), tf.ones((self.output_dim,)))

        logqm = - tf.reduce_sum(.5 * (tf.log(2 * np.pi) + tf.log(qm0) + 1))
        logqm -= logdets

        kldiv_w = tf.reduce_sum(.5 * tf.log(iUp) - tf.log(std_mg) + (
            (Vtilde + tf.square(Mtilde)) / (2 * iUp)) - .5)
        kldiv_bias = tf.reduce_sum(
            .5 * self.pvar_bias - .5 * self.b_log_scale_sq + (
            (tf.exp(self.b_log_scale_sq) +
             tf.square(self.b_loc)) / (2 * tf.exp(self.pvar_bias))) - .5)

        apvar_M = self.apvar_M
        # shared network for hidden layer
        mw = tf.matmul(tf.expand_dims(apvar_M, 0), Mtilde)
        eps = tf.expand_dims(tf.random_normal((self.output_dim,)), 0)
        varw = tf.matmul(tf.square(tf.expand_dims(apvar_M, 0)), Vtilde)
        a = tf.nn.tanh(mw + tf.sqrt(varw) * eps)
        # split at output layer
        if len(tf.squeeze(a).get_shape()) != 0:
            w__ = tf.reduce_mean(outer(self.rsr_M, tf.squeeze(a)), axis=1)
            wv__ = tf.reduce_mean(outer(self.rsri_M, tf.squeeze(a)), axis=1)
        else:
            w__ = self.rsr_M * tf.squeeze(a)
            wv__ = self.rsri_M * tf.squeeze(a)

        M, logrm = self.flow_r.get_output_for(tf.expand_dims(M, 0))
        M = tf.squeeze(M)
        logrm = logrm[0]

        logrm += tf.reduce_sum(
            -.5 * tf.exp(wv__) * tf.square(M - w__) - .5 * tf.log(
                2 * np.pi) + .5 * wv__)

        return - kldiv_w + logrm - logqm - kldiv_bias

    def get_weight(self):
        std_mg = tf.clip_by_value(
            tf.exp(self.w_log_scale_sq), 0., self.thres_var)
        sample_M, _ = self.sample_z()

        w_sample = tf.transpose(sample_M) * self.w_loc
        w_sample += tf.random_normal(tf.shape(w_sample)) * std_mg

        return w_sample

    def call(self, x, sample_shape=None):
        if sample_shape is None:
            sample_shape = tf.shape(x)[0]

        std_mg = tf.clip_by_value(
            tf.exp(self.w_log_scale_sq), 0., self.thres_var)
        var_mg = tf.square(std_mg)
        
        sample_M, _ = self.sample_z(size_M=sample_shape)
        xt = x * sample_M

        mu_out = tf.matmul(xt, self.w_loc)
        varin = tf.matmul(tf.square(x), var_mg)
        if self.use_bias:
            mu_out += self.b_loc
            varin += tf.clip_by_value(
                tf.exp(self.b_log_scale_sq), 0., self.thres_var ** 2)

        xin = tf.sqrt(varin)
        sigma_out = xin * tf.random_normal(tf.shape(mu_out))

        output = mu_out + sigma_out
        return output

class MNFConvLayer(Layer):
    def _build(self, name, input_filter, output_filter, kernel_size,
               padding='SAME', strides=(1, 1, 1, 1), learn_p=False,
               thres_var=1., init_var=-9, use_bias=True):

        self.thres_var = thres_var
        self.input_filter = input_filter
        self.output_filter = output_filter

        self.padding = padding
        self.strides = strides

        self.input_dim = kernel_size * kernel_size * input_filter
        self.w_shape = (kernel_size, kernel_size, input_filter, output_filter)

        flow_dim_h = 50
        with tf.variable_scope(name):
            self.w_loc = tf.get_variable(
                    'w_loc', self.w_shape, tf.float32,
                    tf.variance_scaling_initializer())

            self.w_log_scale_sq = tf.get_variable(
                'w_log_scale_sq', self.w_shape, tf.float32,
                tf.truncated_normal_initializer(init_var, 0.05))

            self.b_loc = tf.get_variable(
                'b_loc', (output_filter, ), tf.float32,
                tf.truncated_normal_initializer(0, 0.05))

            self.b_log_scale_sq = tf.get_variable(
                'b_log_scale_sq', (output_filter, ), tf.float32,
                tf.truncated_normal_initializer(init_var, 0.05))

            self.qzero_mean = tf.get_variable(
                'qzero_mean', (output_filter, ), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.qzero = tf.get_variable(
                'qzero', (output_filter,), tf.float32,
                tf.truncated_normal_initializer(np.log(0.1), 1e-6))

            self.rsr_M = tf.get_variable(
                'var_r_aux', (output_filter,), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.apvar_M = tf.get_variable(
                'apvar_r_aux', (output_filter,), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.rsri_M = tf.get_variable(
                'var_r_auxi', (output_filter,), tf.float32,
                tf.truncated_normal_initializer(0., 0.05))

            self.pvar = tf.get_variable(
                'prior_var_r_p', (self.input_dim,), tf.float32,
                tf.truncated_normal_initializer(1., 1e-6),
                trainable=learn_p)

            self.pvar_bias = tf.get_variable(
                'prior_var_r_p_bias', (1,), tf.float32,
                tf.truncated_normal_initializer(1., 1e-6),
                trainable=learn_p)

            self.flow_r = MaskedNVPFlow(name + '_fr', output_filter,
                                        n_flows=2, # fixed to 2
                                        n_hidden=0,
                                        dim_h=2 * flow_dim_h,
                                        scope=name)

            self.flow_q = MaskedNVPFlow(name + '_fq', output_filter,
                                        n_flows=2,  # fixed to 2
                                        n_hidden=0,
                                        dim_h=flow_dim_h,
                                        scope=name)

        tf.add_to_collection('mnf_kl', -1. * self.kldiv())

        weight_samples = tf.stack([self.get_weight() for _ in range(5)])
        weight_samples = tf.reshape(weight_samples, [5, -1])
        tf.add_to_collection('weight_samples', weight_samples)

    def sample_z(self, size_M=1):

        qm0 = tf.exp(self.qzero)
        isample_M = tf.tile(tf.expand_dims(self.qzero_mean, 0), [size_M, 1])
        eps = tf.random_normal(tf.stack((size_M, self.output_filter)))
        sample_M = isample_M + tf.sqrt(qm0) * eps

        sample_M, logdets = self.flow_q.get_output_for(sample_M)

        return sample_M, logdets

    def kldiv(self):
        M, logdets = self.sample_z()
        logdets = logdets[0]
        M = tf.squeeze(M)

        std_w = tf.exp(self.w_log_scale_sq)
        mu = tf.reshape(self.w_loc, [-1, self.output_filter])
        std_w = tf.reshape(std_w, [-1, self.output_filter])
        Mtilde = mu * tf.expand_dims(M, 0)
        mbias = self.b_loc * M
        Vtilde = tf.square(std_w)

        iUp = outer(tf.exp(self.pvar), tf.ones((self.output_filter,)))

        qm0 = tf.exp(self.qzero)

        logqm = - tf.reduce_sum(.5 * (tf.log(2 * np.pi)
                                      + tf.log(qm0 + 1e-8) +1))
        logqm -= logdets

        kldiv_w = tf.reduce_sum(.5 * tf.log(iUp + 1e-8) - .5 * tf.log(Vtilde)
                                + ((Vtilde + tf.square(Mtilde))
                                   / (2 * iUp)) - .5)
        kldiv_bias = tf.reduce_sum(
            .5 * self.pvar_bias - .5 * self.b_log_scale_sq + (
            (tf.exp(self.b_log_scale_sq) +
             tf.square(mbias)) / (2 * tf.exp(self.pvar_bias))) - .5)

        apvar_M = self.apvar_M
        mw = tf.matmul(Mtilde, tf.expand_dims(apvar_M, 1))
        vw = tf.matmul(Vtilde, tf.expand_dims(tf.square(apvar_M), 1))
        eps = tf.expand_dims(tf.random_normal((self.input_dim,)), 1)
        a = mw + tf.sqrt(vw) * eps
        mb = tf.reduce_sum(mbias * apvar_M)
        vb = tf.reduce_sum(tf.exp(self.b_log_scale_sq) * tf.square(apvar_M))
        a += mb + tf.sqrt(vb) * tf.random_normal(())

        w__ = tf.reduce_mean(outer(tf.squeeze(a), self.rsr_M), axis=0)
        wv__ = tf.reduce_mean(outer(tf.squeeze(a), self.rsri_M), axis=0)

        M, logrm = self.flow_r.get_output_for(tf.expand_dims(M, 0))
        M = tf.squeeze(M)
        logrm = logrm[0]

        logrm += tf.reduce_sum(
            -.5 * tf.exp(wv__) * tf.square(M - w__) - .5 * tf.log(
                2 * np.pi) + .5 * wv__)

        return - kldiv_w + logrm - logqm - kldiv_bias

    def get_mean_var(self, x):
        var_w = tf.clip_by_value(tf.exp(self.w_log_scale_sq), 0., self.thres_var)
        var_w = tf.square(var_w)
        var_b = tf.clip_by_value(tf.exp(self.b_log_scale_sq), 0.,
                                 self.thres_var ** 2)

        # formally we do cross-correlation here
        muout = tf.nn.conv2d(x, self.w_loc, self.strides, self.padding,
                             use_cudnn_on_gpu=True) + self.b_loc
        varout = tf.nn.conv2d(tf.square(x), var_w, self.strides,
                              self.padding, use_cudnn_on_gpu=True) + var_b

        return muout, varout

    def get_weight(self):
        std_mg = tf.clip_by_value(
            tf.exp(self.w_log_scale_sq), 0., self.thres_var)
        sample_M, _ = self.sample_z()

        w_sample = self.w_loc * sample_M
        w_sample += tf.random_normal(tf.shape(w_sample)) * std_mg

        return w_sample

    def call(self, x, *args, **kwargs):
        sample_M, _ = self.sample_z(size_M=tf.shape(x)[0])
        sample_M = tf.expand_dims(tf.expand_dims(sample_M, 1), 2)
        mean_out, var_out = self.get_mean_var(x)
        mean_gout = mean_out * sample_M
        var_gout = tf.sqrt(var_out) * tf.random_normal(tf.shape(mean_gout))
        out = mean_gout + var_gout

        output = out
        return output