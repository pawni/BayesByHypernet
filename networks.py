import tensorflow as tf
import layers
import base_layers
import numpy as np
import copy


def get_bbh_mnist(ops, num_samples=5, sample_output=True, noise_shape=1,
                  layer_wise=False, slice_last_dim=False,
                  force_zero_mean=False,
                  num_slices=1, h_units=(256, 512),
                  aligned_noise=True):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x_inp = tf.reshape(x, [-1, 28, 28, 1])

    h_use_bias = True

    if layer_wise:
        if slice_last_dim:
            num_slices = 20
        c1 = layers.BBHConvLayer('c1', 1, 20, 5, 'VALID',
                                 num_samples=num_samples, num_slices=num_slices,
                                 h_noise_shape=noise_shape,
                                 h_units=h_units, h_use_bias=h_use_bias,
                                 aligned_noise=aligned_noise)

        if slice_last_dim:
            num_slices = 50
        c2 = layers.BBHConvLayer('c2', 20, 50, 5, 'VALID',
                                 num_samples=num_samples, num_slices=num_slices,
                                 h_noise_shape=noise_shape,
                                 h_units=h_units, h_use_bias=h_use_bias,
                                 aligned_noise=aligned_noise)

        if slice_last_dim:
            num_slices = 500
        fc1 = layers.BBHDenseLayer('fc1', 800, 500, h_units=h_units,
                                   num_samples=num_samples,
                                   num_slices=num_slices,
                                   h_noise_shape=noise_shape,
                                   h_use_bias=h_use_bias,
                                   aligned_noise=aligned_noise)

        if slice_last_dim:
            num_slices = 10
        fc2 = layers.BBHDenseLayer('fc2', 500, 10, h_units=h_units,
                                   num_samples=num_samples,
                                   num_slices=num_slices,
                                   h_noise_shape=noise_shape,
                                   h_use_bias=h_use_bias,
                                   aligned_noise=aligned_noise)
    else:
        cond_size = 130

        cond = tf.eye(cond_size)

        weight_shapes = {
            'conv1_w': [5, 5, 1, 20],
            'conv1_b': [20],
            'conv2_w': [5, 5, 20, 50],
            'conv2_b': [50],
            'fc1_w': [800, 500],
            'fc1_b': [500],
            'fc2_w': [500, 10],
            'fc2_b': [10],
        }
        weights = {}

        z = tf.random_normal((num_samples, noise_shape))

        z = tf.stack([tf.concat([
                tf.tile(tf.expand_dims(z[s_dim], 0), [cond_size, 1]),
                cond], 1) for s_dim in range(num_samples)])

        # z_stack = []
        # for s in range(num_samples):
        #     s_stack = []
        #     for c in range(cond_size):
        #         s_stack.append(tf.concat([z[s], cond[c]], 0))
        #     z_stack.append(tf.stack(s_stack))  # [c, -1]
        # z = tf.stack(z_stack)  # [noise, c, -1]
        tf.add_to_collection('gen_weights_conds', z)

        z = tf.reshape(z, [num_samples * cond_size, -1])

        with tf.variable_scope(base_layers.hypernet_vs):
            for unit in h_units:
                z = tf.layers.dense(z, unit, lambda x: tf.maximum(x, 0.1 * x),
                                    use_bias=h_use_bias)

            z = tf.layers.dense(z, 3316, use_bias=h_use_bias)

            z = tf.reshape(z, [num_samples, cond_size, -1])

            tf.add_to_collection('gen_weights_raw', z)  # [noise, c, -1]

            z = tf.reshape(z, [num_samples, -1])
            if force_zero_mean:
                z = z - tf.reduce_mean(z, 0, keepdims=True)
            tf.add_to_collection('gen_weights', z)
            tf.add_to_collection('weight_samples', z)

            idx = 0
            for w, shape in weight_shapes.items():
                end = idx + np.prod(shape)
                weights[w] = tf.reshape(z[:, idx:end], [num_samples, ] + shape)
                idx = end

            # conv 1
            def c1(x, sample=0):
                x = tf.nn.conv2d(x, weights['conv1_w'][sample], [1, 1, 1, 1],
                                 'VALID', use_cudnn_on_gpu=True)
                x = x + weights['conv1_b'][sample][sample]

                return x

            # conv 2
            def c2(x, sample=0):
                x = tf.nn.conv2d(x, weights['conv2_w'][sample], [1, 1, 1, 1],
                                 'VALID', use_cudnn_on_gpu=True)
                x = x + weights['conv2_b'][sample][sample]

                return x

            def fc1(x, sample=0):
                x = tf.matmul(x, weights['fc1_w'][sample])

                x = x + weights['fc1_b'][sample]

                return x

            def fc2(x, sample=0):
                x = tf.matmul(x, weights['fc2_w'][sample])

                x = x + weights['fc2_b'][sample]

                return x

    output_ind = []
    if sample_output:
        output = []

        for i in range(num_samples):
            x = c1(x_inp, i)
            tf.add_to_collection('c1_preact', x)
            x = tf.nn.relu(x)
            tf.add_to_collection('c1_act', x)

            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

            x = c2(x, i)
            tf.add_to_collection('c2_preact', x)
            x = tf.nn.relu(x)
            tf.add_to_collection('c2_act', x)

            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

            x = tf.layers.flatten(x)

            # fc 1

            x = fc1(x, i)
            tf.add_to_collection('fc1_preact', x)
            x = tf.nn.relu(x)
            tf.add_to_collection('fc1_act', x)

            # fc 2
            x = fc2(x, i)
            tf.add_to_collection('fc2_preact', x)
            output_ind.append(x)

            x = tf.nn.softmax(x)
            output.append(x)

        act_names = ['c1_preact', 'c1_act', 'c2_preact', 'c2_act',
                     'fc1_preact', 'fc1_act', 'fc2_preact']
        for name in act_names:
            act = tf.stack(tf.get_collection(name))
            mu, sig = tf.nn.moments(act, 0)
            tf.summary.histogram('act/{}_mu'.format(name), mu)
            tf.summary.histogram('act/{}_sig'.format(name), sig)

        x = tf.log(tf.add_n(output) / float(num_samples) + 1e-8)
    else:
        x = c1(x_inp)
        x = tf.nn.relu(x)

        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

        x = c2(x)
        x = tf.nn.relu(x)

        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

        x = tf.layers.flatten(x)

        # fc 1

        x = fc1(x)
        x = tf.nn.relu(x)

        # fc 2
        x = fc2(x)
        output_ind.append(x)

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    reg_losses = tf.losses.get_regularization_losses()
    if len(reg_losses) > 0:
        ops['loss'] += tf.add_n(reg_losses)

    loss_grads = tf.gradients(ce, ops['x'])[0]
    adv_data = ops['x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_cifar_image(ops):
    x = ops['x']
    is_eval = tf.placeholder(tf.bool, [])

    def distort_input(single_image):
        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.image.resize_image_with_crop_or_pad(
            single_image, 36, 36)
        distorted_image = tf.random_crop(distorted_image, [24, 24, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(
            distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(
            distorted_image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([24, 24, 3])

        return float_image

    def normalise_input(single_image):
        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(
            single_image, 24, 24)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Set the shapes of tensors.
        float_image.set_shape([24, 24, 3])
        return float_image

    x = tf.cond(is_eval,
                true_fn=lambda: tf.map_fn(normalise_input, x),
                false_fn=lambda: tf.map_fn(distort_input, x))

    # x = tf.map_fn(normalise_input, x)

    return x, is_eval


def get_bbh_cifar_resnet(ops, num_samples=5, sample_output=True, noise_shape=1,
                         layer_wise=False, slice_last_dim=False,
                         force_zero_mean=False,
                         aligned_noise=True,
                         num_slices=1, h_units=(256, 512)):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    filters = [16, 16, 32, 64]
    strides = [1, 2, 2, 2]
    num_units = 5

    weight_shapes = {}
    weight_shapes['conv1'] = {
        'w': [3, 3, 3, filters[0]],
        'b': [filters[0]],
    }

    weight_shapes['last'] = {
        'w': [filters[-1], 5],
        'b': [5],
    }

    old_filter = filters[0]
    for scale, filter in enumerate(filters[1:]):
        s = 'scale{}'.format(scale)
        weight_shapes[s] = {}
        for res_unit in range(num_units):
            r = 'unit{}'.format(res_unit)
            weight_shapes[s][r] = {
                'conv1': {'w': [3, 3, old_filter, filter],
                          'b': [filter]},
                'conv2': {'w': [3, 3, filter, filter],
                          'b': [filter]},
            }

            old_filter = filter

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x, is_eval = get_cifar_image(ops)
    ops['is_eval'] = is_eval

    ops['inp_x'] = x

    h_use_bias = True

    print('Building weights for:\n{}'.format(weight_shapes))

    all_layers = {}

    if layer_wise:
        w_shape = weight_shapes['conv1']['w']

        if slice_last_dim:
            num_slices = w_shape[-1]
        else:
            num_slices = 1
        all_layers['conv1'] = layers.BBHConvLayer(
            'conv1', w_shape[-2], w_shape[-1], w_shape[0],
            num_samples=num_samples, num_slices=num_slices,
            h_noise_shape=noise_shape, strides=[1, strides[0], strides[0], 1],
            h_units=h_units, h_use_bias=h_use_bias, aligned_noise=aligned_noise)

        for scale, filter in enumerate(filters[1:]):
            s = 'scale{}'.format(scale)
            all_layers[s] = {}
            stride = strides[scale + 1]
            for res_unit in range(num_units):
                r = 'unit{}'.format(res_unit)
                all_layers[s][r] = {}
                w_shape = weight_shapes[s][r]['conv1']['w']

                if slice_last_dim:
                    num_slices = w_shape[-1]
                else:
                    num_slices = 1

                all_layers[s][r]['bn1'] = tf.layers.BatchNormalization()
                all_layers[s][r]['bn2'] = tf.layers.BatchNormalization()

                all_layers[s][r]['conv1'] = layers.BBHConvLayer(
                    '{}/{}/conv1'.format(s, r),
                    w_shape[-2], w_shape[-1], w_shape[0],
                    num_samples=num_samples, num_slices=num_slices,
                    h_noise_shape=noise_shape, aligned_noise=aligned_noise,
                    strides=[1, stride, stride, 1],
                    h_units=h_units, h_use_bias=h_use_bias)

                all_layers[s][r]['conv2'] = layers.BBHConvLayer(
                    '{}/{}/conv2'.format(s, r),
                    w_shape[-1], w_shape[-1], w_shape[0],
                    num_samples=num_samples, num_slices=num_slices,
                    h_noise_shape=noise_shape, aligned_noise=aligned_noise,
                    strides=[1, 1, 1, 1],
                    h_units=h_units, h_use_bias=h_use_bias)

                stride = 1

        w_shape = weight_shapes['last']['w']

        if slice_last_dim:
            num_slices = w_shape[-1]
        else:
            num_slices = 1
        all_layers['last'] = layers.BBHDenseLayer(
            'last', filters[-1], w_shape[-1],
            num_samples=num_samples, num_slices=num_slices,
            h_noise_shape=noise_shape, aligned_noise=aligned_noise,
            h_units=h_units, h_use_bias=h_use_bias)
    else:
        cond_size = 231

        cond = tf.eye(cond_size)

        z = tf.random_normal((num_samples, noise_shape))

        z = tf.stack([tf.concat([
                tf.tile(tf.expand_dims(z[s_dim], 0), [cond_size, 1]),
                cond], 1) for s_dim in range(num_samples)])

        tf.add_to_collection('gen_weights_conds', z)

        z = tf.reshape(z, [num_samples * cond_size, -1])

        with tf.variable_scope(base_layers.hypernet_vs):
            for unit in h_units:
                z = tf.layers.dense(z, unit, lambda x: tf.maximum(x, 0.1 * x),
                                    use_bias=h_use_bias)

            z = tf.layers.dense(z, 2003, use_bias=h_use_bias)

            z = tf.reshape(z, [num_samples, cond_size, -1])

            tf.add_to_collection('gen_weights_raw', z)  # [noise, c, -1]

            z = tf.reshape(z, [num_samples, -1])
            if force_zero_mean:
                z = z - tf.reduce_mean(z, 0, keepdims=True)
            tf.add_to_collection('gen_weights', z)
            tf.add_to_collection('weight_samples', z)

        all_weights = {}

        idx = 0
        w_shape = weight_shapes['conv1']['w']
        b_shape = weight_shapes['conv1']['b']
        all_weights['conv1'] = {}

        end = idx + np.prod(w_shape)
        all_weights['conv1']['w'] = tf.reshape(
            z[:, idx:end], [num_samples, ] + w_shape)

        idx = end
        end = idx + np.prod(b_shape)
        all_weights['conv1']['b'] = tf.reshape(
            z[:, idx:end], [num_samples, ] + b_shape)

        def call_layer(x, sample=0):
            x = tf.nn.conv2d(x, all_weights['conv1']['w'][sample],
                             [1, strides[0], strides[0], 1],
                             'SAME', use_cudnn_on_gpu=True)
            x = x + all_weights['conv1']['b'][sample]

            return x

        all_layers['conv1'] = call_layer

        for scale, filter in enumerate(filters[1:]):
            s = 'scale{}'.format(scale)
            all_layers[s] = {}
            all_weights[s] = {}
            stride = strides[scale + 1]
            for res_unit in range(num_units):
                r = 'unit{}'.format(res_unit)
                all_layers[s][r] = {}
                all_weights[s][r] = {}

                all_layers[s][r]['bn1'] = tf.layers.BatchNormalization(
                    virtual_batch_size=1)
                all_layers[s][r]['bn2'] = tf.layers.BatchNormalization(
                    virtual_batch_size=1)

                w_shape = weight_shapes[s][r]['conv1']['w']
                b_shape = weight_shapes[s][r]['conv1']['b']
                all_weights[s][r]['conv1'] = {}

                end = idx + np.prod(w_shape)
                all_weights[s][r]['conv1']['w'] = tf.reshape(
                    z[:, idx:end], [num_samples, ] + w_shape)

                idx = end
                end = idx + np.prod(b_shape)
                all_weights[s][r]['conv1']['b'] = tf.reshape(
                    z[:, idx:end], [num_samples, ] + b_shape)

                def call_layer(s, r, stride, x, sample=0):
                    x = tf.nn.conv2d(
                        x, all_weights[s][r]['conv1']['w'][sample],
                        [1, stride, stride, 1],
                        'SAME', use_cudnn_on_gpu=True)
                    x = x + all_weights[s][r]['conv1']['b'][sample]

                    return x

                all_layers[s][r]['conv1'] = call_layer

                w_shape = weight_shapes[s][r]['conv2']['w']
                b_shape = weight_shapes[s][r]['conv2']['b']
                all_weights[s][r]['conv2'] = {}

                end = idx + np.prod(w_shape)
                all_weights[s][r]['conv2']['w'] = tf.reshape(
                    z[:, idx:end], [num_samples, ] + w_shape)

                idx = end
                end = idx + np.prod(b_shape)
                all_weights[s][r]['conv2']['b'] = tf.reshape(
                    z[:, idx:end], [num_samples, ] + b_shape)

                def call_layer(s, r, stride, x, sample=0):
                    x = tf.nn.conv2d(
                        x, all_weights[s][r]['conv2']['w'][sample],
                        [1, 1, 1, 1],
                        'SAME', use_cudnn_on_gpu=True)
                    x = x + all_weights[s][r]['conv2']['b'][sample]

                    return x

                all_layers[s][r]['conv2'] = call_layer

                stride = 1

        w_shape = weight_shapes['last']['w']
        b_shape = weight_shapes['last']['b']
        all_weights['last'] = {}

        end = idx + np.prod(w_shape)
        all_weights['last']['w'] = tf.reshape(
            z[:, idx:end], [num_samples, ] + w_shape)

        idx = end
        end = idx + np.prod(b_shape)
        all_weights['last']['b'] = tf.reshape(
            z[:, idx:end], [num_samples, ] + b_shape)

        def call_layer(x, sample=0):
            x = tf.matmul(x, all_weights['last']['w'][sample])

            x = x + all_weights['last']['b'][sample]

            return x

        all_layers['last'] = call_layer

    def call_resnet(x, sample=0):
        def call_res_unit(x, c1, c2, bn1, bn2, strides):
            in_filters = x.get_shape().as_list()[-1]

            orig_x = x
            if np.prod(strides) != 1:
                orig_x = tf.nn.avg_pool(orig_x, ksize=strides, strides=strides,
                                        padding='VALID')

            with tf.variable_scope('sub_unit0', reuse=tf.AUTO_REUSE):
                # x = bn1(x, training=tf.logical_not(is_eval))
                x = bn1(x, training=True)
                x = tf.nn.relu(x)

                x = c1(x, sample)

            with tf.variable_scope('sub_unit1', reuse=tf.AUTO_REUSE):
                # x = bn2(x, training=tf.logical_not(is_eval))
                x = bn2(x, training=True)
                x = tf.nn.relu(x)

                x = c2(x, sample)

            # Add the residual
            with tf.variable_scope('sub_unit_add'):
                # Handle differences in input and output filter sizes
                out_filters = x.get_shape().as_list()[-1]
                if in_filters < out_filters:
                    orig_x = tf.pad(
                        tensor=orig_x,
                        paddings=[[0, 0]] * (
                                len(x.get_shape().as_list())
                                - 1) + [[int(np.floor((out_filters
                                                       - in_filters) / 2.)),
                                         int(np.ceil((out_filters
                                                      - in_filters) / 2.))]])

                x += orig_x

            return x

        x = all_layers['conv1'](x, sample)

        for scale, filter in enumerate(filters[1:]):
            s = 'scale{}'.format(scale)
            stride = strides[scale + 1]
            for res_unit in range(num_units):
                r = 'unit{}'.format(res_unit)
                with tf.variable_scope('unit_{}_{}'.format(scale, res_unit)):
                    if not layer_wise:
                        def c1(x, sample):
                            return all_layers[s][r]['conv1'](
                                s, r, stride, x, sample)

                        def c2(x, sample):
                            return all_layers[s][r]['conv2'](s, r, 1, x, sample)
                    else:
                        c1 = all_layers[s][r]['conv1']
                        c2 = all_layers[s][r]['conv2']

                    bn1 = all_layers[s][r]['bn1']
                    bn2 = all_layers[s][r]['bn2']
                    x = call_res_unit(
                        x, c1, c2, bn1, bn2,
                        [1, stride, stride, 1])
                    stride = 1

        x = tf.nn.relu(x)

        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')

        x = all_layers['last'](x, sample)

        return x

    output_ind = []
    if sample_output:
        output = []

        for i in range(num_samples):
            x = call_resnet(ops['inp_x'], i)

            x = tf.nn.softmax(x)
            output.append(x)

        x = tf.log(tf.add_n(output) / float(num_samples) + 1e-8)
    else:
        x = call_resnet(ops['inp_x'])
        output_ind.append(x)

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    reg_losses = tf.losses.get_regularization_losses()
    if len(reg_losses) > 0:
        ops['loss'] += tf.add_n(reg_losses)

    loss_grads = tf.gradients(ce, ops['inp_x'])[0]
    adv_data = ops['inp_x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_bbb_mnist(ops, init_var=-15, prior_scale=1., aligned_noise=False):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x_inp = tf.reshape(x, [-1, 28, 28, 1])

    c1 = layers.BBBConvLayer('c1', 1, 20, 5, 'VALID', init_var=init_var,
                             prior_scale=prior_scale,
                             aligned_noise=aligned_noise)

    c2 = layers.BBBConvLayer('c2', 20, 50, 5, 'VALID', init_var=init_var,
                             prior_scale=prior_scale,
                             aligned_noise=aligned_noise)

    fc1 = layers.BBBDenseLayer('fc1', 800, 500, init_var=init_var,
                               prior_scale=prior_scale,
                               aligned_noise=aligned_noise)

    fc2 = layers.BBBDenseLayer('fc2', 500, 10, init_var=init_var,
                               prior_scale=prior_scale)

    x = c1(x_inp)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = c2(x)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = tf.layers.flatten(x)

    x = fc1(x)
    x = tf.nn.relu(x)

    x = fc2(x)

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    reg_losses = tf.losses.get_regularization_losses()
    if len(reg_losses) > 0:
        ops['loss'] += tf.add_n(reg_losses)

    loss_grads = tf.gradients(ce, ops['x'])[0]
    adv_data = ops['x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_bbb_cifar_resnet(ops, init_var=-30, prior_scale=1.):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    filters = [16, 16, 32, 64]
    strides = [1, 2, 2, 2]
    num_units = 5

    weight_shapes = {}
    weight_shapes['conv1'] = {
        'w': [3, 3, 3, filters[0]],
        'b': [filters[0]],
    }

    weight_shapes['last'] = {
        'w': [filters[-1], 5],
        'b': [5],
    }

    old_filter = filters[0]
    for scale, filter in enumerate(filters[1:]):
        s = 'scale{}'.format(scale)
        weight_shapes[s] = {}
        for res_unit in range(num_units):
            r = 'unit{}'.format(res_unit)
            weight_shapes[s][r] = {
                'conv1': {'w': [3, 3, old_filter, filter],
                          'b': [filter]},
                'conv2': {'w': [3, 3, filter, filter],
                          'b': [filter]},
            }

            old_filter = filter

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x, is_eval = get_cifar_image(ops)
    ops['is_eval'] = is_eval

    ops['inp_x'] = x

    h_use_bias = True

    print('Building weights for:\n{}'.format(weight_shapes))

    all_layers = {}

    w_shape = weight_shapes['conv1']['w']

    all_layers['conv1'] = layers.BBBConvLayer(
        'conv1', w_shape[-2], w_shape[-1], w_shape[0],
        init_var=init_var,
        prior_scale=prior_scale, strides=[1, strides[0], strides[0], 1],)

    for scale, filter in enumerate(filters[1:]):
        s = 'scale{}'.format(scale)
        all_layers[s] = {}
        stride = strides[scale + 1]
        for res_unit in range(num_units):
            r = 'unit{}'.format(res_unit)
            all_layers[s][r] = {}
            w_shape = weight_shapes[s][r]['conv1']['w']

            all_layers[s][r]['bn1'] = tf.layers.BatchNormalization(
                virtual_batch_size=1)
            all_layers[s][r]['bn2'] = tf.layers.BatchNormalization(
                virtual_batch_size=1)

            all_layers[s][r]['conv1'] = layers.BBBConvLayer(
                '{}/{}/conv1'.format(s, r),
                w_shape[-2], w_shape[-1], w_shape[0],
                init_var=init_var,
                prior_scale=prior_scale,
                strides=[1, stride, stride, 1])

            all_layers[s][r]['conv2'] = layers.BBBConvLayer(
                '{}/{}/conv2'.format(s, r),
                w_shape[-1], w_shape[-1], w_shape[0],
                init_var=init_var,
                prior_scale=prior_scale,
                strides=[1, 1, 1, 1])

            stride = 1

    w_shape = weight_shapes['last']['w']

    all_layers['last'] = layers.BBBDenseLayer(
        'last', filters[-1], w_shape[-1],
        init_var=init_var,
        prior_scale=prior_scale)

    def call_resnet(x, sample=0):
        def call_res_unit(x, c1, c2, bn1, bn2, strides):
            in_filters = x.get_shape().as_list()[-1]

            orig_x = x
            if np.prod(strides) != 1:
                orig_x = tf.nn.avg_pool(orig_x, ksize=strides, strides=strides,
                                        padding='VALID')

            with tf.variable_scope('sub_unit0', reuse=tf.AUTO_REUSE):
                # x = bn1(x, training=tf.logical_not(is_eval))
                x = bn1(x, training=True)
                x = tf.nn.relu(x)

                x = c1(x, sample)

            with tf.variable_scope('sub_unit1', reuse=tf.AUTO_REUSE):
                # x = bn2(x, training=tf.logical_not(is_eval))
                x = bn2(x, training=True)
                x = tf.nn.relu(x)

                x = c2(x, sample)

            # Add the residual
            with tf.variable_scope('sub_unit_add'):
                # Handle differences in input and output filter sizes
                out_filters = x.get_shape().as_list()[-1]
                if in_filters < out_filters:
                    orig_x = tf.pad(
                        tensor=orig_x,
                        paddings=[[0, 0]] * (
                                len(x.get_shape().as_list())
                                - 1) + [[int(np.floor((out_filters
                                                       - in_filters) / 2.)),
                                         int(np.ceil((out_filters
                                                      - in_filters) / 2.))]])

                x += orig_x

            return x

        x = all_layers['conv1'](x, sample)

        for scale, filter in enumerate(filters[1:]):
            s = 'scale{}'.format(scale)
            stride = strides[scale + 1]
            for res_unit in range(num_units):
                r = 'unit{}'.format(res_unit)
                with tf.variable_scope('unit_{}_{}'.format(scale, res_unit)):
                    c1 = all_layers[s][r]['conv1']
                    c2 = all_layers[s][r]['conv2']

                    bn1 = all_layers[s][r]['bn1']
                    bn2 = all_layers[s][r]['bn2']
                    x = call_res_unit(
                        x, c1, c2, bn1, bn2,
                        [1, stride, stride, 1])
                    stride = 1

        x = tf.nn.relu(x)

        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')

        x = all_layers['last'](x, sample)

        return x

    output_ind = []

    x = call_resnet(ops['inp_x'])
    output_ind.append(x)

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    reg_losses = tf.losses.get_regularization_losses()
    if len(reg_losses) > 0:
        ops['loss'] += tf.add_n(reg_losses)

    loss_grads = tf.gradients(ce, ops['inp_x'])[0]
    adv_data = ops['inp_x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_mnf_mnist(ops):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    learn_p = False

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x_inp = tf.reshape(x, [-1, 28, 28, 1])

    c1 = layers.MNFConvLayer('c1', 1, 20, 5, 'VALID',
                             thres_var=0.5, learn_p=learn_p)

    c2 = layers.MNFConvLayer('c2', 20, 50, 5, 'VALID',
                             thres_var=0.5, learn_p=learn_p)

    fc1 = layers.MNFDenseLayer('fc1', 800, 500,
                               thres_var=0.5, learn_p=learn_p)

    fc2 = layers.MNFDenseLayer('fc2', 500, 10,
                               thres_var=0.5, learn_p=learn_p)

    x = c1(x_inp)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = c2(x)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = tf.layers.flatten(x)

    x = fc1(x)
    x = tf.nn.relu(x)

    x = fc2(x)

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    reg_losses = tf.losses.get_regularization_losses()
    if len(reg_losses) > 0:
        ops['loss'] += tf.add_n(reg_losses)

    loss_grads = tf.gradients(ce, ops['x'])[0]
    adv_data = ops['x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_mnf_cifar_resnet(ops, learn_p=False, thres_var=0.3):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    filters = [16, 16, 32, 64]
    strides = [1, 2, 2, 2]
    num_units = 5

    weight_shapes = {}
    weight_shapes['conv1'] = {
        'w': [3, 3, 3, filters[0]],
        'b': [filters[0]],
    }

    weight_shapes['last'] = {
        'w': [filters[-1], 5],
        'b': [5],
    }

    old_filter = filters[0]
    for scale, filter in enumerate(filters[1:]):
        s = 'scale{}'.format(scale)
        weight_shapes[s] = {}
        for res_unit in range(num_units):
            r = 'unit{}'.format(res_unit)
            weight_shapes[s][r] = {
                'conv1': {'w': [3, 3, old_filter, filter],
                          'b': [filter]},
                'conv2': {'w': [3, 3, filter, filter],
                          'b': [filter]},
            }

            old_filter = filter

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x, is_eval = get_cifar_image(ops)
    ops['is_eval'] = is_eval

    ops['inp_x'] = x

    print('Building weights for:\n{}'.format(weight_shapes))

    all_layers = {}

    w_shape = weight_shapes['conv1']['w']

    all_layers['conv1'] = layers.MNFConvLayer(
        'conv1', w_shape[-2], w_shape[-1], w_shape[0],
        learn_p=learn_p, thres_var=thres_var,
        strides=[1, strides[0], strides[0], 1])

    for scale, filter in enumerate(filters[1:]):
        s = 'scale{}'.format(scale)
        all_layers[s] = {}
        stride = strides[scale + 1]
        for res_unit in range(num_units):
            r = 'unit{}'.format(res_unit)
            all_layers[s][r] = {}
            w_shape = weight_shapes[s][r]['conv1']['w']

            all_layers[s][r]['bn1'] = tf.layers.BatchNormalization(
                virtual_batch_size=1)
            all_layers[s][r]['bn2'] = tf.layers.BatchNormalization(
                virtual_batch_size=1)

            all_layers[s][r]['conv1'] = layers.MNFConvLayer(
                '{}/{}/conv1'.format(s, r),
                w_shape[-2], w_shape[-1], w_shape[0],
                learn_p=learn_p, thres_var=thres_var,
                strides=[1, stride, stride, 1])

            all_layers[s][r]['conv2'] = layers.MNFConvLayer(
                '{}/{}/conv2'.format(s, r),
                w_shape[-1], w_shape[-1], w_shape[0],
                learn_p=learn_p, thres_var=thres_var,
                strides=[1, 1, 1, 1])

            stride = 1

    w_shape = weight_shapes['last']['w']

    all_layers['last'] = layers.MNFDenseLayer(
        'last', filters[-1], w_shape[-1],
        learn_p=learn_p, thres_var=thres_var)

    def call_resnet(x, sample=0):
        def call_res_unit(x, c1, c2, bn1, bn2, strides):
            in_filters = x.get_shape().as_list()[-1]

            orig_x = x
            if np.prod(strides) != 1:
                orig_x = tf.nn.avg_pool(orig_x, ksize=strides, strides=strides,
                                        padding='VALID')

            with tf.variable_scope('sub_unit0', reuse=tf.AUTO_REUSE):
                # x = bn1(x, training=tf.logical_not(is_eval))
                x = bn1(x, training=True)
                x = tf.nn.relu(x)

                x = c1(x, sample)

            with tf.variable_scope('sub_unit1', reuse=tf.AUTO_REUSE):
                # x = bn2(x, training=tf.logical_not(is_eval))
                x = bn2(x, training=True)
                x = tf.nn.relu(x)

                x = c2(x, sample)

            # Add the residual
            with tf.variable_scope('sub_unit_add'):
                # Handle differences in input and output filter sizes
                out_filters = x.get_shape().as_list()[-1]
                if in_filters < out_filters:
                    orig_x = tf.pad(
                        tensor=orig_x,
                        paddings=[[0, 0]] * (
                                len(x.get_shape().as_list())
                                - 1) + [[int(np.floor((out_filters
                                                       - in_filters) / 2.)),
                                         int(np.ceil((out_filters
                                                      - in_filters) / 2.))]])

                x += orig_x

            return x

        x = all_layers['conv1'](x, sample)

        for scale, filter in enumerate(filters[1:]):
            s = 'scale{}'.format(scale)
            stride = strides[scale + 1]
            for res_unit in range(num_units):
                r = 'unit{}'.format(res_unit)
                with tf.variable_scope('unit_{}_{}'.format(scale, res_unit)):
                    c1 = all_layers[s][r]['conv1']
                    c2 = all_layers[s][r]['conv2']

                    bn1 = all_layers[s][r]['bn1']
                    bn2 = all_layers[s][r]['bn2']
                    x = call_res_unit(
                        x, c1, c2, bn1, bn2,
                        [1, stride, stride, 1])
                    stride = 1

        x = tf.nn.relu(x)

        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')

        x = all_layers['last'](x, sample)

        return x

    output_ind = []

    x = call_resnet(ops['inp_x'])
    output_ind.append(x)

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    loss_grads = tf.gradients(ce, ops['inp_x'])[0]
    adv_data = ops['inp_x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_vanilla_mnist(ops, prior_scale=1.):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x_inp = tf.reshape(x, [-1, 28, 28, 1])

    regularizer = tf.contrib.layers.l2_regularizer(scale=1. / prior_scale)

    x = tf.layers.conv2d(inputs=x_inp, kernel_size=5, filters=20,
                         activation=tf.nn.relu, padding='VALID',
                         kernel_regularizer=regularizer,
                         kernel_initializer=tf.variance_scaling_initializer())
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = tf.layers.conv2d(inputs=x, kernel_size=5, filters=50,
                         activation=tf.nn.relu, padding='VALID',
                         kernel_regularizer=regularizer,
                         kernel_initializer=tf.variance_scaling_initializer())
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = tf.layers.flatten(x)

    x = tf.layers.dense(inputs=x, units=500, activation=tf.nn.relu,
                        kernel_regularizer=regularizer,
                        kernel_initializer=tf.variance_scaling_initializer())

    x = tf.layers.dense(inputs=x, units=10,
                        kernel_regularizer=regularizer,
                        kernel_initializer=tf.variance_scaling_initializer())

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    loss_grads = tf.gradients(ce, ops['x'])[0]
    adv_data = ops['x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_vanilla_cifar_resnet(ops, prior_scale=1.):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    filters = [16, 16, 32, 64]
    strides = [1, 2, 2, 2]
    num_units = 5
    num_classes = 5

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x, is_eval = get_cifar_image(ops)
    ops['is_eval'] = is_eval

    ops['inp_x'] = x

    regularizer = tf.contrib.layers.l2_regularizer(scale=1. / prior_scale)

    def res_unit(x, out_filters, stride=1):
        strides = [1, stride, stride, 1]
        in_filters = x.get_shape().as_list()[-1]

        orig_x = x
        if np.prod(strides) != 1:
            orig_x = tf.nn.avg_pool(orig_x, ksize=strides,
                                    strides=strides, padding='VALID')

        with tf.variable_scope('sub_unit0'):
            x = tf.layers.batch_normalization(
                x, virtual_batch_size=1, training=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(
                inputs=x, kernel_size=3, filters=out_filters, padding='SAME',
                kernel_regularizer=regularizer, strides=stride,
                kernel_initializer=tf.variance_scaling_initializer())

        with tf.variable_scope('sub_unit1'):
            x = tf.layers.batch_normalization(
                x, virtual_batch_size=1, training=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(
                inputs=x, kernel_size=3, filters=out_filters, padding='SAME',
                kernel_regularizer=regularizer,
                kernel_initializer=tf.variance_scaling_initializer())

        # Add the residual
        with tf.variable_scope('sub_unit_add'):

            # Handle differences in input and output filter sizes
            if in_filters < out_filters:
                orig_x = tf.pad(
                    tensor=orig_x,
                    paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                        int(np.floor((out_filters - in_filters) / 2.)),
                        int(np.ceil((out_filters - in_filters) / 2.))]])

            x += orig_x

        return x

    # init_conv
    x = tf.layers.conv2d(
        inputs=x, kernel_size=3, filters=filters[0], padding='SAME',
        kernel_regularizer=regularizer,
        kernel_initializer=tf.variance_scaling_initializer())

    for scale in range(1, len(filters)):
        with tf.variable_scope('unit_{}_0'.format(scale)):
            x = res_unit(x, filters[scale], strides[scale])

        for unit in range(1, num_units):
            with tf.variable_scope('unit_{}_{}'.format(scale, unit)):
                x = res_unit(x, filters[scale])

    x = tf.layers.batch_normalization(x, virtual_batch_size=1, training=True)
    x = tf.nn.relu(x)

    x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
    # logits
    x = tf.layers.dense(inputs=x, units=num_classes,
                        kernel_regularizer=regularizer,
                        kernel_initializer=tf.variance_scaling_initializer())

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    loss_grads = tf.gradients(ce, ops['inp_x'])[0]
    adv_data = ops['inp_x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_dropout_mnist(ops, prior_scale=1., keep_prob=0.5):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x_inp = tf.reshape(x, [-1, 28, 28, 1])

    regularizer = tf.contrib.layers.l2_regularizer(scale=1. / prior_scale)

    x = tf.layers.conv2d(inputs=x_inp, kernel_size=5, filters=20,
                         activation=tf.nn.relu, padding='VALID',
                         kernel_regularizer=regularizer,
                         kernel_initializer=tf.variance_scaling_initializer())
    x = tf.nn.dropout(x, keep_prob)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = tf.layers.conv2d(inputs=x, kernel_size=5, filters=50,
                         activation=tf.nn.relu, padding='VALID',
                         kernel_regularizer=regularizer,
                         kernel_initializer=tf.variance_scaling_initializer())
    x = tf.nn.dropout(x, keep_prob)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

    x = tf.layers.flatten(x)

    x = tf.layers.dense(inputs=x, units=500, activation=tf.nn.relu,
                        kernel_regularizer=regularizer,
                        kernel_initializer=tf.variance_scaling_initializer())
    x = tf.nn.dropout(x, keep_prob)

    x = tf.layers.dense(inputs=x, units=10,
                        kernel_regularizer=regularizer,
                        kernel_initializer=tf.variance_scaling_initializer())

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    loss_grads = tf.gradients(ce, ops['x'])[0]
    adv_data = ops['x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_dropout_cifar_resnet(ops, prior_scale=1., keep_prob=0.5):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    filters = [16, 16, 32, 64]
    strides = [1, 2, 2, 2]
    num_units = 5
    num_classes = 5

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x, is_eval = get_cifar_image(ops)
    ops['is_eval'] = is_eval

    ops['inp_x'] = x

    regularizer = tf.contrib.layers.l2_regularizer(scale=1. / prior_scale)

    def res_unit(x, out_filters, stride=1):
        strides = [1, stride, stride, 1]
        in_filters = x.get_shape().as_list()[-1]

        orig_x = x
        if np.prod(strides) != 1:
            orig_x = tf.nn.avg_pool(orig_x, ksize=strides,
                                    strides=strides, padding='VALID')

        with tf.variable_scope('sub_unit0'):
            x = tf.layers.batch_normalization(
                x, virtual_batch_size=1, training=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(
                inputs=x, kernel_size=3, filters=out_filters, padding='SAME',
                kernel_regularizer=regularizer, strides=stride,
                kernel_initializer=tf.variance_scaling_initializer())
            x = tf.nn.dropout(x, keep_prob)

        with tf.variable_scope('sub_unit1'):
            x = tf.layers.batch_normalization(
                x, virtual_batch_size=1, training=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(
                inputs=x, kernel_size=3, filters=out_filters, padding='SAME',
                kernel_regularizer=regularizer,
                kernel_initializer=tf.variance_scaling_initializer())
            x = tf.nn.dropout(x, keep_prob)

        # Add the residual
        with tf.variable_scope('sub_unit_add'):

            # Handle differences in input and output filter sizes
            if in_filters < out_filters:
                orig_x = tf.pad(
                    tensor=orig_x,
                    paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                        int(np.floor((out_filters - in_filters) / 2.)),
                        int(np.ceil((out_filters - in_filters) / 2.))]])

            x += orig_x

        return x

    # init_conv
    x = tf.layers.conv2d(
        inputs=x, kernel_size=3, filters=filters[0], padding='SAME',
        kernel_regularizer=regularizer,
        kernel_initializer=tf.variance_scaling_initializer())
    x = tf.nn.dropout(x, keep_prob)

    for scale in range(1, len(filters)):
        with tf.variable_scope('unit_{}_0'.format(scale)):
            x = res_unit(x, filters[scale], strides[scale])

        for unit in range(1, num_units):
            with tf.variable_scope('unit_{}_{}'.format(scale, unit)):
                x = res_unit(x, filters[scale])

    x = tf.layers.batch_normalization(x, virtual_batch_size=1, training=True)
    x = tf.nn.relu(x)

    x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
    # logits
    x = tf.layers.dense(inputs=x, units=num_classes,
                        kernel_regularizer=regularizer,
                        kernel_initializer=tf.variance_scaling_initializer())

    ops['logits'] = x
    # build function to hold predictions
    pred = tf.argmax(ops['logits'], -1, output_type=tf.int32)

    # create tensor to calculate accuracy of predictions
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, ops['y']), tf.float32))
    ops['acc'] = acc

    probs = tf.nn.softmax(ops['logits'])
    ops['probs'] = probs

    ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=ops['logits'],
            labels=ops['y']))

    ops['loss'] = ce

    loss_grads = tf.gradients(ce, ops['inp_x'])[0]
    adv_data = ops['inp_x'] + adv_eps * tf.sign(loss_grads)
    ops['adv_data'] = adv_data

    return ops


def get_ensemble_mnist(ops):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x_inp = tf.reshape(x, [-1, 28, 28, 1])

    adv_alpha = 0.5
    # adv_eps = 1e-2
    ops['logits'] = []
    ops['acc'] = []
    ops['probs'] = []
    ops['loss'] = []
    ops['adv_data'] = []
    ops['tot_loss'] = []
    for i in range(10):
        with tf.variable_scope('ens{}'.format(i)):
            conv1 = tf.layers.Conv2D(
                kernel_size=5, filters=20,
                activation=tf.nn.relu, padding='VALID',
                kernel_initializer=tf.variance_scaling_initializer())
            conv2 = tf.layers.Conv2D(
                kernel_size=5, filters=50,
                activation=tf.nn.relu, padding='VALID',
                kernel_initializer=tf.variance_scaling_initializer())
            fc1 = tf.layers.Dense(
                units=500, activation=tf.nn.relu,
                kernel_initializer=tf.variance_scaling_initializer())
            fc2 = tf.layers.Dense(
                units=10,
                kernel_initializer=tf.variance_scaling_initializer())

            def get_out(h):
                h = conv1(h)
                h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

                h = conv2(h)
                h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

                h = tf.layers.flatten(h)

                h = fc1(h)

                h = fc2(h)

                return h

            logits = get_out(x_inp)
            ops['logits'].append(logits)
            # build function to hold predictions
            pred = tf.argmax(logits, -1, output_type=tf.int32)

            # create tensor to calculate accuracy of predictions
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))
            ops['acc'].append(acc)

            probs = tf.nn.softmax(logits)
            ops['probs'].append(probs)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                               labels=y))
            ops['loss'].append(loss)

            loss_grads = tf.gradients(adv_alpha * loss, ops['x'])[0]
            adv_data = ops['x'] + adv_eps * tf.sign(loss_grads)
            adv_data = tf.stop_gradient(adv_data)
            ops['adv_data'].append(adv_data)

            adv_logits = get_out(tf.reshape(adv_data, [-1, 28, 28, 1]))
            adv_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=adv_logits,
                    labels=y))

            tot_loss = adv_alpha * loss + (1 - adv_alpha) * adv_loss

            ops['tot_loss'].append(tot_loss)

    return ops


def get_ensemble_cifar_resnet(ops):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int32, [None])
    adv_eps = tf.placeholder_with_default(1e-2, [])

    filters = [16, 16, 32, 64]
    strides = [1, 2, 2, 2]
    num_units = 5
    num_classes = 5
    num_ensembles = 5

    ops['x'] = x
    ops['y'] = y
    ops['adv_eps'] = adv_eps

    x, is_eval = get_cifar_image(ops)
    ops['is_eval'] = is_eval

    ops['inp_x'] = x

    adv_alpha = 0.5
    # adv_eps = 1e-2
    ops['logits'] = []
    ops['acc'] = []
    ops['probs'] = []
    ops['loss'] = []
    ops['adv_data'] = []
    ops['tot_loss'] = []

    def apply_resunit(x, layer_dict, stride):
        stride = [1, stride, stride, 1]
        in_filters = x.get_shape().as_list()[-1]

        orig_x = x
        if np.prod(stride) != 1:
            orig_x = tf.nn.avg_pool(orig_x, ksize=stride, strides=stride,
                                    padding='VALID')

        with tf.variable_scope('sub_unit0'):
            x = layer_dict['bn1'](x, training=True)
            x = tf.nn.relu(x)

            x = layer_dict['conv1'](x)

        with tf.variable_scope('sub_unit1'):
            x = layer_dict['bn2'](x, training=True)
            x = tf.nn.relu(x)

            x = layer_dict['conv2'](x)

        out_filters = x.get_shape().as_list()[-1]
        # Add the residual
        with tf.variable_scope('sub_unit_add'):

            # Handle differences in input and output filter sizes
            if in_filters < out_filters:
                orig_x = tf.pad(
                    tensor=orig_x,
                    paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                        int(np.floor((out_filters - in_filters) / 2.)),
                        int(np.ceil((out_filters - in_filters) / 2.))]])

            x += orig_x

        return x

    for i in range(num_ensembles):
        with tf.variable_scope('ens{}'.format(i)):
            init_conv = tf.layers.Conv2D(
                kernel_size=3, filters=filters[0],
                padding='SAME',
                kernel_initializer=tf.variance_scaling_initializer())

            res_units = []
            for scale in range(1, len(filters)):
                with tf.variable_scope('unit_{}_0'.format(scale)):
                    res_dict = {
                        'bn1': tf.layers.BatchNormalization(
                            virtual_batch_size=1, name='bn1'),
                        'conv1': tf.layers.Conv2D(
                            kernel_size=3, filters=filters[scale],
                            padding='SAME', strides=strides[scale],
                            name='conv1',
                            kernel_initializer=tf.variance_scaling_initializer()),
                        'bn2': tf.layers.BatchNormalization(
                            virtual_batch_size=1, name='bn2'),
                        'conv2': tf.layers.Conv2D(
                            kernel_size=3, filters=filters[scale],
                            padding='SAME', name='conv2',
                            kernel_initializer=tf.variance_scaling_initializer())
                    }

                    res_units.append(res_dict)

                for unit in range(1, num_units):
                    with tf.variable_scope('unit_{}_{}'.format(scale, unit)):
                        res_dict = {
                            'bn1': tf.layers.BatchNormalization(
                                virtual_batch_size=1, name='bn1'),
                            'conv1': tf.layers.Conv2D(
                                kernel_size=3, filters=filters[scale],
                                padding='SAME', name='conv1',
                                kernel_initializer=tf.variance_scaling_initializer()),
                            'bn2': tf.layers.BatchNormalization(
                                virtual_batch_size=1, name='bn2'),
                            'conv2': tf.layers.Conv2D(
                                kernel_size=3, filters=filters[scale],
                                padding='SAME', name='conv2',
                                kernel_initializer=tf.variance_scaling_initializer())
                        }
                        res_units.append(res_dict)
            last_bn = tf.layers.BatchNormalization(
                virtual_batch_size=1, name='last_bn')
            last = tf.layers.Dense(
                units=num_classes,
                kernel_initializer=tf.variance_scaling_initializer())

            def get_out(h):
                h = init_conv(h)

                i = 0
                for scale in range(1, len(filters)):
                    with tf.variable_scope('unit_{}_0'.format(scale)):
                        h = apply_resunit(h, res_units[i], strides[scale])
                    i += 1
                    for unit in range(1, num_units):
                        with tf.variable_scope(
                                'unit_{}_{}'.format(scale, unit)):
                            h = apply_resunit(h, res_units[i], 1)
                        i += 1

                h = last_bn(h, training=True)
                h = tf.nn.relu(h)
                h = tf.reduce_mean(h, [1, 2])

                h = last(h)

                return h

            logits = get_out(x)
            ops['logits'].append(logits)
            # build function to hold predictions
            pred = tf.argmax(logits, -1, output_type=tf.int32)

            # create tensor to calculate accuracy of predictions
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))
            ops['acc'].append(acc)

            probs = tf.nn.softmax(logits)
            ops['probs'].append(probs)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                               labels=y))
            ops['loss'].append(loss)

            loss_grads = tf.gradients(adv_alpha * loss, ops['inp_x'])[0]
            adv_data = ops['inp_x'] + adv_eps * tf.sign(loss_grads)
            adv_data = tf.stop_gradient(adv_data)
            ops['adv_data'].append(adv_data)

            adv_logits = get_out(tf.reshape(adv_data, [-1, 24, 24, 3]))
            adv_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=adv_logits,
                    labels=y))

            tot_loss = adv_alpha * loss + (1 - adv_alpha) * adv_loss

            ops['tot_loss'].append(tot_loss)

    return ops
