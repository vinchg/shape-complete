import tensorflow as tf
from utils import ops


def construct_encoder_block(conf, inputs, name, down_outputs, conv_size, first=False, last=False):
    num_outputs = conf.start_channel_num if first else 2 * \
                                                            inputs.shape[conf.channel_axis].value
    pad = conf.pad if not last else conf.pad - 1
    stride = conf.stride if not last else (1, 1, 1)
    inputs = pad_input(inputs, pad)
    conv1 = tf.layers.conv3d(inputs, num_outputs, conv_size, strides=stride, name=name + '/conv1', activation=None,
                             use_bias=True)
    conv1_bn = tf.contrib.layers.batch_norm(
        conv1, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=name + '/conv1/batch_norm')
    down_outputs.append(conv1_bn)
    return conv1_bn

def construct_fcn(conf, inputs, name, first = False):
    num_outputs = inputs.shape[conf.channel_axis].value if first else inputs.shape[1].value
    inputs = tf.reshape(inputs,[-1,num_outputs]) if first else inputs
    fcn_1 = tf.layers.dense(inputs, num_outputs,activation=None, use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(),name=name)
    fcn_1_bn = tf.contrib.layers.batch_norm(
        fcn_1, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=name + '/fcn_1/batch_norm')
    return fcn_1_bn


def construct_decoder_deconv_block(conf, inputs, name, down_inputs, conv_size, first = False, last = False):
    num_outputs = 1 if last else int(inputs.shape[conf.channel_axis].value / 2)
    stride = (1,1,1) if first else conf.stride
    cat = tf.concat([inputs, down_inputs], conf.channel_axis, name=name + '/concat')
    deconv1 = tf.layers.conv3d_transpose(cat,num_outputs,conv_size if first else stride, strides=stride, name = name + '/deconv1', activation=None, use_bias=True)
    deconv1_bn = tf.contrib.layers.batch_norm(
        deconv1, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=name + '/deconv1/batch_norm')
    return deconv1_bn

def construct_decoder_pdcn_block(conf, inputs, name, down_inputs, conv_size, first = False, last = False):
    num_outputs = 1 if last else int(inputs.shape[conf.channel_axis].value / 2)
    cat = tf.concat([inputs, down_inputs], conf.channel_axis, name = name + '/concat')
    deconv1 = deconv_func(conf)(
        cat, num_outputs, (4, 4, 4), name + '/deconv1', conf.data_type
    )
    if first:
        deconv2 = deconv_func(conf)(
            deconv1, num_outputs, (4, 4, 4), name + '/deconv2', conf.data_type
        )
        return deconv2
    return deconv1

def deconv_func(conf):
    return getattr(ops, conf.pixel_name)

def pad_input(inputs,pad):
    paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]])
    inputs = tf.pad(inputs, paddings, "CONSTANT")
    return inputs