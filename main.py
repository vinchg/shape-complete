import time
import argparse
import tensorflow as tf
from epnet import EPNet as ep
"""
This file provides configuration to build U-NET for semantic segmentation.

"""


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_string('data_type', '3D', '2D or 3D')
    flags.DEFINE_integer('train_step', 125, '# of step for trainingt')
    flags.DEFINE_integer('test_step', 1, '# of step to test a model')
    flags.DEFINE_integer('save_step', 5, '# of step to save a model')
    flags.DEFINE_integer('summary_step', 1, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    # data
    flags.DEFINE_string('data_dir', './h5/', 'Name of data directory')
    flags.DEFINE_integer('num_train_files', 16, 'Number of training files')
    flags.DEFINE_integer('num_test_files', 4, 'Number of test files')
    # flags.DEFINE_string('train_data', 'train', 'Training data')
    # flags.DEFINE_string('valid_data', 'validation', 'Validation data')
    # flags.DEFINE_string('test_data', 'test', 'Testing data')
    flags.DEFINE_integer('batch', 64, 'batch size')
    flags.DEFINE_integer('channel', 2, 'channel size')
    flags.DEFINE_integer('height', 32, 'height size')
    flags.DEFINE_integer('width', 32, 'width size')
    flags.DEFINE_integer('depth', 32, 'depth size')
    flags.DEFINE_integer('channel_axis', 4, 'channel axis')
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('sampledir', './samples/', 'Sample directory')
    flags.DEFINE_string('model_name', 'shape_complete', 'Model file name')
    flags.DEFINE_integer('reload_step', -1, 'Reload step to continue training')
    #flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('network_down', 5, 'network depth for EPN-Net')
    flags.DEFINE_integer('fcn_depth',2,'fully connected layers')
    flags.DEFINE_integer('pad',1,'padding for convolutions')
    flags.DEFINE_integer('class_num', 2, 'output class number')
    flags.DEFINE_integer('stride', 2, 'stride for convolution')
    flags.DEFINE_integer('start_channel_num', 80,
                         'start number of outputs for the first conv layer')
    ##### Architecture Type ####
    #Note: when decoder_type is set as deconv, the pixel_name flag does not affect architecture
    flags.DEFINE_string(
        'decoder_type', 'pdcn',
        'define architecture to be used: deconv, pdcn')
    #Now only used to specify type of pixel_dcl
    flags.DEFINE_string(
        'pixel_name','pixel_dcl',
        'Use which pixel_deconv op in decoder: pixel_dcl, ipixel_dcl')
    flags.DEFINE_string(
        'conv_name', 'conv2d',
        'Use which conv op in decoder: conv2d or ipixel_cl')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train','test','predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    else:
        #tf.ConfigProto is used to only require as much GPU mem as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model = ep(tf.Session(config=config), configure())
        getattr(model, args.action)()
        #getattr(model, 'train')()

        print('done')

if __name__ == '__main__':
    # configure which gpu or cpu to use
    tf.app.run()
