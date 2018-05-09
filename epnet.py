import tensorflow as tf
import h5py
import os
import numpy as np
import time
from copy import deepcopy
from utils import encoder_decoder_architectures as eda
from utils import ops
import random

class EPNet():
    def __init__(self, sess, conf):
        self.sess= sess
        self.conf = conf
        self.conv_size = (4,4,4)
        self.axis, self.channel_axis = (1,2,3),4
        self.input_shape = [
             conf.batch, conf.height, conf.width, conf.depth, conf.channel,
        ]
        self.output_shape = [conf.batch, conf.height, conf.width, conf.depth]
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sampledir):
            os.makedirs(conf.sampledir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def configure_networks(self):
        self.build_network()
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.inputs = tf.placeholder(tf.float32, self.input_shape, name = 'inputs')
        self.mask = tf.placeholder(tf.float32, self.output_shape, name='mask')
        self.annotations = tf.placeholder(tf.float32, self.output_shape, name='annonations')
        self.predictions = self.inference(self.inputs)
        self.cal_loss()

    def cal_loss(self):
        self.masked_annotations = np.multiply(self.annotations, self.mask)  # Apply mask
        predictions = np.multiply(self.predictions, self.mask)  # Apply mask
        losses = tf.losses.absolute_difference(self.masked_annotations, predictions, scope="loss/losses")
        self.loss_op = tf.reduce_mean(losses,name = 'loss/loss_op')
        self.correct_prediction = tf.equal(self.annotations, self.predictions)
        self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32,name = 'accuracy/cast'))

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name + '/loss', self.loss_op))
        #summarys.append(tf.summary.scalar(name + '/accuracy', self.accuracy_op))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        #Flow of the model, uses reshapes between portions of the network to ensure
        #compatible layers from encoder -> fcn -> decoder
        outputs = inputs
        down_outputs = []
        outputs = self.encoder(outputs, down_outputs)
        outputs = self.bottom_fcn(outputs)
        outputs = tf.reshape(outputs, [-1,1,1,1,outputs.shape[1].value])
        outputs = self.decoder(outputs, down_outputs)
        outputs = tf.reshape(outputs,[-1,self.conf.width,self.conf.height,self.conf.depth])
        return outputs

    def encoder(self, inputs, down_outputs):
        #Creates encoder portion of the network by running methods in encoder_decoder_architectures.py
        outputs = inputs
        for layer_index in range(self.conf.network_down - 1):
            is_first = True if not layer_index else False
            is_last = True if layer_index == self.conf.network_down - 2 else False
            name = 'encoder%s' % layer_index
            outputs = self.encoder_block()(
                 self.conf, outputs, name, down_outputs, self.conv_size, first = is_first,last = is_last
            )
        return outputs

    def bottom_fcn(self, outputs):
        for layer_index in range(self.conf.fcn_depth):
            is_first = True if not layer_index else False
            name = 'fcn%s' % layer_index
            outputs = self.fcn_block()(
                self.conf, outputs, name, is_first)
        return outputs

    def decoder(self, outputs, down_outputs):
        for layer_index in range(self.conf.network_down - 2, -1, -1):
            is_last = True if not layer_index else False
            is_first = True if layer_index == self.conf.network_down - 2 else False
            name = 'decoder%s' %layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.decoder_block()(
               self.conf, outputs, name, down_inputs, self.conv_size, is_first, is_last
            )
        return outputs

    def encoder_block(self):
        #For clean code, run construction_encoder_block in eda from this method instead of encoder directly
        return getattr(eda, 'construct_encoder_block')

    def fcn_block(self):
        # For clean code, run construct_fcn in eda from this method instead of bottom_fcn directly
        return getattr(eda, 'construct_fcn')

    def decoder_block(self):
        #For clean code, tun construct_decoder_pcdn/deconv(depending on flags)_block in eda from this method
        #instead of decoder
        return getattr(eda, 'construct_decoder_' + self.conf.decoder_type + '_block')

    def load_data(self, set, type):
        path = self.conf.data_dir + type + '_%d.h5' % set
        print('Loading:', path)
        self.f = h5py.File(path, 'r')
        x = self.f['data']    # [index, images/spaces, 32, 32, 32] - Range: [0-9999, 0-1, image]
        y = self.f['target']  # [index, labels, 32, 32, 32] - Range: [0-9999, 0, image]
        return x, y
        
    def gen_indexes(self, data):
        return np.random.permutation(range(data.shape[0]))
        
    def get_batch(self, x, y):
        next_index = self.current_index + self.conf.batch
        batch_indexes = list(self.indexes[self.current_index:next_index])
        self.current_index = next_index
        batch_indexes = sorted(set(batch_indexes))
        try:
            return x[batch_indexes], y[batch_indexes]
        except Exception as e:
            print(e)
            return np.empty(0), np.empty(0)
        
    def process_batch(self, x_batch, y_batch):
        images = x_batch[:, 0]
        spaces = x_batch[:, 1]
        mask = deepcopy(spaces)
        mask[mask == 1] = 0
        mask[mask == -1] = 1
        input = np.stack((images, spaces), axis=4)
        input = input[:, :, :, :, 0:self.conf.channel]
        labels = y_batch[:, 0] # [index, images(32x32x32)]
        
        return input, mask, labels
       
    def train(self):
        print('---->Training 3DEPN')
        train_step_start = 0
        if (self.reload(self.conf.reload_step)): # Attempt to reload model
            train_step_start = self.conf.reload_step
        for epoch in range(train_step_start, self.conf.train_step):
            start = time.time()
            for file_itr in range(self.conf.num_train_files):
                x, y = self.load_data(file_itr, 'train')
                self.indexes = self.gen_indexes(x)
                self.current_index = 0

                while True:
                    x_batch, y_batch = self.get_batch(x, y)
                    if x_batch.shape[0] < self.conf.batch:
                        break
                    input, mask, labels = self.process_batch(x_batch, y_batch)

                    feed_dict = {self.inputs: input, self.mask: mask, self.annotations: labels}
                    loss, _ = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
                    print(self.current_index, '/', len(self.indexes), ' Training loss:', loss)
                    
                self.f.close()

            if epoch % self.conf.save_step == 0:
                self.save(epoch)
            if epoch % self.conf.summary_step == 0:


                f_num = random.randint(0,3)
                #Load testing batch
                x, y = self.load_data(f_num, 'test')
                self.indexes = self.gen_indexes(x)
                self.current_index = 0

                x_batch, y_batch = self.get_batch(x, y)
                input, mask, labels = self.process_batch(x_batch, y_batch)
                test_feed_dict = {self.inputs: input, self.mask: mask, self.annotations: labels}

                train_loss, train_summary = self.sess.run([self.loss_op, self.train_summary], feed_dict=feed_dict)
                test_loss, test_summary = self.sess.run([self.loss_op, self.valid_summary],feed_dict=test_feed_dict)
                self.f.close()

                self.save_summary(train_summary, epoch)
                self.save_summary(test_summary, epoch)

                print('--Training loss:', train_loss, '  --Validation loss:', test_loss, '   --Epoch: ', epoch)

            
            end = time.time()
            print("------TIME------")
            print(end - start)

        print('Training Complete')            

    def test(self):
        print('--->Testing 3DEPN')
        if self.conf.reload_step >= 0: # Reload model if exists
            if (self.reload(self.conf.reload_step)):
                train_step_start = self.conf.reload_step
        else:
            train_step_start = 0
        
        count = 0
        losses = []
        accuracies = []
        
        #for file_itr in range(self.conf.num_test_files):
        x, y = self.load_data(4, 'test')
        self.indexes = self.gen_indexes(x)
        self.current_index = 0
        
        while True:
            x_batch, y_batch = self.get_batch(x, y)
            if x_batch.shape[0] < self.conf.batch:
                break
            input, mask, labels = self.process_batch(x_batch, y_batch)
            
            feed_dict = {self.inputs:input, self.mask: mask, self.annotations:labels}
            loss, accuracy = self.sess.run([self.loss_op, self.accuracy_op], feed_dict=feed_dict)
            print(self.current_index, '/', len(self.indexes), ' Testing loss:', loss, '  --Accuracy:', accuracy)
            count += 1
            losses.append(loss)
            accuracies.append(accuracy)

        self.f.close()
            
        print('Loss: ', np.mean(losses))
        print('Accuracy: ', np.mean(accuracies))

    def predict(self):
        print('--->Prediction 3DEPN')
        if self.conf.reload_step >= 0:
            self.reload(self.conf.reload_step)

        x, y = self.load_data(0, 'train')
        self.indexes = self.gen_indexes(x)
        self.current_index = 0
        
        #predictions = []
        
        #while True:
        x_batch, y_batch = self.get_batch(x, y)
        input, masks, labels = self.process_batch(x_batch, y_batch)
        feed_dict = {self.inputs: input}

        #Got rid of the list of predictions.
        #predictions.append(self.sess.run([self.predictions], feed_dict=feed_dict))
        predictions = self.sess.run(self.predictions, feed_dict=feed_dict)
        
        print('----Predicions---')
        #Saves the input and predictions, the input is being spliced and only getting the 0 index(sdf representation).
        for i in range(10):
            # The i is the index of the batch, the colons : get every index in dimension, 0 index is the sdf representation.
            #Splits the input into two representations sdf = sign distance field, ku = known/unknown
            sdf = input[i, :, :, :, 0]
            #ku = input[i, :, :, :, 1]
            #Reshape prediction and ku to do element wise comparison to get rid of known space output from model
            #ku = np.reshape(ku, (32768, 1))
            #pred = np.reshape(predictions[i], (32768, 1))
            #unknown_pred = np.asarray([3 if y == 1 else x[0] for x, y in zip(pred, ku)])
            #Reshaping back to 3d representation
            #unknown_pred = np.reshape(unknown_pred, (32, 32, 32))
            #print(predictions[i])
            #print(ku)
            #Saving
            np.save('testing/input_{}'.format(i), sdf)
            np.save('testing/prediction_{}'.format(i), predictions[i])

    def save_summary(self,summary,step):
        print('---->Summarizing')
        self.writer.add_summary(summary, step)
        
    def save(self, step):
        print('---->Saving: ', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name + '_' + self.conf.decoder_type
        )
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name
        )
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path+'.meta'):
            print('---->no checkpoint',model_path)
            return False
        print('---->checkpoint found', model_path)
        print('Restoring Model')
        self.saver.restore(self.sess,model_path)
        return True
