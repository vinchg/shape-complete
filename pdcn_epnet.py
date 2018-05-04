import tensorflow as tf
import h5py
import os
import numpy as np
import time
from utils import ops
from copy import deepcopy

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
        annotations = np.multiply(self.annotations, self.mask)  # Apply mask
        predictions = np.multiply(self.predictions, self.mask)  # Apply mask
        losses = tf.losses.absolute_difference(annotations, predictions, scope="loss/losses")
        self.loss_op = tf.reduce_mean(losses,name = 'loss/loss_op')
        self.correct_prediction = tf.equal(self.annotations, self.predictions)
        self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32,name = 'accuracy/cast'))

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name + '/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name + '/accuracy', self.accuracy_op))
        if name == 'valid':
            summarys.append(tf.summary.image(
                name + '/input', self.inputs, max_outputs=100))
            summarys.append(tf.summary.image(
                name +
                '/annotation', tf.cast(tf.expand_dims(
                    self.annotations, -1), tf.float32),
                max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        outputs = inputs
        down_outputs = []
        for layer_index in range(self.conf.network_down - 1):
            is_first = True if not layer_index else False
            is_last = True if layer_index == self.conf.network_down - 2 else False
            name = 'encoder%s' % layer_index
            outputs = self.construct_encoder_block(
                outputs,name,down_outputs,first = is_first,last = is_last
            )
            print('Down_Stream: ', outputs)
        for layer_index in range(self.conf.fcn_depth):
            is_first = True if not layer_index else False
            name = 'fcn%s' % layer_index
            outputs = self.construct_fcn(outputs, name, is_first)
            print('Bottom Layer', outputs)
        outputs = tf.reshape(outputs,[-1,1,1,1,outputs.shape[1].value])
        print('Reshaped: ', outputs)
        for layer_index in range(self.conf.network_down - 2, -1, -1):
            is_last = True if not layer_index else False
            is_first = True if layer_index == self.conf.network_down - 2 else False
            name = 'decoder%s' %layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.construct_decoder_block(
            outputs, name, down_inputs,is_first, is_last
            )
            print('Output: ', outputs)
        outputs = tf.reshape(outputs,[-1, self.conf.width, self.conf.height, self.conf.depth])
        return outputs

    def construct_encoder_block(self, inputs, name, down_outputs, first=False, last=False):
        num_outputs = self.conf.start_channel_num if first else 2 * \
                                                              inputs.shape[self.channel_axis].value
        pad = self.conf.pad if not last else self.conf.pad - 1
        stride = self.conf.stride if not last else (1,1,1)
        inputs = self.pad_input(inputs,pad)
        conv1 = tf.layers.conv3d(inputs,num_outputs,self.conv_size,strides=stride,name = name + '/conv1', activation=None,use_bias=True)
        conv1_bn = tf.contrib.layers.batch_norm(
            conv1, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
            epsilon=1e-5, scope=name + '/conv1/batch_norm')
        down_outputs.append(conv1_bn)
        return conv1_bn

    def construct_fcn(self,inputs,name,first = False):
        num_outputs = inputs.shape[self.channel_axis].value if first else inputs.shape[1].value
        inputs = tf.reshape(inputs,[-1,num_outputs]) if first else inputs
        fcn_1 = tf.layers.dense(inputs, num_outputs,activation=None, use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(),name=name)
        fcn_1_bn = tf.contrib.layers.batch_norm(
            fcn_1, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
            epsilon=1e-5, scope=name + '/fcn_1/batch_norm')
        return fcn_1_bn

    def construct_decoder_block(self,inputs, name, down_inputs, first = False, last = False):
        num_outputs = 1 if last else int(inputs.shape[self.channel_axis].value / 2)
        #stride = (1,1,1) if first else self.conf.stride
        cat = tf.concat([inputs, down_inputs], self.channel_axis, name = name + '/concat')
        #deconv1 = tf.layers.conv3d_transpose(cat,num_outputs,self.conv_size if first else stride, strides=stride, name = name + '/deconv1', activation=None, use_bias=True)
        #deconv1_bn = tf.contrib.layers.batch_norm(
        #    deconv1, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        #    epsilon=1e-5, scope=name + '/deconv1/batch_norm')
        print(cat)
        deconv1 = self.deconv_func()(
            cat, num_outputs, (4, 4, 4), name + '/deconv1', self.conf.data_type
        )
        if first:
            deconv2 = self.deconv_func()(
                deconv1, num_outputs, (4,4,4), name + '/deconv2', self.conf.data_type
            )
            return deconv2
        return deconv1

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def pad_input(self,inputs,pad):
        paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]])
        inputs = tf.pad(inputs, paddings, "CONSTANT")
        return inputs

    def load_data(self, set, type):
        path = self.conf.data_dir + type + '_%d.h5' % set
        #path = self.conf.data_dir + 'train_0.h5'
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
                loss, _, summary = self.sess.run(
                        [self.loss_op, self.train_op, self.train_summary],
                                feed_dict=feed_dict)
                print('--Training loss:', loss,'   --Epoch: ', epoch)
                self.save_summary(summary, epoch)
            
            end = time.time()
            print("------TIME------")
            print(end - start)                 
        
        print('Training Complete')            

    def test(self):
        print('--->Testing 3DEPN')
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        count = 0
        losses = []
        accuracies = []
        
        for file_itr in range(self.conf.num_test_files):
            x, y = self.load_data(file_itr, 'test')
            self.indexes = self.gen_indexes(x)
            self.current_index = 0
            
            while True:
                x_batch, y_batch = self.get_batch(x, y)
                if x_batch.shape[0] < self.conf.batch:
                    break
                input, mask, labels = self.process_batch(x_batch, y_batch)
                
                feed_dict = {self.inputs:input, self.mask: mask, self.annotations:labels}
                loss, accuracy = self.sess.run([self.loss_op, self.accuracy_op], feed_dict=feed_dict)
                print('----Testing loss:', loss, '  ----Accuracy:', accuracy)
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
        
        predictions = []
        
        #while True:
        x_batch, y_batch = self.get_batch(x, y)
        print(x_batch.shape[0])
        #if x_batch.shape[0] < self.conf.batch:
        #    break
        input, masks, labels = self.process_batch(x_batch, y_batch)

        feed_dict = {self.inputs: input}
        predictions.append(self.sess.run([self.predictions], feed_dict=feed_dict))
            
        print('----Predicions---')
        # TODO: Save predictions for visualization in MATLAB
        print(predictions[0].shape)
        np.save('predict0', np.asarray(predictions[0]))


    def save_summary(self,summary,step):
        print('---->Summarizing')
        self.writer.add_summary(summary,step)
        
    def save(self,step):
        print('---->Saving: ',step)
        checkpoint_path = os.path.join(
            self.conf.modeldir,self.conf.model_name
        )
        self.saver.save(self.sess,checkpoint_path,global_step=step)

    def reload(self,step):
        checkpoint_path = os.path.join(
            self.conf.modeldir,self.conf.model_name
        )
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('---->no checkpoint', model_path)
            return False
        print('---->checkpoint found', model_path)
        print('Restoring Model')
        self.saver.restore(self.sess,model_path)
        return True