# created by kenan@armoi.com

import tensorflow as tf

class Model:
    # def __init__(self, input_image, batch_size, keep_prop):
    #     self.data = input_image
    #     self.batch_size = batch_size
    #     self.keep_prop = keep_prop

    # Based on AlexNet,
    # add: batch normalization
    # delete: local_response_normalization layer, regulation term in each matrix multiplex
    # output is 28 float numbers which indicate the coordinate of 14 location points
    def inference(self, data, batch_size, keep_prop, is_training):
        # layer1
        # conv1 = self.conv(data=self.data, scan_window=[11,11,3,96], stride=[1,4,4,1],
        #                   padding=[[0,0],[0,0],[0,0]], pad_type='SAME')
        train_phase = tf.cond(is_training, lambda:True, lambda:False)

        with tf.variable_scope('layer1') as scope:
            conv1 = self.conv(data, [11,11,3,96], [1,4,4,1], 'SAME')
            pool1 = self.pool(conv1, [1,3,3,1], [1,2,2,1], 'VALID')
            BN_r1 = self.BN_relu(pool1, train_phase)

        # layer2
        with tf.variable_scope('layer2') as scope:
            conv2 = self.conv(BN_r1, [5,5,96,256], [1,1,1,1], 'SAME')
            pool2 = self.pool(conv2, [1,3,3,1], [1,2,2,1], 'VALID')
            BN_r2 = self.BN_relu(pool2, train_phase)

        # layer3
        with tf.variable_scope('layer3') as scope:
            conv3 = self.conv(BN_r2, [3,3,256,384], [1,1,1,1], 'SAME')
            BN_r3 = self.BN_relu(conv3, train_phase)

        # layer4
        with tf.variable_scope('layer4') as scope:
            conv4 = self.conv(BN_r3, [3,3,384,384], [1,1,1,1], 'SAME')
            BN_r4 = self.BN_relu(conv4, train_phase)

        # layer5
        with tf.variable_scope('layer5') as scope:
            conv5 = self.conv(BN_r4, [3,3,384,256], [1,1,1,1], 'SAME')
            pool3 = self.pool(conv5, [1,3,3,1], [1,2,2,1], 'VALID')
            BN_r5 = self.BN_relu(pool3, train_phase)

        # transform data from multi-dimension to one dimension
        fc_input = tf.reshape(BN_r5, shape=[batch_size, -1])
        fc_input_dimension = fc_input.get_shape()[1].value # =9216

        # layer6
        with tf.variable_scope('layer6') as scope:
            # fc1 = self.fc(fc_input, fc_input_dimension, 4096, keep_prop, train_phase)
            fc1 = self.fc(fc_input, 9216, 4096, keep_prop, train_phase)
        # layer7
        with tf.variable_scope('layer7') as scope:
            fc2 = self.fc(fc1, 4096, 4096, keep_prop, train_phase)

        # output layer
        with tf.variable_scope('Out_layer') as scope:
            O_Weight = tf.get_variable(name='OutWeight',
                                      shape=[4096, 28],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
            O_bias = tf.get_variable(name='outputBias',
                                     shape=[28],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer()) #tf.constant_initializer(0.1))
            logits = tf.add(tf.matmul(fc2, O_Weight), O_bias, name='logits') # matrix with size:[batch_size, 28]
            # logits_xy = tf.reshape(logits, [batch_size, 14, 2])
            # logits_xy in the shape of [[x0,y0],
            #                            [x1, y1],
            #                              ...
            #                            [x13,y13]] in each batch cell
        # return logits, BN_r1, BN_r2,BN_r3,BN_r4,BN_r5
        return logits

    # l2 distance of labels and logits, which is (1/nk)sigma_batch(sigma_k_points(||label(x,y)-logit(x,y)||2)),
    # count only marks are not 0
    # labels in a shape of (batch, 14, 2), logits_xy in a shape of (batch, 14, 2), marks in a shape of (batch, 14, 1)
    # all of labels, logits and marks in the format of tf.float32
    def loss(self, logits, labels, marks, batch_size):
        # logits_xy in the shape of [[x0,y0],
        #                            [x1, y1],
        #                              ...
        #                            [x13,y13]] in each batch cell
        with tf.variable_scope('loss') as scope:
            logits_xy = tf.reshape(logits, [batch_size, 14, 2])
            coordinate_diff = (logits_xy - labels) * (marks+1E-7)
            norm_cell = tf.norm(coordinate_diff, axis=2, keepdims=True)
            loss = tf.reduce_mean(norm_cell, name='loss')
            # loss = tf.losses.mean_squared_error(labels*marks, logits_xy*marks)
            # loss = tf.add(loss,0,name='loss')
            tf.summary.scalar(scope.name + '/loss', loss)
            return loss


    def training(self, loss, learning_rate):
        # use adam to do optimization, learning_rate = (1e-3, 5e-4)
        with tf.variable_scope('train') as scope:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                global_step = tf.Variable(0, name='global_step', trainable=False)
                train_op = optimizer.minimize(loss, global_step=global_step,name='train')
            return train_op

    def evaluate(self, logits, labels, marks, batch_size):
        with tf.variable_scope('accuracy') as scope:
            logits_xy = tf.reshape(logits, [batch_size, 14, 2])
            coordinate_diff = (logits_xy - labels) * marks
            norm_cell = tf.norm(coordinate_diff, axis=2, keepdims=True)
            accuracy = 1 - tf.reduce_mean(norm_cell / 1.42)  # 1.42 is the approximation of sqrt(2)
            accuracy = tf.add(accuracy,0,name='accuracy')
            tf.summary.scalar(scope.name + '/accuracy', accuracy)
        return accuracy


    # def conv(self, data, scan_window, stride, padding, pad_type):
    def conv(self, data, scan_window, stride, pad_type):
        conWeight = tf.get_variable(name='conWeight',
                                 shape=scan_window,
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # bias = tf.get_variable(name='conBias',
        #                        shape=scan_window[3],
        #                        dtype=tf.float32,
        #                        initializer=tf.contrib.layers.xavier_initializer())
        # data = tf.pad(data, padding) #padding: [[batch],[column],[row]]
        conv_scan = tf.nn.conv2d(data, conWeight, strides=stride, padding=pad_type)
        # conv_result = tf.nn.bias_add(conv_scan, bias)
        return conv_scan

    def pool(self, data, scan_window, stride, pad_tpye):
        pool_result = tf.nn.max_pool(data, ksize=scan_window, strides=stride, padding=pad_tpye)
        return pool_result

    def BN_relu(self, data, train_phase):
        BN = tf.layers.batch_normalization(data,training=train_phase)
        rl = tf.nn.relu(BN)
        return rl

    def fc(self, data, in_size, out_size, keep_prop, train_pahse):
        # if not train_pahse:
        #     keep_prop = 1
        fcWeight = tf.get_variable(name='fcWeight',
                                 shape=[in_size, out_size],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # fcWeight = tf.get_variable(name='fcWeight',
        #                            shape=[in_size, out_size],
        #                            dtype=tf.float32,
        #                            initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        fc_Mul = tf.matmul(data, fcWeight)
        BN_rl = self.BN_relu(fc_Mul, train_pahse)
        dropL = tf.nn.dropout(BN_rl, keep_prop)
        return dropL