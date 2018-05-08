#created by kenany@armoi.com

import tensorflow as tf

class Model:
    # ResNet-50 from He.'s Paper Deep Residual Learning for Image Recognition
    def inference(self, data, is_training):
        train_phase = tf.cond(is_training, lambda: True, lambda: False)
        # input layer - conv1
        with tf.variable_scope('conv1'):
            conv1 = self.conv(data, [7, 7, 3, 64], [1, 2, 2, 1], 'SAME')
            BN1 = self.BN(conv1, train_phase)
            relu1 = tf.nn.relu(BN1)
            pool1 = self.maxpool(relu1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x : 56*56
        # conv2_1
        with tf.variable_scope('conv2_1'):
            with tf.variable_scope('shortcut1'):
                conv_shortcut1 = self.conv(pool1, [1, 1, 64, 256], [1, 1, 1, 1], 'SAME')
                BN_shortcut1 = self.BN(conv_shortcut1, train_phase)
            with tf.variable_scope('layer1'):
                conv2 = self.conv(pool1, [1, 1, 64, 64], [1, 1, 1, 1], 'SAME')
                BN2= self.BN(conv2, train_phase)
                relu2 = tf.nn.relu(BN2)
            with tf.variable_scope('layer2'):
                conv3 = self.conv(relu2, [3, 3, 64, 64], [1, 1, 1, 1], 'SAME')
                BN3 = self.BN(conv3, train_phase)
                relu3 = tf.nn.relu(BN3)
            with tf.variable_scope('layer3'):
                conv4 = self.conv(relu3, [1, 1, 64, 256], [1, 1, 1, 1], 'SAME')
                BN4 = self.BN(conv4, train_phase)

                res1 = tf.add(BN4, BN_shortcut1)
                BN_relu1 = self.BN(res1, train_phase)
                relu4 = tf.nn.relu(BN_relu1)
        # conv2_2
        with tf.variable_scope('conv2_2'):
            with tf.variable_scope('layer1'):
                conv5 = self.conv(relu4, [1, 1, 256, 64], [1, 1, 1, 1], 'SAME')
                BN5= self.BN(conv5, train_phase)
                relu5 = tf.nn.relu(BN5)
            with tf.variable_scope('layer2'):
                conv6 = self.conv(relu5, [3, 3, 64, 64], [1, 1, 1, 1], 'SAME')
                BN6 = self.BN(conv6, train_phase)
                relu6 = tf.nn.relu(BN6)
            with tf.variable_scope('layer3'):
                conv7 = self.conv(relu6, [1, 1, 64, 256], [1, 1, 1, 1], 'SAME')
                BN7 = self.BN(conv7, train_phase)
                res2 = tf.add(BN7, relu4)
                BN_relu2 = self.BN(res2, train_phase)
                relu7 = tf.nn.relu(BN_relu2)
        # conv2_3
        with tf.variable_scope('conv2_3'):
            with tf.variable_scope('layer1'):
                conv8 = self.conv(relu7, [1, 1, 256, 64], [1, 1, 1, 1], 'SAME')
                BN8= self.BN(conv8, train_phase)
                relu8 = tf.nn.relu(BN8)
            with tf.variable_scope('layer2'):
                conv9 = self.conv(relu8, [3, 3, 64, 64], [1, 1, 1, 1], 'SAME')
                BN9 = self.BN(conv9, train_phase)
                relu9 = tf.nn.relu(BN9)
            with tf.variable_scope('layer3'):
                conv10 = self.conv(relu9, [1, 1, 64, 256], [1, 1, 1, 1], 'SAME')
                BN10 = self.BN(conv10, train_phase)
                res3 = tf.add(BN10, relu7)
                BN_relu3 = self.BN(res3, train_phase)
                relu10 = tf.nn.relu(BN_relu3)

        # conv3_x : 28*28
        # conv3_1
        with tf.variable_scope('conv3_1'):
            with tf.variable_scope('shortcut2'):
                conv_shortcut2 = self.conv(relu10, [1, 1, 256, 512], [1, 2, 2, 1], 'SAME')
                BN_shortcut2 = self.BN(conv_shortcut2, train_phase)
            with tf.variable_scope('layer1'):
                conv11 = self.conv(relu10, [1, 1, 256, 128], [1, 2, 2, 1], 'SAME')
                BN11= self.BN(conv11, train_phase)
                relu11 = tf.nn.relu(BN11)
            with tf.variable_scope('layer2'):
                conv12 = self.conv(relu11, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME')
                BN12 = self.BN(conv12, train_phase)
                relu12 = tf.nn.relu(BN12)
            with tf.variable_scope('layer3'):
                conv13 = self.conv(relu12, [1, 1, 128, 512], [1, 1, 1, 1], 'SAME')
                BN13 = self.BN(conv13, train_phase)

                res4 = tf.add(BN13, BN_shortcut2)
                BN_relu4 = self.BN(res4, train_phase)
                relu13 = tf.nn.relu(BN_relu4)
        # conv3_2
        with tf.variable_scope('conv3_2'):
            with tf.variable_scope('layer1'):
                conv14 = self.conv(relu13, [1, 1, 512, 128], [1, 1, 1, 1], 'SAME')
                BN14= self.BN(conv14, train_phase)
                relu14 = tf.nn.relu(BN14)
            with tf.variable_scope('layer2'):
                conv15 = self.conv(relu14, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME')
                BN15 = self.BN(conv15, train_phase)
                relu15 = tf.nn.relu(BN15)
            with tf.variable_scope('layer3'):
                conv16 = self.conv(relu15, [1, 1, 128, 512], [1, 1, 1, 1], 'SAME')
                BN16 = self.BN(conv16, train_phase)
                res5 = tf.add(BN16, relu13)
                BN_relu5 = self.BN(res5, train_phase)
                relu16 = tf.nn.relu(BN_relu5)
        # conv3_3
        with tf.variable_scope('conv3_3'):
            with tf.variable_scope('layer1'):
                conv17 = self.conv(relu16, [1, 1, 512, 128], [1, 1, 1, 1], 'SAME')
                BN17= self.BN(conv17, train_phase)
                relu17 = tf.nn.relu(BN17)
            with tf.variable_scope('layer2'):
                conv18 = self.conv(relu17, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME')
                BN18 = self.BN(conv18, train_phase)
                relu18 = tf.nn.relu(BN18)
            with tf.variable_scope('layer3'):
                conv19 = self.conv(relu18, [1, 1, 128, 512], [1, 1, 1, 1], 'SAME')
                BN19 = self.BN(conv19, train_phase)
                res6 = tf.add(BN19, relu16)
                BN_relu6 = self.BN(res6, train_phase)
                relu19 = tf.nn.relu(BN_relu6)
        # conv3_4
        with tf.variable_scope('conv3_4'):
            with tf.variable_scope('layer1'):
                conv20 = self.conv(relu19, [1, 1, 512, 128], [1, 1, 1, 1], 'SAME')
                BN20= self.BN(conv20, train_phase)
                relu20 = tf.nn.relu(BN20)
            with tf.variable_scope('layer2'):
                conv21 = self.conv(relu20, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME')
                BN21 = self.BN(conv21, train_phase)
                relu21 = tf.nn.relu(BN21)
            with tf.variable_scope('layer3'):
                conv22 = self.conv(relu21, [1, 1, 128, 512], [1, 1, 1, 1], 'SAME')
                BN22 = self.BN(conv22, train_phase)
                res7 = tf.add(BN22, relu19)
                BN_relu7 = self.BN(res7, train_phase)
                relu22 = tf.nn.relu(BN_relu7)

        # conv4_x : 14*14
        # conv4_1
        with tf.variable_scope('conv4_1'):
            with tf.variable_scope('shortcut3'):
                conv_shortcut3 = self.conv(relu22, [1, 1, 512, 1024], [1, 2, 2, 1], 'SAME')
                BN_shortcut3 = self.BN(conv_shortcut3, train_phase)
            with tf.variable_scope('layer1'):
                conv23 = self.conv(relu22, [1, 1, 512, 256], [1, 2, 2, 1], 'SAME')
                BN23= self.BN(conv23, train_phase)
                relu23 = tf.nn.relu(BN23)
            with tf.variable_scope('layer2'):
                conv24 = self.conv(relu23, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME')
                BN24 = self.BN(conv24, train_phase)
                relu24 = tf.nn.relu(BN24)
            with tf.variable_scope('layer3'):
                conv25 = self.conv(relu24, [1, 1, 256, 1024], [1, 1, 1, 1], 'SAME')
                BN25 = self.BN(conv25, train_phase)

                res8 = tf.add(BN25, BN_shortcut3)
                BN_relu8 = self.BN(res8, train_phase)
                relu25 = tf.nn.relu(BN_relu8)
        # conv4_2
        with tf.variable_scope('conv4_2'):
            with tf.variable_scope('layer1'):
                conv26 = self.conv(relu25, [1, 1, 1024, 256], [1, 1, 1, 1], 'SAME')
                BN26= self.BN(conv26, train_phase)
                relu26 = tf.nn.relu(BN26)
            with tf.variable_scope('layer2'):
                conv27 = self.conv(relu26, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME')
                BN27 = self.BN(conv27, train_phase)
                relu27 = tf.nn.relu(BN27)
            with tf.variable_scope('layer3'):
                conv28 = self.conv(relu27, [1, 1, 256, 1024], [1, 1, 1, 1], 'SAME')
                BN28 = self.BN(conv28, train_phase)
                res9 = tf.add(BN28, relu25)
                BN_relu9 = self.BN(res9, train_phase)
                relu28 = tf.nn.relu(BN_relu9)
        # conv4_3
        with tf.variable_scope('conv4_3'):
            with tf.variable_scope('layer1'):
                conv29 = self.conv(relu28, [1, 1, 1024, 256], [1, 1, 1, 1], 'SAME')
                BN29= self.BN(conv29, train_phase)
                relu29 = tf.nn.relu(BN29)
            with tf.variable_scope('layer2'):
                conv30 = self.conv(relu29, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME')
                BN30 = self.BN(conv30, train_phase)
                relu30 = tf.nn.relu(BN30)
            with tf.variable_scope('layer3'):
                conv31 = self.conv(relu30, [1, 1, 256, 1024], [1, 1, 1, 1], 'SAME')
                BN31 = self.BN(conv31, train_phase)
                res10 = tf.add(BN31, relu28)
                BN_relu10 = self.BN(res10, train_phase)
                relu31 = tf.nn.relu(BN_relu10)
        # conv4_4
        with tf.variable_scope('conv4_4'):
            with tf.variable_scope('layer1'):
                conv32 = self.conv(relu31, [1, 1, 1024, 256], [1, 1, 1, 1], 'SAME')
                BN32= self.BN(conv32, train_phase)
                relu32 = tf.nn.relu(BN32)
            with tf.variable_scope('layer2'):
                conv33 = self.conv(relu32, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME')
                BN33 = self.BN(conv33, train_phase)
                relu33 = tf.nn.relu(BN33)
            with tf.variable_scope('layer3'):
                conv34 = self.conv(relu33, [1, 1, 256, 1024], [1, 1, 1, 1], 'SAME')
                BN34 = self.BN(conv34, train_phase)
                res11 = tf.add(BN34, relu31)
                BN_relu11 = self.BN(res11, train_phase)
                relu34 = tf.nn.relu(BN_relu11)
        # conv4_5
        with tf.variable_scope('conv4_5'):
            with tf.variable_scope('layer1'):
                conv35 = self.conv(relu34, [1, 1, 1024, 256], [1, 1, 1, 1], 'SAME')
                BN35= self.BN(conv35, train_phase)
                relu35 = tf.nn.relu(BN35)
            with tf.variable_scope('layer2'):
                conv36 = self.conv(relu35, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME')
                BN36 = self.BN(conv36, train_phase)
                relu36 = tf.nn.relu(BN36)
            with tf.variable_scope('layer3'):
                conv37 = self.conv(relu36, [1, 1, 256, 1024], [1, 1, 1, 1], 'SAME')
                BN37 = self.BN(conv37, train_phase)
                res12 = tf.add(BN37, relu34)
                BN_relu12 = self.BN(res12, train_phase)
                relu37 = tf.nn.relu(BN_relu12)
        # conv4_6
        with tf.variable_scope('conv4_6'):
            with tf.variable_scope('layer1'):
                conv38 = self.conv(relu37, [1, 1, 1024, 256], [1, 1, 1, 1], 'SAME')
                BN38= self.BN(conv38, train_phase)
                relu38 = tf.nn.relu(BN38)
            with tf.variable_scope('layer2'):
                conv39 = self.conv(relu38, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME')
                BN39 = self.BN(conv39, train_phase)
                relu39 = tf.nn.relu(BN39)
            with tf.variable_scope('layer3'):
                conv40 = self.conv(relu39, [1, 1, 256, 1024], [1, 1, 1, 1], 'SAME')
                BN40 = self.BN(conv40, train_phase)
                res13 = tf.add(BN40, relu37)
                BN_relu13 = self.BN(res13, train_phase)
                relu40 = tf.nn.relu(BN_relu13)

        # conv5_x : 7*7
        # conv5_1
        with tf.variable_scope('conv5_1'):
            with tf.variable_scope('shortcut4'):
                conv_shortcut4 = self.conv(relu40, [1, 1, 1024, 2048], [1, 2, 2, 1], 'SAME')
                BN_shortcut4 = self.BN(conv_shortcut4, train_phase)
            with tf.variable_scope('layer1'):
                conv41 = self.conv(relu40, [1, 1, 1024, 512], [1, 2, 2, 1], 'SAME')
                BN41= self.BN(conv41, train_phase)
                relu41 = tf.nn.relu(BN41)
            with tf.variable_scope('layer2'):
                conv42 = self.conv(relu41, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME')
                BN42 = self.BN(conv42, train_phase)
                relu42 = tf.nn.relu(BN42)
            with tf.variable_scope('layer3'):
                conv43 = self.conv(relu42, [1, 1, 512, 2048], [1, 1, 1, 1], 'SAME')
                BN43 = self.BN(conv43, train_phase)

                res14 = tf.add(BN43, BN_shortcut4)
                BN_relu14 = self.BN(res14, train_phase)
                relu43 = tf.nn.relu(BN_relu14)
        # conv5_2
        with tf.variable_scope('conv5_2'):
            with tf.variable_scope('layer1'):
                conv44 = self.conv(relu43, [1, 1, 2048, 512], [1, 1, 1, 1], 'SAME')
                BN44= self.BN(conv44, train_phase)
                relu44 = tf.nn.relu(BN44)
            with tf.variable_scope('layer2'):
                conv45 = self.conv(relu44, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME')
                BN45 = self.BN(conv45, train_phase)
                relu45 = tf.nn.relu(BN45)
            with tf.variable_scope('layer3'):
                conv46 = self.conv(relu45, [1, 1, 512, 2048], [1, 1, 1, 1], 'SAME')
                BN46 = self.BN(conv46, train_phase)
                res15 = tf.add(BN46, relu43)
                BN_relu15 = self.BN(res15, train_phase)
                relu46 = tf.nn.relu(BN_relu15)
        # conv5_3
        with tf.variable_scope('conv5_3'):
            with tf.variable_scope('layer1'):
                conv47 = self.conv(relu46, [1, 1, 2048, 512], [1, 1, 1, 1], 'SAME')
                BN47= self.BN(conv47, train_phase)
                relu47 = tf.nn.relu(BN47)
            with tf.variable_scope('layer2'):
                conv48 = self.conv(relu47, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME')
                BN48 = self.BN(conv48, train_phase)
                relu48 = tf.nn.relu(BN48)
            with tf.variable_scope('layer3'):
                conv49 = self.conv(relu48, [1, 1, 512, 2048], [1, 1, 1, 1], 'SAME')
                BN49 = self.BN(conv49, train_phase)
                res16 = tf.add(BN49, relu46)
                BN_relu16 = self.BN(res16, train_phase)
                relu49 = tf.nn.relu(BN_relu16)
        # final layer
        with tf.variable_scope('output_layer'):
            global_avg_pool = tf.reduce_mean(relu49, [1, 2])
            O_Weight = tf.get_variable(name='OutWeight',
                                       shape=[2048, 28],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
            O_bias = tf.get_variable(name='outputBias',
                                     shape=[28],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())  # tf.constant_initializer(0.1))
            logits = tf.add(tf.matmul(global_avg_pool, O_Weight), O_bias, name='logits')

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
            coordinate_diff = (logits_xy - labels) * (marks + 1E-10)
            norm_cell = tf.norm(coordinate_diff, axis=2, keepdims=True)
            loss = tf.reduce_mean(norm_cell, name='loss')
            tf.summary.scalar(scope.name + '/loss', loss)
            return loss

    def training(self, loss, learning_rate):
        # use adam to do optimization, learning_rate = (1e-3, 5e-4)
        with tf.variable_scope('train') as scope:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                global_step = tf.Variable(0, name='global_step', trainable=False)
                train_op = optimizer.minimize(loss, global_step=global_step, name='train')
            return train_op

    def evaluate(self, logits, labels, marks, batch_size):
        with tf.variable_scope('accuracy') as scope:
            logits_xy = tf.reshape(logits, [batch_size, 14, 2])
            coordinate_diff = (logits_xy - labels) * marks
            norm_cell = tf.norm(coordinate_diff, axis=2, keepdims=True)
            accuracy = 1 - tf.reduce_mean(norm_cell / 1.42)  # 1.42 is the approximation of sqrt(2)
            accuracy = tf.add(accuracy, 0, name='accuracy')
            tf.summary.scalar(scope.name + '/accuracy', accuracy)
        return accuracy

    # def conv(self, data, scan_window, stride, padding, pad_type):
    def conv(self, data, scan_window, stride, pad_type):
        conWeight = tf.get_variable(name='conWeight',
                                 shape=scan_window,
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv_scan = tf.nn.conv2d(data, conWeight, strides=stride, padding=pad_type)
        return conv_scan

    def BN(self, data, train_phase):
        return tf.layers.batch_normalization(data,training=train_phase)

    def maxpool(self, data, scan_window, stride, pad_tpye):
        pool_result = tf.nn.max_pool(data, ksize=scan_window, strides=stride, padding=pad_tpye)
        return pool_result
