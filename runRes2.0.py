# created by kenany@armoi.com

import modelresnet
import loadInput
import tensorflow as tf
import numpy as np
import os
# import evaluation

train_batch_size = 48
vali_batch_size = 1000
MAX_STEP = 10000


logs_dir = '/home/Kenany/logs'
vali_logs_dir = '/home/Kenany/logs/vali'

files_dir = '/home/Kenany/data/image/'

mark_dir = '/home/Kenany/data/joints_mark.mat'
label_dir = '/home/Kenany/data/ResLabel.mat'
Pweight_dir = '/home/Kenany/data/Pweight.mat'

# load data:
l_d = loadInput.loadInput()
labels, marks, Pweights = l_d.get_labels(mark_dir, label_dir, Pweight_dir)
file_dir = l_d.get_file_dir(files_dir)

# separate data into three category
train_image_dir, vali_image_dir, test_image_dir, \
train_label, vali_label, test_label, \
train_mark, vali_mark, test_mark, \
train_Pweights, vali_Pweights, test_Pweights = l_d.get_train_validation_test_set(file_dir, labels, marks, Pweights)

keep_prop = tf.placeholder(tf.float32, name='prop')
is_training = tf.placeholder(tf.bool, name='is_training')
batch_size = tf.placeholder(tf.int32, name='b_s')
image_batch = tf.placeholder(tf.float32, name='im_b')
label_batch = tf.placeholder(tf.float32, name='l_b')
mark_batch = tf.placeholder(tf.float32, name='m_b')
learning_rate = tf.placeholder(tf.float32, name='l_rate')
Pweights_batch = tf.placeholder(tf.float32, name='pw_b')

def run_model():
    # get training batch
    train_image_batch, train_label_batch, train_mark_batch, train_Pweight_batch= \
        l_d.get_batch(train_image_dir, train_label, train_mark, train_Pweights, train_batch_size)

    # put the data into network
    md = modelresnet.Model()

    logits = md.inference(image_batch, is_training)
    # calculate the loss
    loss = md.loss(logits, label_batch, mark_batch, Pweights_batch, batch_size)

    # calculate the accuracy
    #
    #
    # do optimization
    train_op = md.training(loss, learning_rate=learning_rate)

    # collect all variables
    summary_op = tf.summary.merge_all()

    # Initialization and start running
    with tf.Session() as sess:
        # store varibles:
        train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=3)
        # initialize the variables
        sess.run(tf.global_variables_initializer())

        vali_acc_highest = 0
        learn = 1E-3
        plateaus = 0

        validation_batch = l_d.get_validation_images(vali_image_dir)

        validation_feed_value = {
            is_training: False,
            image_batch: validation_batch
        }

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                train_value = {
                    is_training: True,
                    batch_size: train_batch_size,
                    image_batch: sess.run(train_image_batch),
                    label_batch: sess.run(train_label_batch),
                    mark_batch: sess.run(train_mark_batch),
                    Pweights_batch: sess.run(train_Pweight_batch),
                    learning_rate: learn
                }
                _, tra_loss = sess.run([train_op, loss], feed_dict=train_value)

                if step % 50 == 0:

                    print('Step %d, train loss = %.5f' % (step, tra_loss))

                    summary_str = sess.run(summary_op, feed_dict=train_value)
                    train_writer.add_summary(summary_str, step)


                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    vali_logits = sess.run(logits, feed_dict=validation_feed_value)

                    logits_bpxy = tf.reshape(vali_logits, [vali_batch_size, 14, 3])

                    logits_pbxy = tf.transpose(logits_bpxy, perm=[2, 0, 1])
                    logits_marks = tf.reshape(logits_pbxy[0], [batch_size, 14, 1])
                    logits_xy = tf.transpose(logits_pbxy[-2:], perm=[1, 2, 0])
                    logits_marks_sigmoid = tf.sigmoid(logits_marks)

                    mark_diff_abs = tf.abs(vali_mark - logits_marks_sigmoid)
                    mark_diff = sess.run(vali_mark - logits_marks_sigmoid, feed_dict={batch_size: vali_batch_size})
                    total_mark = 0
                    correct_mark = 0
                    batch_index = 0
                    for batch in sess.run(mark_diff_abs, feed_dict={batch_size: vali_batch_size}):
                        value_index = 0
                        for value in batch:
                            total_mark = total_mark +1
                            if value < 0.1:
                                xy = logits_xy[batch_index][value_index]
                                dx = sess.run(tf.abs(xy[0]-vali_label[batch_index][value_index][0]))
                                dy = sess.run(tf.abs(xy[1]-vali_label[batch_index][value_index][1]))
                                if mark_diff[batch_index][value_index] < 0:
                                    correct_mark = correct_mark + 1
                                elif dx<0.1 and dy<0.1:
                                    correct_mark = correct_mark + 1
                            value_index = value_index +1
                        batch_index = batch_index + 1
                    vali_accuracy = correct_mark/total_mark
                    print('Accuracy is %.5f' % (vali_accuracy))

                    if vali_accuracy >= vali_acc_highest or step == 0:
                        plateaus = 0
                        print('updata logs')
                        vali_acc_highest = vali_accuracy
                        checkpoint_path = os.path.join(vali_logs_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                    else:
                        plateaus = plateaus + 1
                        if plateaus == 2:
                            plateaus = 0
                            learn = learn / 5
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        train_writer.close()
        coord.join(threads)
        sess.close()



run_model()

