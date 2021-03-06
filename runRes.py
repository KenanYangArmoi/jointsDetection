# created by kenany@armoi.com

import modelresnet
import loadInput
import tensorflow as tf
import numpy as np
import os

train_batch_size = 2
vali_batch_size = 1
MAX_STEP = 10


logs_dir = '/Users/kenanyang/Desktop/Armoi/TF/logs'
vali_logs_dir = '/Users/kenanyang/Desktop/Armoi/TF/logs/vali'

files_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/224/image/'

mark_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/224/joints_mark.mat'
label_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/224/ResLabel.mat'

# load data:
l_d = loadInput.loadInput()
labels, marks = l_d.get_labels(mark_dir, label_dir)
file_dir = l_d.get_file_dir(files_dir)

# separate data into three category
train_image_dir, vali_image_dir, test_image_dir, \
train_label, vali_label, test_label, train_mark, \
vali_mark, test_mark = l_d.get_train_validation_test_set(file_dir, labels, marks)

keep_prop = tf.placeholder(tf.float32, name='prop')
is_training = tf.placeholder(tf.bool, name='is_training')
batch_size = tf.placeholder(tf.int32, name='b_s')
image_batch = tf.placeholder(tf.float32, name='im_b')
label_batch = tf.placeholder(tf.float32, name='l_b')
mark_batch = tf.placeholder(tf.float32, name='m_b')
learning_rate = tf.placeholder(tf.float32, name='l_rate')
# evaluation_logits_tra = tf.placeholder(tf.float32, shape=[train_batch_size, 42],name='e_logits_tra')
# evaluation_logits_vali = tf.placeholder(tf.float32, shape=[vali_batch_size, 42],name='e_logits_vali')


def run_model():
    # get training batch
    train_image_batch, train_label_batch, train_mark_batch = \
        l_d.get_batch(train_image_dir, train_label, train_mark, train_batch_size)

    # put the data into network
    md = modelresnet.Model()
    # logits = md.inference(train_image_batch, batch_size, keep_prop=0.5, train_phase=True)
    # logits_xy,b1,b2,b3,b4,b5 = md.inference(train_image_batch, batch_size, keep_prop=0.5, train_phase=True)
    logits = md.inference(image_batch, is_training)
    # calculate the loss
    loss = md.loss(logits, label_batch, mark_batch, batch_size)

    # calculate the accuracy
    # acc_mark_tra, acc_dist_tra = md.evaluate(evaluation_logits_tra, label_batch, mark_batch, batch_size)
    # acc_mark_vali, acc_dist_vali = md.evaluate(evaluation_logits_vali, label_batch, mark_batch, batch_size)
    # do optimization
    train_op = md.training(loss, learning_rate=learning_rate)

    # collect all variables
    summary_op = tf.summary.merge_all()

    # get validation batch
    # l_d_v = loadInput.loadInput()
    # validation_image_batch, validation_label_batch, validation_mark_batch = \
    #     l_d_v.get_batch(vali_image_dir, vali_label, vali_mark, vali_batch_size)

    # Initialization and start running
    with tf.Session() as sess:
        # store varibles:
        train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=3)
        # initialize the variables
        sess.run(tf.global_variables_initializer())

        vali_acc_highest = 0
        vali_acc_mark = 0
        learn = 1E-3
        plateaus = 0

        vali_batch = l_d.get_validation_images(vali_image_dir)
        vali_value = {
            is_training: False,
            batch_size: vali_batch_size,
            image_batch: vali_batch,
            # label_batch: sess.run(vali_label),
            # mark_batch: sess.run(vali_mark)
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
                    learning_rate: learn
                }
                _, tra_loss = sess.run([train_op, loss], feed_dict=train_value)

                # tra_acc_value = {
                #     # evaluation_logits_tra: tra_logits,
                #     label_batch: sess.run(train_label_batch),
                #     mark_batch: sess.run(train_mark_batch),
                #     batch_size: train_batch_size
                # }
                # tra_acc_mark, tra_acc_dist = sess.run([acc_mark_tra, acc_dist_tra],feed_dict=tra_acc_value)
                if step % 1 == 0:

                    print('Step %d, train loss = %.5f' % (step, tra_loss))

                    summary_str = sess.run(summary_op, feed_dict=train_value)
                    train_writer.add_summary(summary_str, step)


                if step % 10 == 0 or (step + 1) == MAX_STEP:
                    # acc_mark, acc_dist = evaluate(sess, logits, label_batch, mark_batch, batch_size)
                    # vali_logits = sess.run([logits], feed_dict=vali_value)
                    # vali_acc_value = {
                    #     # evaluation_logits_vali: vali_logits,
                    #     label_batch: sess.run(vali_label),
                    #     mark_batch: sess.run(vali_mark),
                    #     batch_size: vali_batch_size
                    # }
                    # vali_acc_mark, vali_acc_dist = sess.run([acc_mark_vali, acc_dist_vali], feed_dict=vali_acc_value)
                    # print('Step %d, validation accuracy mark = %.5f, validation accuracy distance = %.5f' % (step))
                    # checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                    # saver.save(sess, checkpoint_path, global_step=step)
                    if vali_acc_mark > vali_acc_highest or step == 0:
                        plateaus = 0
                        print('updata logs')
                        vali_acc_highest = vali_acc_mark
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
        coord.join(threads)
        sess.close()



run_model()

