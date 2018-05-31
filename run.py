# created by kenany@armoi.com

import model
import loadInput
import tensorflow as tf
import numpy as np
import os

train_batch_size = 128
vali_batch_size = 1000
# learning_rate = 1E-4
# Round_1_STEP = 1
MAX_STEP = 10000

vali_acc_highest = 0

logs_dir = '/Users/kenanyang/Desktop/Armoi/TF/logs'
vali_logs_dir = '/Users/kenanyang/Desktop/Armoi/TF/logs/vali'

files_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/new_images/'

mark_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/joints_mark.mat'
label_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/joints_label_positive.mat'

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


def run_model():
    # get training batch
    train_image_batch, train_label_batch, train_mark_batch = \
        l_d.get_batch(train_image_dir, train_label, train_mark, train_batch_size)

    # put the data into network
    md = model.Model()
    # logits = md.inference(train_image_batch, batch_size, keep_prop=0.5, train_phase=True)
    # logits_xy,b1,b2,b3,b4,b5 = md.inference(train_image_batch, batch_size, keep_prop=0.5, train_phase=True)
    logits = md.inference(image_batch, batch_size, keep_prop, is_training)
    # calculate the loss
    loss = md.loss(logits, label_batch, mark_batch, batch_size)

    # calculate the accuracy
    acc = md.evaluate(logits, label_batch, mark_batch, batch_size)

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
        learn = 1E-3
        plateaus = 0

        vali_batch = l_d.get_validation_images(vali_image_dir)
        vali_value = {
            keep_prop: 1,
            is_training: False,
            batch_size: vali_batch_size,
            image_batch: vali_batch,
            label_batch: sess.run(vali_label),
            mark_batch: sess.run(vali_mark)
        }

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                # _, tra_loss,bb1,bb2,bb3,bb4,bb5,_xy = sess.run([train_op, train_loss,b1,b2,b3,b4,b5,logits_xy])
                # _, tra_loss = sess.run([train_op, train_loss])
                ##########################
                #####start training#######
                train_value = {
                    keep_prop: 0.8,
                    is_training: True,
                    batch_size: train_batch_size,
                    image_batch: sess.run(train_image_batch),
                    label_batch: sess.run(train_label_batch),
                    mark_batch: sess.run(train_mark_batch),
                    learning_rate: learn
                }
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc], feed_dict=train_value)
                ##########################
                ##########################
                if step % 50 == 0:
                    # tra_loss, tra_acc = sess.run([train_loss, train_acc])

                    # vali_value = {
                    #     keep_prop: 1,
                    #     is_training: False,
                    #     batch_size: vali_batch_size,
                    #     image_batch: sess.run(validation_image_batch),
                    #     label_batch: sess.run(validation_label_batch),
                    #     mark_batch: sess.run(validation_mark_batch)
                    # }
                    # vali_acc = sess.run([acc], feed_dict=vali_value)
                    print('Step %d, train loss = %.5f, train accuracy = %.5f' % (step, tra_loss, tra_acc))
                    # print(_xy)
                    # print(sess.run(b1))
                    # print(sess.run(b2))
                    # print(sess.run(b3))
                    # print(sess.run(b4))
                    # print(sess.run(b5))
                    summary_str = sess.run(summary_op, feed_dict=train_value)
                    train_writer.add_summary(summary_str, step)
                    # checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                    # # saver.save(sess, checkpoint_path, global_step=step)
                    # saver.save(sess, checkpoint_path)

                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    vali_acc = sess.run(acc, feed_dict=vali_value)
                    print('Step %d, validation accuracy = %.5f' % (step, vali_acc))
                    # checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                    # saver.save(sess, checkpoint_path, global_step=step)
                    if vali_acc > vali_acc_highest:
                        plateaus = 0
                        print('updata logs')
                        vali_acc_highest = vali_acc
                        checkpoint_path = os.path.join(vali_logs_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                    else:
                        plateaus = plateaus + 1
                        if plateaus == 2:
                            plateaus = 0
                            learn = learn / 10

                    # acc = accuracy_evaluation(vali_image_dir, vali_label, vali_mark, batch_size)
                    # print('validation accuracy: %.5f' % (acc))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


# def keep_training():
#     sess = tf.Session()
#     graph_path = os.path.join(logs_dir, 'model.ckpt.meta')
#     saver = tf.train.import_meta_graph(graph_path)
#     ckpt = tf.train.get_checkpoint_state(logs_dir)
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#     else:
#         print('No checkpoint file found')
#     # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
#     graph = tf.get_default_graph()
#     # logits = graph.get_operation_by_name('logits')
#     # calculate the loss
#     loss = graph.get_operation_by_name('loss/loss')
#
#     # calculate the accuracy
#     acc = graph.get_operation_by_name('accuracy/accuracy')
#
#     # do optimization
#     train_op = graph.get_operation_by_name('train/train')
#     del saver
#     sess.close()
#
#     # collect all variables
#     summary_op = tf.summary.merge_all()
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#
#
#
#         # keep training
#         # get training batch
#         train_image_batch, train_label_batch, train_mark_batch = \
#             l_d.get_batch(train_image_dir, train_label, train_mark, train_batch_size)
#
#
#         # initialize the queue threads to start to shovel data
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#         train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
#         try:
#             for step in np.arange(MAX_STEP):
#                 if coord.should_stop():
#                     break
#                 train_value = {
#                     keep_prop: 0.5,
#                     is_training: True,
#                     batch_size: train_batch_size,
#                     image_batch: sess.run(train_image_batch),
#                     label_batch: sess.run(train_label_batch),
#                     mark_batch: sess.run(train_mark_batch)
#                 }
#                 _, tra_loss, tra_acc = sess.run([train_op, loss, acc], feed_dict=train_value)
#
#                 if step % 1 == 0:
#                     print('Step %d, train loss = %.5f, train accuracy = %.5f' % (step, tra_loss, tra_acc))
#
#                     summary_str = sess.run(summary_op, feed_dict=train_value)
#                     train_writer.add_summary(summary_str, step)
#
#                 if step % 5000 == 0 or (step + 1) == MAX_STEP:
#                     checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
#                     saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
#         except tf.errors.OutOfRangeError:
#             print('Done training -- epoch limit reached')
#         finally:
#             coord.request_stop()
#         coord.join(threads)
#         sess.close()


run_model()
# keep_training()
