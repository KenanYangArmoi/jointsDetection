# created by kenan@armoi.com

import scipy.io as sio
# import numpy as np
import tensorflow as tf
import os
import random

class loadInput():

    def get_labels(self, mark_dir, label_dir):
        mark_mat = sio.loadmat(mark_dir)
        label_mat = sio.loadmat(label_dir)
        marks_mat = mark_mat['joints_mark']
        labels_mat = label_mat['ResLabel']

        # marks = np.array(marks_mat)  # marks in a shape of (14 ,1, 10000)
        marks_tf = tf.cast(marks_mat, tf.float32)
        marks_t = tf.transpose(marks_tf,
                               perm=[2, 0, 1])  # marks_t in a shape of [10000, 14 ,1], with format tf.float32
        # labels = np.array(labels_mat)  # labels in a shape of (14 ,2, 10000)
        labels_tf = tf.cast(labels_mat, tf.float32)
        labels_t = tf.transpose(labels_tf,
                                perm=[2, 0, 1])  # labels_t in a shape of [10000, 14, 2], with format tf.float32
        return labels_t, marks_t

    def get_file_dir(self, file_dir):
        # file_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/images/'
        image_dir = []
        for file in os.listdir(file_dir):
            image_dir.append(file_dir+file)
        image_dir.sort()
        image_dir = tf.cast(image_dir, tf.string) # In tensorflow format
        return image_dir

    def get_train_validation_test_set(self, image_dir, labels, marks): # 10% to validate and 10% to test
        validation_size = 400
        test_size = 1000
        partitions = [0] * 10000
        partitions[:validation_size] = [1] * validation_size
        partitions[validation_size:validation_size+test_size] = [2] * test_size
        random.shuffle(partitions)

        train_image_dir, vali_image_dir, test_image_dir = tf.dynamic_partition(image_dir, partitions, 3)
        train_label, vali_label, test_label = tf.dynamic_partition(labels, partitions, 3)
        train_mark, vali_mark, test_mark = tf.dynamic_partition(marks, partitions, 3)

        return train_image_dir, vali_image_dir, test_image_dir,\
               train_label, vali_label, test_label,\
               train_mark, vali_mark, test_mark

    def get_batch(self, image_dir, labels, marks, batch_size):
        target_H = 224
        target_W = 224
        num_channels = 3
        # new_central = tf.constant(110, dtype=tf.float32, shape=[14,2]) # new_central = [target_H/2, target_W/2]

        # create the queues
        input_queue = tf.train.slice_input_producer([image_dir, labels, marks])
        # handle each slice
        image_content = tf.read_file(input_queue[0])
        label = input_queue[1]
        mark = input_queue[2]
        image = tf.image.decode_jpeg(image_content, channels=3)

        # # Modify the size of the image
        # with tf.Session() as sess_batch:
        #     sess_batch.run(tf.initialize_all_variables())
        #     coordx = tf.train.Coordinator()
        #     threadsx = tf.train.start_queue_runners(coord=coordx)
        #     im_modified, height, new_h = self.modify_image(sess_batch, image, target_H, target_W)
        #     # calculate new label (centralized and normalized)
        #     label_resize = label * (new_h / height)
        #     label_right_coordinate = (label_resize - new_central)/target_H
        #     # get batch
        #     image_batch, label_batch, mark_batch = tf.train.batch([im_modified, label_right_coordinate, mark],
        #                                                           batch_size=batch_size)
        #     coordx.request_stop()
        #     coordx.join(threadsx)
        #     sess_batch.close()
        # Normalize Image
        image.set_shape([target_H, target_W, num_channels])
        im_standardized = tf.image.per_image_standardization(image)
        image_batch, label_batch, mark_batch = tf.train.batch([im_standardized, label, mark], batch_size=batch_size)


        return image_batch, label_batch, mark_batch

    # def modify_image(self, sess_batch, image, target_H, target_W):
    #
    #     height = sess_batch.run(tf.shape(image[:, 0, 0]))
    #     width = sess_batch.run(tf.shape(image[0, :, 0]))
    #     if height > width:
    #         new_h = target_H
    #         new_w = new_h * width / height
    #     else:
    #         new_w = target_W
    #         new_h = new_w * height / width
    #
    #     im_resize = tf.image.resize_images(image, [int(new_h), int(new_w)], method=1)
    #     # im_resize = tf.image.resize_images(image, [int(new_h), int(new_w)])
    #     im_right_shape = tf.image.resize_image_with_crop_or_pad(im_resize, target_H, target_W)
    #     im_standardized = tf.image.per_image_standardization(im_right_shape)
    #
    #     return im_standardized, height, new_h
    def get_validation_images(self, image_dir):
        with tf.Session() as sess:
            target_H, target_W, num_channels = 224, 224, 3
            x = sess.run(tf.shape(image_dir))[0]
            image_batch = []
            for i in range(x):
                image_file = tf.read_file(image_dir[i])
                image = tf.image.decode_jpeg(image_file, channels=3)
                image.set_shape([target_H, target_W, num_channels])
                im_standardized = tf.image.per_image_standardization(image)
                image_batch.append(sess.run(im_standardized))

            sess.close()
        return image_batch

