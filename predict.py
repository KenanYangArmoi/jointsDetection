import tensorflow as tf
import os
import numpy as np

file_dir = '/Users/kenanyang/Desktop/Armoi/lspet_dataset/new_images/im00552.jpg'
logs_dir = '/Users/kenanyang/Desktop/Armoi/TF/logs0423_coordinate_loss'

def prediction(logs_dir, file_dir):
    with tf.Session() as sess:
        model_dir = os.path.join(logs_dir, 'model.ckpt-499.meta')
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        saver = tf.train.import_meta_graph(model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        graph = tf.get_default_graph()
        logits = graph.get_tensor_by_name("Out_layer/logits:0")
        keep_prop = graph.get_tensor_by_name("prop:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        batch_size = graph.get_tensor_by_name("b_s:0")
        image_batch = graph.get_tensor_by_name("im_b:0")

        # keep_prop = tf.placeholder(tf.float32, name='prop')
        # is_training = tf.placeholder(tf.bool, name='is_training')
        # batch_size = tf.placeholder(tf.int32, name='b_s')
        # image_batch = tf.placeholder(tf.float32, name='im_b')

        target_H, target_W, num_channels = 220, 220, 3
        image_file = tf.read_file(file_dir)
        image = tf.image.decode_jpeg(image_file, channels=3)
        image.set_shape([target_H, target_W, num_channels])
        im_standardized = tf.image.per_image_standardization(image)
        im_batch = tf.reshape(im_standardized, [1, target_H, target_W, num_channels])
        image = sess.run(im_batch)
        feed_dict = {
            keep_prop: 1.0,
            is_training: False,
            batch_size: 1,
            image_batch: image
        }

        y = sess.run([logits], feed_dict)
        _xy = np.reshape(y, [14, 2])
        xy = np.add(np.multiply(_xy, 220),110)

        return xy

y = prediction(logs_dir, file_dir)
print(y)