# coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

import model as model
import numpy as np
from reader import Cifar10Reader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 30, "訓練するEpoch数")
tf.app.flags.DEFINE_string('data_dir', './cifar-10-batches-bin 2/', "訓練データのディレクトリ")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/',
                           "チェックポイントを保存するディレクトリ")
tf.app.flags.DEFINE_string('test_data', None, "テストデータのパス")

def _loss(logits, label):
  labels = tf.cast(label, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels,name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  return cross_entropy_mean


def _train(total_loss, global_step):
  opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  grads = opt.compute_gradients(total_loss)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return train_op

filenames = [
    os.path.join(
        FLAGS.data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)
    ]


def main(argv=None):
    global_step = tf.Variable(0,trainable=False)
    train_placeholder = tf.placeholder(tf.float32,
                                       shape=[32, 32, 3],
                                       name='input_image')

    label_placeholder = tf.placeholder(tf.int32,shape=[1],name='label')
    # (width, height, depth) -> (batch, width, height, depth)
    image_node = tf.expand_dims(train_placeholder, 0)

    logits = model.inference(image_node)
    total_loss = _loss(logits,label_placeholder)
    train_op = _train(total_loss,global_step)

    top_k_op = tf.nn.in_top_k(logits,label_placeholder,1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        total_duration = 0

        for epoch in range(1, FLAGS.epoch + 1):
            start_time = time.time()

            for file_index in range(5):
                print('Epoch %d: %s' % (epoch, filenames[file_index]))
                reader = Cifar10Reader(filenames[file_index])

                for index in range(10000):
                    image = reader.read(index)

                    logits_value = sess.run([logits],
                                            feed_dict={
                                                train_placeholder: image.byte_array
                                            })
                    _,loss_value = sess.run([train_op,total_loss],
                                             feed_dict={
                                                 train_placeholder: image.byte_array,
                                                 label_placeholder: image.label
                                             }
                    )

                    if index % 1000 == 0:
                        print('[%d]: %r' % (image.label, logits_value))

                    assert not np.isnan(loss_value), \
                        'Model diverged with loss = NaN'
                reader.close()

            duration = time.time() - start_time
            total_duration += duration

            prediction = _eval(sess,top_k_op,train_placeholder,label_placeholder)
            print('epoch %d duration = %d sec' % (epoch, duration))

            tf.train.SummaryWriter(FLAGS.checkpoint_dir, sess.graph)

        print('Total duration = %d sec' % total_duration)

def _eval(sess,top_k_op,train_placeholder,label_placeholder):
  if not FLAGS.test_data:
      return np.nan

  image_reader = Cifar10Reader(FLAGS.test_data)
  true_count = 0
  for index in range(10000):
      image = image_reader.read(index)

      predictions = sess.run([top_k_op],
                             feed_dict={
                                 input_image: image.image,
                                 label_placeholder:image.label
                             }
      )
      true_count += np.sum(predictions)
  image_reader.close()

if __name__ == '__main__':
    tf.app.run()


