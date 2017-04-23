# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from PIL import Image

from reader import Cifar10Reader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('file',None,"処理するファイルのパス")
tf.app.flags.DEFINE_integer('offset',0,"読み飛ばすレコード数")
tf.app.flags.DEFINE_integer('length',16,"読み込んで変換するレコード数")

basename = os.path.basename(FLAGS.file)
path = os.path.dirname(FLAGS.file)

reader = Cifar10Reader(FLAGS.file)

stop = FLAGS.offset + FLAGS.length

for index in range(FLAGS.offset,stop):
    image = reader.read(index)

    print('label: %d' % image.label)
    imageshow = Image.fromarray(image.byte_array.astype(np.unit8))

    file_name = '%s-%02d-%d.png' % (basename,index,image.label)
    file = os.path.join(path,file_name)
    with open(file,mode='wb') as out:
        imageshow.save(out,format='png')

reader.close()