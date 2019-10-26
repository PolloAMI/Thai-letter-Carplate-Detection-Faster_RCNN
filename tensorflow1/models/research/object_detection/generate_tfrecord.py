#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'txt01':
        return 1
    elif row_label == 'txt02':
        return 2
    elif row_label == 'txt03':
        return 3
    elif row_label == 'txt04':
        return 4
    elif row_label == 'txt05':
        return 5
    elif row_label == 'txt06':
        return 6
    elif row_label == 'txt07':
        return 7
    elif row_label == 'txt08':
        return 8
    elif row_label == 'txt09':
        return 9
    elif row_label == 'txt10':
        return 10
    elif row_label == 'txt11':
        return 11
    elif row_label == 'txt12':
        return 12
    elif row_label == 'txt13':
        return 13
    elif row_label == 'txt14':
        return 14
    elif row_label == 'txt16':
        return 16
    elif row_label == 'txt17':
        return 17
    elif row_label == 'txt18':
        return 18
    elif row_label == 'txt19':
        return 19
    elif row_label == 'txt20':
        return 20
    elif row_label == 'txt21':
        return 21
    elif row_label == 'txt22':
        return 22
    elif row_label == 'txt23':
        return 23
    elif row_label == 'txt25':
        return 25
    elif row_label == 'txt26':
        return 26
    elif row_label == 'txt27':
        return 27
    elif row_label == 'txt28':
        return 28
    elif row_label == 'txt29':
        return 29
    elif row_label == 'txt30':
        return 30
    elif row_label == 'txt31':
        return 31
    elif row_label == 'txt32':
        return 32
    elif row_label == 'txt33':
        return 33
    elif row_label == 'txt34':
        return 34
    elif row_label == 'txt35':
        return 35
    elif row_label == 'txt36':
        return 36
    elif row_label == 'txt37':
        return 37
    elif row_label == 'txt38':
        return 38
    elif row_label == 'txt39':
        return 39
    elif row_label == 'num1':
        return 41
    elif row_label == 'num2':
        return 42
    elif row_label == 'num3':
        return 43
    elif row_label == 'num4':
        return 44
    elif row_label == 'num5':
        return 45
    elif row_label == 'num6':
        return 46
    elif row_label == 'num7':
        return 47
    elif row_label == 'num9':
        return 49
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(str(row['class']).encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
