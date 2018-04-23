#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright 2018 AI Futurelab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
# AI FUTURELAB
Image scene classification baseline for regional contest 2018

## Useage

# Training

```
python image_scene_classification.py --mode=train --dataset_dir=<dir> --max_steps=<max_steps> --checkpoint_dir=<checkpoint_dir>
```

 --mode Define the running mode
 --dataset_dir Path to directory of training set, e.g. /home/ubuntu/image_scene_training_v1
 --checkpoint_dir Path to directory of checkpoint files.
 --max_steps Maximum traning steps

# Test
```
python scene.py --mode=test --dataset_dir=<testset_dir> --checkpoint_dir=<checkpoint_dir> --target_file=<target_file>
```

 --mode Define the running mode
 --dataset_dir Path to directory of test set, e.g. /home/ubuntu/image_scene_test_v1
 --checkpoint_dir Path to directory of checkpoint files.
 --target_file Path to result file

'''

import tensorflow as tf
import numpy as np
import time, argparse

# local packages
import dataset

LEARNINGRATE = 1e-3
IMAGE_SIDE_LENGTH = 227
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
TOP_N = 3
CATEGORIES = 20

def train(train_dir, max_step, checkpoint_dir):
    # train the model
    scene_data = dataset.TraningSet(train_dir)
    features = tf.placeholder(np.float32, shape=[None, IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH, IMAGE_CHANNELS], name="features")
    labels = tf.placeholder(np.float32, [None], name="labels")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CATEGORIES)
    train_step, cross_entropy, logits, keep_prob = create_network(features, one_hot_labels)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % checkpoint.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, checkpoint.model_checkpoint_path)
            start_step = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('New training')

        start_time = time.time()
        for step in range(start_step, start_step + max_step):
            x, y = scene_data.next_batch(BATCH_SIZE, IMAGE_SIDE_LENGTH)
            sess.run(train_step, feed_dict={features: x, labels: y, keep_prob: 0.5})
            if step % 50 == 0 and step > 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: x, labels: y, keep_prob: 1})
                train_loss = sess.run(cross_entropy, feed_dict={features: x, labels: y, keep_prob: 1})
                duration = time.time() - start_time
                print ("Step %d: training accuracy %g, loss is %g (%0.3f sec)" % (step, train_accuracy, train_loss, duration))
                start_time = time.time()
            if step % 200 == 0 and step > 0:
                saver.save(sess, checkpoint_dir, global_step=step)
                print 'Write checkpoint at %s' % step

def test(test_dir, checkpoint_dir, target_file):
    # test_images = os.listdir(test_dir)
    listfile = test_dir+"/list.csv"
    test_list = dataset.read_csvlist(listfile)

    features = tf.placeholder(tf.float32, shape=[None, IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH, IMAGE_CHANNELS], name="features")
    labels = tf.placeholder(tf.float32, [None], name="labels")
    one_hot_categories = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CATEGORIES)
    train_step, cross_entropy, logits, keep_prob = create_network(features, one_hot_categories)
    values, indices = tf.nn.top_k(logits, TOP_N)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            raise Exception('Fail to load checkpoint')

        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Load checkpoint %s" % checkpoint.model_checkpoint_path

        results = []
        for row in test_list:
            file_id = row[0] #the first cell of a row is file_id
            imagefile = test_dir+'/data/'+file_id+'.jpg'
            x = dataset.img_resize(imagefile, IMAGE_SIDE_LENGTH)
            feed_dict = {features: np.expand_dims(x, axis=0), keep_prob: 1}
            y = np.squeeze(sess.run(indices, feed_dict = feed_dict), axis=0)
            result = [file_id]
            result.extend(y.tolist())
            results.append(result)
            print('FileID: %s, Categroies: %d,%d,%d' % (file_id, y[0], y[1], y[2]))

        dataset.write_csvlist(target_file, results, header=['FILE_ID', 'CATEGORY_ID0', 'CATEGORY_ID1', 'CATEGORY_ID2'])
        print "Write results to file %s" % target_file




def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')

def create_network(features, one_hot_labels):
    # network structure
    # conv1
    W_conv1 = weight_variable([11, 11, 3, 64], stddev=1e-4)
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(features, W_conv1) + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    # norm1
    norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # conv2
    W_conv2 = weight_variable([5, 5, 64, 64], stddev=1e-2)
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
    # norm2
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    h_pool2 = tf.nn.max_pool(norm2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    # conv3
    W_conv3 = weight_variable([5, 5, 64, 64], stddev=1e-2)
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # conv4
    W_conv4 = weight_variable([5, 5, 64, 64], stddev=1e-2)
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    h_pool4 = tf.nn.max_pool(h_conv4, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    # fc1
    W_fc1 = weight_variable([29 * 29 * 64, 128])
    b_fc1 = bias_variable([128])
    h_pool3_flat = tf.reshape(h_pool4, [-1, 29 * 29 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # introduce dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # fc2
    W_fc2 = weight_variable([128, 20])
    b_fc2 = bias_variable([20])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # calculate loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer(LEARNINGRATE).minimize(cross_entropy)

    return train_step, cross_entropy, y_conv, keep_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="Define the running mode as 'training' or 'test'.")
    parser.add_argument('--dataset_dir', type=str, help="Path to directory of training set or test set, depends on the running mode.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help="Path to directory of checkpoint.")
    parser.add_argument('--max_steps', type=int, default=65000, help="Maximum training steps.")
    parser.add_argument('--target_file', type=str, default='./test_results.csv', help='Path to test result file.')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.dataset_dir, args.max_steps, args.checkpoint_dir)
    elif args.mode == 'test':
        test(args.dataset_dir, args.checkpoint_dir, args.target_file)
    else:
        raise Exception('--mode can be train or test only')