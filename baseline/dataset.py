#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright 2018 AI Futurelab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import PIL.Image as Image
import csv, random

random.seed(1)

class TraningSet(object):

    def __init__(self, dataset_dir):
        # get all the image name and its label
        label_path = dataset_dir + '/list.csv'
        self.all_data = read_csvlist(label_path)
        random.shuffle(self.all_data)
        self.start = 0
        self.end = 0
        self.dataset_dir = dataset_dir

    def next_batch(self, batch_size, img_side_length):
        # fetch a batch of images
        size_of_dataset = len(self.all_data)
        self.start = self.end
        if self.start >= size_of_dataset:
            self.start = 0
            print '------ New iteration -----'
        images = []
        categories = []
        index = self.start
        while len(images) < batch_size:
            if index >= size_of_dataset:
                index = 0
            file_id, category_id = self.all_data[index]
            file_path = self.dataset_dir+'/data/'+ file_id + '.jpg'
            images.append(img_resize(file_path, img_side_length))
            categories.append(int(category_id))
            index += 1
        self.end = index
        return np.array(images), np.asarray(categories)

def write_csvlist(listfile, data, header = []):
    with open(listfile, 'wb') as fh:
        csv_writer = csv.writer(fh)
        if (len(header)):
            csv_writer.writerow(header)
        results = []
        for row in data:
            csv_writer.writerow(row)

def read_csvlist(listfile):
    csv_reader = csv.reader(open(listfile))
    is_first_line = True
    results = []
    for row in csv_reader:
        # Skip the head line.
        if (is_first_line):
            is_first_line = False
            continue
        results.append(row)
    return results

def img_resize(filepath, side_length):
    image = Image.open(filepath)
    if image.mode == 'L':
        image = image.convert('RGB')
    short_length = min(image.width, image.height)
    left = (image.width - short_length)/2
    top = (image.height - short_length)/2
    image = image.crop((left, top, short_length, short_length));
    image = image.resize((side_length, side_length))
    image_array = (np.asarray(image, np.float32)-127)/255
    return image_array
