# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict
from datasets.custom_json_dataset import ViratClassInfo
from datasets.custom_json_dataset import CustomJsonDataset


import os
import pickle
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import IM_DIR

def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds
# ------------------------------------------------------------------------------

def get_virat_dataset(name='virat2_mix'):
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()

    virat_data_info = ViratClassInfo()
    # classes = [
    #     '__background__', 'sedan', 'suv', 'truck', 'person'
    # ]
    temp = CustomJsonDataset(name+'_train')

    classes = temp.classes
    
    rotation_classes = virat_data_info.rotations
    color_classes = virat_data_info.colors
    x_classes = virat_data_info.x
    y_classes = virat_data_info.y

    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.color_classes = {i: name for i, name in enumerate(color_classes)}
    ds.rotation_classes = {i: float(name) for i, name in enumerate(rotation_classes)}
    ds.x_classes = {i: float(name) for i, name in enumerate(x_classes)}
    ds.y_classes = {i: float(name) for i, name in enumerate(y_classes)}

    if name.startswith('virat1'):
        name = 'virat1_mix'
    elif name.startswith('virat2'):
        name = 'virat2_mix'

    lookup_table_dir = DATASETS[name+'_train'][IM_DIR].replace('virat2_mix/images', 'lookuptables')

    # assert os.path.exists(lookup_table_dir), \
    #         'LookupTable directory \'{}\' not found'.format(lookup_table_dir)

    # with open(os.path.join(lookup_table_dir, 'sedan.p'), "rb") as f:
    #     ds.lookup_table = pickle.load(f)

    return ds
# ------------------------------------------------------------------------------
