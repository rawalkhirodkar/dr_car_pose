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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import cv2
import sys
# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------

class ViratClassInfo():

    def __init__(self):
        # self.colors = ('black', 'red', 'blue', 'white')
        self.colors = ('white', 'black', 'red', 'blue', 'brown', 'silver', 'cyan', 'yellow')
        self.colors_to_ind = dict(zip(self.colors, range(len(self.colors)))) #class ind: 0, 1, 2, 3

        # ----------------------------
        MAX_VAL = 360.0; MIN_VAL = 0.0; STEP = 10.0
        temp = list(np.arange(MIN_VAL, MAX_VAL, STEP) + STEP/2)
        self.rotations = [str(x) for x in temp] #20 classes
        self.rotations_to_ind = dict(zip(self.rotations, range(len(self.rotations))))
        self.rotation_max_val = MAX_VAL
        self.rotation_min_val = MIN_VAL
        self.rotation_range = MAX_VAL - MIN_VAL

        # ----------------------------
        MAX_VAL = 1.0; MIN_VAL = -1.0; STEP = 0.1
        temp = list(np.arange(MIN_VAL, MAX_VAL, STEP) + STEP/2)
        self.x = [str(x) for x in temp] #20 classes
        self.x_to_ind = dict(zip(self.x, range(len(self.x))))
        self.x_max_val = MAX_VAL
        self.x_min_val = MIN_VAL
        self.x_range = MAX_VAL - MIN_VAL
        
        # ----------------------------
        MAX_VAL = 1.0; MIN_VAL = -1.0; STEP = 0.1
        temp = list(np.arange(MIN_VAL, MAX_VAL, STEP) + STEP/2)
        self.y = [str(x) for x in temp] #20 classes
        self.y_to_ind = dict(zip(self.y, range(len(self.y))))
        self.y_max_val = MAX_VAL
        self.y_min_val = MIN_VAL
        self.y_range = MAX_VAL - MIN_VAL
        
        # ----------------------------
        assert(len(self.colors) == cfg.MODEL.COLOR_NUM_CLASSES)
        assert(len(self.rotations) == cfg.MODEL.ROTATION_NUM_CLASSES)
        assert(len(self.x) == cfg.MODEL.X_NUM_CLASSES)
        assert(len(self.y) == cfg.MODEL.Y_NUM_CLASSES)
        # ------------------------------

        return

    def get_color_id(self, raw_color):
        raw_color = raw_color.lower().strip()
        color_id = self.colors_to_ind[raw_color]
        return color_id

    def get_rotation_id(self, raw_rotation):
        if(raw_rotation == self.rotation_max_val):
            raw_rotation = self.rotation_min_val

        degree_per_class = self.rotation_range/len(self.rotations) #should be equal to 10
        
        raw_rotation = raw_rotation - self.rotation_min_val #normalize
        rotation_id = int(raw_rotation/degree_per_class) #take floor, 0 to 35
        assert(rotation_id < len(self.rotations))
        return rotation_id

    def get_x_id(self, raw_x):
        if(raw_x == self.x_max_val):
            raw_x = self.x_min_val

        val_per_class = self.x_range/len(self.x) #should be equal to 0.1

        raw_x = raw_x - self.x_min_val
        x_id = int(raw_x/val_per_class) #take floor, 0 to 35
        assert(x_id < len(self.x))
        return x_id

    def get_y_id(self, raw_y):
        if(raw_y == self.y_max_val):
            raw_y = self.y_min_val

        val_per_class = self.y_range/len(self.y) #should be equal to 10
        
        raw_y = raw_y - self.y_min_val
        y_id = int(raw_y/val_per_class) #take floor, 0 to 35
        assert(y_id < len(self.y))
        return y_id

    def get_depth(self, depth_path):
        gt_depth = cv2.imread(depth_path)
        gt_depth = cv2.resize(gt_depth, (cfg.MODEL.DEPTH_WIDTH, cfg.MODEL.DEPTH_HEIGHT), interpolation=cv2.INTER_LINEAR)
        gt_depth_labels = gt_depth[:, :, 0].astype(np.float32)
        gt_depth_labels = gt_depth_labels/(255.0 - 0.0) #max depth is 255, min depth is 0
        gt_depth_labels = gt_depth_labels*(cfg.MODEL.DEPTH_NUM_CLASSES-1) #in total cfg.MODEL.DEPTH_NUM_CLASSES
        gt_depth_labels = np.rint(gt_depth_labels)
        gt_depth_labels = gt_depth_labels.astype(np.uint8) #labels < 255

        return gt_depth_labels

    def get_normal(self, normal_path):
        gt_normal = cv2.imread(normal_path)
        gt_normal = cv2.resize(gt_normal, (cfg.MODEL.NORMAL_WIDTH, cfg.MODEL.NORMAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        gt_normal = gt_normal.astype(np.float32)        
        per_component_class = round((cfg.MODEL.NORMAL_NUM_CLASSES)**(1./3)) #should be 4
        assert(per_component_class**3 == cfg.MODEL.NORMAL_NUM_CLASSES)

        gt_normal = gt_normal/(255.0 - 0.0) #each dimension is from 0 to 255
        gt_normal = gt_normal*(per_component_class - 1)
        gt_normal = np.rint(gt_normal)
        #(i,j,k) -> 16i + 4j + k
        gt_normal_labels = (per_component_class**2)*gt_normal[:,:,0] + (per_component_class**1)*gt_normal[:,:,1] + (per_component_class**0)*gt_normal[:,:,2]

        gt_normal_labels = np.rint(gt_normal_labels)
        gt_normal_labels = gt_normal_labels.astype(np.uint8)
        return gt_normal_labels

    def get_seg(self, seg_path):
        gt_seg = cv2.imread(seg_path)
        gt_seg = cv2.resize(gt_seg, (cfg.MODEL.DEPTH_WIDTH, cfg.MODEL.DEPTH_HEIGHT), interpolation=cv2.INTER_LINEAR)
        gt_seg = gt_seg.astype(np.uint8)
        return gt_seg
# ------------------------------------------------------------------
# My own dataset overload to handle more attributes
class CustomJsonDataset(object):
    """A class representing a VIRAT json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.COCO = COCO(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()

        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)

        self.virat_class_info = ViratClassInfo()

        # eg: {1: 1, 2: 2, 3: 3}
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }

        # eg: {1: 1, 2: 2, 3: 3}
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self._init_keypoints()

        # # Set cfg.MODEL.NUM_CLASSES
        # if cfg.MODEL.NUM_CLASSES != -1:
        #     assert cfg.MODEL.NUM_CLASSES == 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes, \
        #         "number of classes should equal when using multiple datasets"
        # else:
        #     cfg.MODEL.NUM_CLASSES = 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        #add new attributes to the end
        keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map', 'gt_colors', 'gt_rotations',
                'gt_x', 'gt_y', 'gt_depth', 'gt_normal', 'gt_seg', 'gt_is_real']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds() #list of all image ids
        image_ids.sort()

        if cfg.DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:100]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))

        # eg: entry = {'file_name': '000439.JPEG', 'coco_url': '', 'width': 1440, 'id': 1, 'date_captured': '2018-05-05 04:04:18.098860', 'flickr_url': '', 'license': 1, 'height': 810}
        for i, entry in enumerate(roidb):
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            cache_filepath = os.path.join(self.cache_path, self.name+'_gt_roidb.pkl')
            if os.path.exists(cache_filepath) and not cfg.DEBUG:
                self.debug_timer.tic()
                self._add_gt_from_cache(roidb, cache_filepath)
                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                for entry in roidb:
                    sys.stdout.write("Creating roidb: %d%%   \r" % ((i+1)*100/len(roidb)) )
                    sys.stdout.flush()
                    self._add_gt_annotations(entry)
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                if not cfg.DEBUG:
                    with open(cache_filepath, 'wb') as fp:
                        pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', cache_filepath)
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self #pointer to the JSON Dataset object
        
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )

        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        # ------------------------------------------------------
        entry['gt_colors'] = np.empty((0), dtype=np.int32)
        entry['gt_rotations'] = np.empty((0), dtype=np.int32)
        entry['gt_x'] = np.empty((0), dtype=np.int32)
        entry['gt_y'] = np.empty((0), dtype=np.int32)

        entry['gt_depth'] = np.empty((0), dtype=np.int32)
        entry['gt_normal'] = np.empty((0), dtype=np.int32)
        entry['gt_seg'] = np.empty((0), dtype=np.int32)
        
        entry['gt_is_real'] = np.empty((0), dtype=np.int32) #0 or 1
        # ------------------------------------------------------        
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]
        return

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids) #len(objs) == num_objects in the scene

        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']

        gt_is_real = None
        #obj has all the information from the json
        for obj in objs:
            # ---------------------------------
            gt_is_real = obj['is_real']
            # ---------------------------------
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])

        
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        # ------------------------------------------------------------------------
        gt_colors = np.zeros((num_valid_objs), dtype=entry['gt_colors'].dtype)
        gt_rotations = np.zeros((num_valid_objs), dtype=entry['gt_rotations'].dtype)
        gt_x = np.zeros((num_valid_objs), dtype=entry['gt_x'].dtype)
        gt_y = np.zeros((num_valid_objs), dtype=entry['gt_y'].dtype)

        if(gt_is_real):
            gt_depth = None
            gt_normal = None
            gt_seg = None
        else:
            gt_depth = self.virat_class_info.get_depth(entry['image'].replace("images", "depths")) #path to the depth file
            gt_normal = self.virat_class_info.get_normal(entry['image'].replace("images", "normals")) #path to the normal file
            gt_seg = self.virat_class_info.get_normal(entry['image'].replace("images", "segs")) #path to the normal file

        # ------------------------------------------------------------------------
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            # --------some hack-------------
            if(obj['is_real'] and obj['category_id'] == 5):
                obj['category_id'] = 4
            # -------------------------------            
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            # --------------------------------------------------------
            gt_colors[ix] = self.virat_class_info.get_color_id(obj['color'])
            gt_rotations[ix] = self.virat_class_info.get_rotation_id(obj['rotation'])
            gt_x[ix] = self.virat_class_info.get_x_id(obj['x'])
            gt_y[ix] = self.virat_class_info.get_y_id(obj['y'])
            # --------------------------------------------------------
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        # --------------------------------------------------------------
        entry['gt_colors'] = np.append(entry['gt_colors'], gt_colors)
        entry['gt_rotations'] = np.append(entry['gt_rotations'], gt_rotations)
        entry['gt_x'] = np.append(entry['gt_x'], gt_x)
        entry['gt_y'] = np.append(entry['gt_y'], gt_y)

        entry['gt_depth'] = gt_depth
        entry['gt_normal'] = gt_normal
        entry['gt_seg'] = gt_seg
        entry['gt_is_real'] = gt_is_real
        # --------------------------------------------------------------
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

        return
    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        assert len(roidb) == len(cached_roidb)

        real_images = 0
        for entry, cached_entry in zip(roidb, cached_roidb):

            # # valid_cached_keys = keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
            #     'is_crowd', 'box_to_gt_ind_map', 'gt_colors', 'gt_rotations',
            #     'gt_x', 'gt_y', 'gt_depth', 'gt_normal', 'gt_seg', 'gt_is_real']
            # ----------------------------------------------------------------
            values = [cached_entry[key] for key in self.valid_cached_keys] #note the order of the keys matter
            boxes, segms, gt_classes, seg_areas, \
            gt_overlaps, is_crowd, box_to_gt_ind_map, \
            gt_colors, gt_rotations, gt_x, gt_y, gt_depth, gt_normal, gt_seg, gt_is_real = values[:7+4+2+1+1]

            if(gt_is_real):
                real_images += 1
                gt_depth = None
                gt_normal = None
                gt_seg = None
            else:
                gt_depth = cv2.resize(gt_depth.astype(np.uint8), (cfg.MODEL.DEPTH_WIDTH, cfg.MODEL.DEPTH_HEIGHT))
                gt_normal = cv2.resize(gt_normal.astype(np.uint8), (cfg.MODEL.NORMAL_WIDTH, cfg.MODEL.NORMAL_HEIGHT))
                gt_seg = cv2.resize(gt_seg.astype(np.uint8), (cfg.MODEL.DEPTH_WIDTH, cfg.MODEL.DEPTH_HEIGHT))
            # ----------------------------------------------------------------

            if self.keypoints is not None:
                gt_keypoints, has_visible_keypoints = values[7:]
            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'].extend(segms)
            # To match the original implementation:
            # entry['boxes'] = np.append(
            #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )
            # ----------------------------------------------------------------
            entry['gt_colors'] = np.append(entry['gt_colors'], gt_colors)
            entry['gt_rotations'] = np.append(entry['gt_rotations'], gt_rotations)
            entry['gt_x'] = np.append(entry['gt_x'], gt_x)
            entry['gt_y'] = np.append(entry['gt_y'], gt_y)

            entry['gt_depth'] = gt_depth
            entry['gt_normal'] = gt_normal
            entry['gt_seg'] = gt_seg
            entry['gt_is_real'] = gt_is_real

            # ----------------------------------------------------------------

            if self.keypoints is not None:
                entry['gt_keypoints'] = np.append(
                    entry['gt_keypoints'], gt_keypoints, axis=0
                )
                entry['has_visible_keypoints'] = has_visible_keypoints

        print("Loaded {} real images...".format(real_images))
        return
    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            if cfg.KRCNN.NUM_KEYPOINTS != -1:
                assert cfg.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
                    "number of keypoints should equal when using multiple datasets"
            else:
                cfg.KRCNN.NUM_KEYPOINTS = self.num_keypoints
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)

    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        # --------------------------------------------------------
        #initialize as -1 for background classes
        entry['gt_colors'] = np.append(
            entry['gt_colors'],
            np.zeros((num_boxes), dtype=entry['gt_colors'].dtype) - 1
        )

        entry['gt_rotations'] = np.append(
            entry['gt_rotations'],
            np.zeros((num_boxes), dtype=entry['gt_rotations'].dtype) - 1
        )

        entry['gt_x'] = np.append(
            entry['gt_x'],
            np.zeros((num_boxes), dtype=entry['gt_x'].dtype) - 1
        )

        entry['gt_y'] = np.append(
            entry['gt_y'],
            np.zeros((num_boxes), dtype=entry['gt_y'].dtype) - 1
        )
        # --------------------------------------------------------

        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )

    return

def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]
