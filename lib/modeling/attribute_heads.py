import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np

class attribute_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        # -----------------------------------------------------
        self.color_cls_score_fc1 = nn.Linear(dim_in, 100)
        self.color_cls_score_fc2 = nn.Linear(100, cfg.MODEL.COLOR_NUM_CLASSES)

        self.rotation_cls_score_fc1 = nn.Linear(dim_in, 100)
        self.rotation_cls_score_fc2 = nn.Linear(100, cfg.MODEL.ROTATION_NUM_CLASSES)

        self.x_cls_score_fc1 = nn.Linear(dim_in, 100)
        self.x_cls_score_fc2 = nn.Linear(100, cfg.MODEL.X_NUM_CLASSES)

        self.y_cls_score_fc1 = nn.Linear(dim_in, 100)
        self.y_cls_score_fc2 = nn.Linear(100, cfg.MODEL.Y_NUM_CLASSES)

        # -----------------------------------------------------

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.color_cls_score_fc1.weight, std=0.01)
        init.constant_(self.color_cls_score_fc1.bias, 0)
        
        init.normal_(self.color_cls_score_fc2.weight, std=0.01)
        init.constant_(self.color_cls_score_fc2.bias, 0)
        
        init.normal_(self.rotation_cls_score_fc1.weight, std=0.01)
        init.constant_(self.rotation_cls_score_fc1.bias, 0)

        init.normal_(self.rotation_cls_score_fc2.weight, std=0.01)
        init.constant_(self.rotation_cls_score_fc2.bias, 0)
        
        init.normal_(self.x_cls_score_fc1.weight, std=0.01)
        init.constant_(self.x_cls_score_fc1.bias, 0)

        init.normal_(self.x_cls_score_fc2.weight, std=0.01)
        init.constant_(self.x_cls_score_fc2.bias, 0)
        
        init.normal_(self.y_cls_score_fc1.weight, std=0.01)
        init.constant_(self.y_cls_score_fc1.bias, 0)

        init.normal_(self.y_cls_score_fc2.weight, std=0.01)
        init.constant_(self.y_cls_score_fc2.bias, 0)
        
        # ------------------------------------------
        return


    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'color_cls_score_fc1.weight': 'color_cls_score_fc1_w',
            'color_cls_score_fc1.bias': 'color_cls_score_fc1_b',
            'color_cls_score_fc2.weight': 'color_cls_score_fc2_w',
            'color_cls_score_fc2.bias': 'color_cls_score_fc2_b',
            
            'rotation_cls_score_fc1.weight': 'rotation_cls_score_fc1_w',
            'rotation_cls_score_fc1.bias': 'rotation_cls_score_fc1_b',
            'rotation_cls_score_fc2.weight': 'rotation_cls_score_fc2_w',
            'rotation_cls_score_fc2.bias': 'rotation_cls_score_fc2_b',
            
            'x_cls_score_fc1.weight': 'x_cls_score_fc1_w',
            'x_cls_score_fc1.bias': 'x_cls_score_fc1_b',
            'x_cls_score_fc2.weight': 'x_cls_score_fc2_w',
            'x_cls_score_fc2.bias': 'x_cls_score_fc2_b',

            'y_cls_score_fc1.weight': 'y_cls_score_fc1_w',
            'y_cls_score_fc1.bias': 'y_cls_score_fc1_b',
            'y_cls_score_fc2.weight': 'y_cls_score_fc2_w',
            'y_cls_score_fc2.bias': 'y_cls_score_fc2_b'
        }

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        
        color_cls_score = self.color_cls_score_fc2(F.relu(self.color_cls_score_fc1(x), inplace=True))
        rotation_cls_score = self.rotation_cls_score_fc2(F.relu(self.rotation_cls_score_fc1(x), inplace=True))
        x_cls_score = self.x_cls_score_fc2(F.relu(self.x_cls_score_fc1(x), inplace=True))
        y_cls_score = self.y_cls_score_fc2(F.relu(self.y_cls_score_fc1(x), inplace=True))

        if not self.training:
            color_cls_score = F.softmax(color_cls_score, dim=1)
            rotation_cls_score = F.softmax(rotation_cls_score, dim=1)
            x_cls_score = F.softmax(x_cls_score, dim=1)
            y_cls_score = F.softmax(y_cls_score, dim=1)

        return color_cls_score, rotation_cls_score, x_cls_score, y_cls_score

# -----------------------------------------------------------------------------------------------------------

def attribute_losses(color_cls_score, rotation_cls_score,
                     x_cls_score, y_cls_score,
                     labels_int32,
                     color_label_int32, rotation_label_int32,
                     x_label_int32, y_label_int32):
    # --------------------------------------------
    valid_inds = (labels_int32 > 0)
    
    #chop off negative ROIs
    color_label_int32 = color_label_int32[valid_inds]
    rotation_label_int32 = rotation_label_int32[valid_inds]
    x_label_int32 = x_label_int32[valid_inds]
    y_label_int32 = y_label_int32[valid_inds]

    device_id = color_cls_score.get_device()
    valid_inds = Variable(torch.from_numpy(valid_inds.astype(int))).cuda(device_id)
    valid_inds = torch.nonzero(valid_inds).view(-1)
    color_cls_score = color_cls_score[valid_inds]
    rotation_cls_score = rotation_cls_score[valid_inds]
    x_cls_score = x_cls_score[valid_inds]
    y_cls_score = y_cls_score[valid_inds]

    #only compute loss on valid labels
    device_id = color_cls_score.get_device()
    color_rois_label = Variable(torch.from_numpy(color_label_int32.astype('int64'))).cuda(device_id)
    color_loss_cls = F.cross_entropy(color_cls_score, color_rois_label)
    color_cls_preds = color_cls_score.max(dim=1)[1].type_as(color_rois_label)
    color_accuracy_cls = color_cls_preds.eq(color_rois_label).float().mean(dim=0)

    device_id = rotation_cls_score.get_device()
    rotation_rois_label = Variable(torch.from_numpy(rotation_label_int32.astype('int64'))).cuda(device_id)
    rotation_loss_cls = F.cross_entropy(rotation_cls_score, rotation_rois_label)
    rotation_cls_preds = rotation_cls_score.max(dim=1)[1].type_as(rotation_rois_label)
    rotation_accuracy_cls = rotation_cls_preds.eq(rotation_rois_label).float().mean(dim=0)

    device_id = x_cls_score.get_device()
    x_rois_label = Variable(torch.from_numpy(x_label_int32.astype('int64'))).cuda(device_id)
    x_loss_cls = F.cross_entropy(x_cls_score, x_rois_label)
    x_cls_preds = x_cls_score.max(dim=1)[1].type_as(x_rois_label)
    x_accuracy_cls = x_cls_preds.eq(x_rois_label).float().mean(dim=0)

    device_id = y_cls_score.get_device()
    y_rois_label = Variable(torch.from_numpy(y_label_int32.astype('int64'))).cuda(device_id)
    y_loss_cls = F.cross_entropy(y_cls_score, y_rois_label)
    y_cls_preds = y_cls_score.max(dim=1)[1].type_as(y_rois_label)
    y_accuracy_cls = y_cls_preds.eq(y_rois_label).float().mean(dim=0)

    # --------------------------------------------

    return  color_loss_cls, color_accuracy_cls,\
            rotation_loss_cls, rotation_accuracy_cls,\
            x_loss_cls, x_accuracy_cls,\
            y_loss_cls, y_accuracy_cls

# -----------------------------------------------------------------------------------------------------------
