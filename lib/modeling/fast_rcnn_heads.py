import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np

class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            self.bbox_pred = nn.Linear(dim_in, 4)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

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
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

        # ------------------------------------------
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

        # ------------------------------------------------
        # detectron_weight_mapping = {
        #     'cls_score.weight': 'cls_score_w',
        #     'cls_score.bias': 'cls_score_b',
        #     'bbox_pred.weight': 'bbox_pred_w',
        #     'bbox_pred.bias': 'bbox_pred_b'
        # }

        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b',
            
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

        # -----------------------------------------------


        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)


        # ------------------------------------------
        color_cls_score = self.color_cls_score_fc2(F.relu(self.color_cls_score_fc1(x), inplace=True))
        rotation_cls_score = self.rotation_cls_score_fc2(F.relu(self.rotation_cls_score_fc1(x), inplace=True))
        x_cls_score = self.x_cls_score_fc2(F.relu(self.x_cls_score_fc1(x), inplace=True))
        y_cls_score = self.y_cls_score_fc2(F.relu(self.y_cls_score_fc1(x), inplace=True))

        if not self.training:
            color_cls_score = F.softmax(color_cls_score, dim=1)
            rotation_cls_score = F.softmax(rotation_cls_score, dim=1)
            x_cls_score = F.softmax(x_cls_score, dim=1)
            y_cls_score = F.softmax(y_cls_score, dim=1)

        # ------------------------------------------

        # return cls_score, bbox_pred
        return cls_score, bbox_pred, \
                color_cls_score, rotation_cls_score, x_cls_score, y_cls_score

# -----------------------------------------------------------------------------------------------------------
# def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
#                      bbox_inside_weights, bbox_outside_weights):
#     device_id = cls_score.get_device()
#     rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
#     loss_cls = F.cross_entropy(cls_score, rois_label)

#     bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
#     bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
#     bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
#     loss_bbox = net_utils.smooth_l1_loss(
#         bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

#     # class accuracy
#     cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
#     accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)

#     return loss_cls, loss_bbox, accuracy_cls

def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights,
                     color_cls_score, rotation_cls_score,
                     x_cls_score, y_cls_score,
                     color_label_int32, rotation_label_int32,
                     x_label_int32, y_label_int32):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    loss_cls = F.cross_entropy(cls_score, rois_label)

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)
    # --------------------------------------------
    valid_inds = (color_label_int32 >= 0)
    assert(np.all(valid_inds == (rotation_label_int32 >= 0)) == True)
    assert(np.all(valid_inds == (x_label_int32 >= 0)) == True)
    assert(np.all(valid_inds == (y_label_int32 >= 0)) == True)

    #chop off negative ROIs
    color_label_int32 = color_label_int32[valid_inds]
    rotation_label_int32 = rotation_label_int32[valid_inds]
    x_label_int32 = x_label_int32[valid_inds]
    y_label_int32 = y_label_int32[valid_inds]

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

    return loss_cls, loss_bbox, accuracy_cls,\
            color_loss_cls, color_accuracy_cls,\
            rotation_loss_cls, rotation_accuracy_cls,\
            x_loss_cls, x_accuracy_cls,\
            y_loss_cls, y_accuracy_cls


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x
