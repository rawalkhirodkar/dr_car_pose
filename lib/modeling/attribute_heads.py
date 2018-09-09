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
        self.rotation_cls_score_fc1 = nn.Linear(dim_in, 100)
        self.rotation_cls_score_fc2 = nn.Linear(100, cfg.ROTATION.NUM_CLASSES)
        # -----------------------------------------------------

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.rotation_cls_score_fc1.weight, std=0.01)
        init.constant_(self.rotation_cls_score_fc1.bias, 0)

        init.normal_(self.rotation_cls_score_fc2.weight, std=0.01)
        init.constant_(self.rotation_cls_score_fc2.bias, 0)
        # ------------------------------------------
        return


    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'rotation_cls_score_fc1.weight': 'rotation_cls_score_fc1_w',
            'rotation_cls_score_fc1.bias': 'rotation_cls_score_fc1_b',
            'rotation_cls_score_fc2.weight': 'rotation_cls_score_fc2_w',
            'rotation_cls_score_fc2.bias': 'rotation_cls_score_fc2_b'   
        }

        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        
        rotation_cls_score = self.rotation_cls_score_fc2(F.relu(self.rotation_cls_score_fc1(x), inplace=True))

        if not self.training:
            rotation_cls_score = F.softmax(rotation_cls_score, dim=1)

        return rotation_cls_score

# -----------------------------------------------------------------------------------------------------------

def attribute_losses(cls_score,
                     rotation_cls_score,
                     labels_int32,
                     rotation_label_int32):
    # --------------------------------------------
    valid_inds = ((rotation_label_int32 >= 0) * (labels_int32 > 0)) #no color and rotation for person
    device_id = rotation_cls_score.get_device()

    if valid_inds.sum() == 0:
        return rotation_cls_score.sum()*0.0, torch.tensor(0.0).cuda(device_id)
    
    #chop off negative ROIs
    rotation_label_int32 = rotation_label_int32[valid_inds]

    valid_inds = Variable(torch.from_numpy(valid_inds.astype(int))).cuda(device_id)
    valid_inds = torch.nonzero(valid_inds).view(-1)
    rotation_cls_score = rotation_cls_score[valid_inds]

    #only compute loss on valid labels
    device_id = rotation_cls_score.get_device()
    rotation_rois_label = Variable(torch.from_numpy(rotation_label_int32.astype('int64'))).cuda(device_id)
    rotation_loss_cls = F.cross_entropy(rotation_cls_score, rotation_rois_label)
    rotation_cls_preds = rotation_cls_score.max(dim=1)[1].type_as(rotation_rois_label)
    rotation_accuracy_cls = rotation_cls_preds.eq(rotation_rois_label).float().mean(dim=0)

    # --------------------------------------------
    return  rotation_loss_cls, rotation_accuracy_cls

# -----------------------------------------------------------------------------------------------------------
