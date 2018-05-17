import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from core.config import cfg
import utils.net as net_utils
import utils.blob as blob_utils
import modeling.ResNet as ResNet
from modeling.generate_anchors import generate_anchors
from modeling.generate_proposals import GenerateProposalsOp
from modeling.collect_and_distribute_fpn_rpn_proposals import CollectAndDistributeFpnRpnProposalsOp
import nn as mynn
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np


# ---------------------------------------------------------------------------- #
# Depth Prediction with an FPN backbone
# ---------------------------------------------------------------------------- #
#list of len 5: 
#2x256x192x336
#2x256x96x168
#2x256x48x84
#2x256x24x42
#2x256x12x21, 

# --------------------------------------------------------
  
#entry from model_builder.py
class depth_outputs(nn.Module):
    """Add Depth on FPN specific outputs."""
    def __init__(self, dim_in, spatial_scales):
        super().__init__()
        self.dim_in = dim_in #mostly 256
        self.spatial_scales = spatial_scales
        self.dim_out = self.dim_in

        # Create conv ops shared by all FPN levels
        # first up sample all the features to 256x192x336, 
        # and then add them all
        # and then apply convolution and up sample and conv 1 x 1 for classification


        #this will be the final conv, then relu and then softmax

        #store 5 transformations


        dim_score = cfg.MODEL.DEPTH_NUM_CLASSES
        self.FPN_depth_conv1 = nn.Conv2d(self.dim_in, self.dim_in, 3, stride=1, padding=1) #256 in, 256 out
        self.FPN_depth_conv2 = nn.Conv2d(self.dim_in, self.dim_in, 3, stride=1, padding=1) #256 in, 256 out
        self.FPN_depth_conv3 = nn.Conv2d(self.dim_in, self.dim_in, 3, stride=1, padding=1) #256 in, 256 out
        self.FPN_depth_cls_score = nn.Conv2d(self.dim_out, dim_score, 1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.FPN_depth_conv1.weight, std=0.01)
        init.constant_(self.FPN_depth_conv1.bias, 0)
        init.normal_(self.FPN_depth_conv2.weight, std=0.01)
        init.constant_(self.FPN_depth_conv2.bias, 0)
        init.normal_(self.FPN_depth_conv3.weight, std=0.01)
        init.constant_(self.FPN_depth_conv3.bias, 0)

        init.normal_(self.FPN_depth_cls_score.weight, std=0.01)
        init.constant_(self.FPN_depth_cls_score.bias, 0)

    def detectron_weight_mapping(self):
        k_min = cfg.FPN.RPN_MIN_LEVEL #same as the RPN
        mapping_to_detectron = {
            'FPN_depth_conv1.weight': 'conv1_depth_fpn%d_w' % k_min,
            'FPN_depth_conv1.bias': 'conv1_depth_fpn%d_b' % k_min,

            'FPN_depth_conv2.weight': 'conv2_depth_fpn%d_w' % k_min,
            'FPN_depth_conv2.bias': 'conv2_depth_fpn%d_b' % k_min,
            
            'FPN_depth_conv3.weight': 'conv3_depth_fpn%d_w' % k_min,
            'FPN_depth_conv3.bias': 'conv3_depth_fpn%d_b' % k_min,
            
            'FPN_depth_cls_score.weight': 'depth_cls_logits_fpn%d_w' % k_min,
            'FPN_depth_cls_score.bias': 'depth_cls_logits_fpn%d_b' % k_min
        }
        return mapping_to_detectron, []

    def forward(self, blobs_in, im_info, roidb=None):
        
        # ---------------------------------------
        k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
        k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
        assert len(blobs_in) == k_max - k_min + 1
        return_dict = {}
        feature = 0

        # import pdb; pdb.set_trace()

        for lvl in range(k_min, k_max + 1):
            slvl = str(lvl)
            bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
            bl_in = self.FPN_depth_conv2(F.relu(self.FPN_depth_conv1(bl_in), inplace=True))
            bl_in = F.upsample(bl_in, scale_factor=2**(lvl-k_min), mode='bilinear', align_corners=True) #upsample by 2
            feature = bl_in + feature

        fpn_depth_conv = F.relu(self.FPN_depth_conv3(feature), inplace=True) #torch.Size([2, 64, 192, 336])
        fpn_depth_cls_score = F.upsample(self.FPN_depth_cls_score(fpn_depth_conv), scale_factor=2, mode='bilinear', align_corners=True) #B, 64, 384, 672
        return_dict['depth_cls_logits'] = fpn_depth_cls_score

        if not self.training:
            fpn_depth_cls_probs = F.softmax(fpn_depth_cls_score, dim=1)
            return_dict['depth_cls_probs'] = fpn_depth_cls_probs

        return return_dict


def depth_losses(depth_cls_score, roidb):
    """Add Depth on FPN specific losses."""
    # ------------------------------------------------------------------------
    batch_size, n_classes, depth_height, depth_width = depth_cls_score.size()
    assert(depth_height == cfg.MODEL.DEPTH_HEIGHT and depth_width == cfg.MODEL.DEPTH_WIDTH)
    flat_depth_size = cfg.MODEL.DEPTH_WIDTH * cfg.MODEL.DEPTH_HEIGHT
    n_samples = batch_size*flat_depth_size
    
    depths_int32 = blob_utils.zeros((n_samples), int32=True)

    for idx, entry in enumerate(roidb):
        start_idx = idx*flat_depth_size
        depths_int32[start_idx: start_idx + flat_depth_size] = np.reshape(entry['gt_depth'], flat_depth_size)
    # -------------------------------------------------------------------------
    device_id = depth_cls_score.get_device()
    depth_label = Variable(torch.from_numpy(depths_int32.astype('int64'))).cuda(device_id)
    
    depth_cls_score = depth_cls_score.view(n_samples, -1) #flat the 2d input
    depth_loss_cls = F.cross_entropy(depth_cls_score, depth_label) #averaging taken care of

    depth_cls_preds = depth_cls_score.max(dim=1)[1].type_as(depth_label)
    depth_accuracy_cls = (depth_cls_preds.eq(depth_label).float().mean(dim=0)).mean(dim=0)

    return depth_loss_cls*cfg.MODEL.WEIGHT_LOSS_DEPTH, depth_accuracy_cls

# ---------------------------------------------------------------------------------------------
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# -----------------------------------------------------------
class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """

        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss

def make_soft_labels(hard_labels, n_classes, std_dev=0.4):
    #assume hard_labels is a vector    
    n_samples = len(hard_labels)    
    class_ids = np.arange(0, n_classes, 1)
    soft_labels = np.tile(class_ids, [n_samples, 1]) #n_samples x class_ids

    soft_labels = (soft_labels.transpose() - hard_labels).transpose() #x - mean
    soft_labels = np.exp(-np.power(soft_labels, 2.) / (2*np.power(std_dev, 2)))

    row_sum = soft_labels.sum(axis=1)
    soft_labels = soft_labels/row_sum[:, np.newaxis]

    return soft_labels


def soft_depth_losses(depth_cls_score, roidb):
    """Add Soft Labels Depth on FPN specific losses."""
    # ------------------------------------------------------------------------
    batch_size, n_classes, depth_height, depth_width = depth_cls_score.size()
    
    assert(depth_height == cfg.MODEL.DEPTH_HEIGHT and depth_width == cfg.MODEL.DEPTH_WIDTH)
    flat_depth_size = cfg.MODEL.DEPTH_WIDTH * cfg.MODEL.DEPTH_HEIGHT
    
    depths_int32 = blob_utils.zeros((batch_size, flat_depth_size), int32=True)
    depths_soft_float32 = np.zeros((batch_size, flat_depth_size, n_classes)) #each pixels is a sample

    for idx, entry in enumerate(roidb):
        depths_int32[idx, :] = np.reshape(entry['gt_depth'], flat_depth_size)
        depths_soft_float32[idx, :, :] = make_soft_labels(np.reshape(entry['gt_depth'], flat_depth_size), n_classes)
    # -------------------------------------------------------------------------

    device_id = depth_cls_score.get_device()
    depth_label = Variable(torch.from_numpy(depths_int32.astype('int64'))).cuda(device_id) #batch_size x num_pixels
    depth_soft_label = Variable(torch.from_numpy(depths_soft_float32.astype('float32'))).cuda(device_id)
    
    depth_cls_score = depth_cls_score.view(batch_size, n_classes, -1)
    depth_cls_score = depth_cls_score.permute(0, 2, 1) #batch_size x num_pixels x n_classes
    # ----------------------------------
    log_likelihood = -F.log_softmax(depth_cls_score, dim=2) #batch_size x num_pixels x n_classes
    depth_loss_cls = torch.sum(torch.mul(log_likelihood, depth_soft_label))/(batch_size*flat_depth_size)
    # ----------------------------------

    depth_cls_preds = depth_cls_score.max(dim=2)[1].type_as(depth_label) #batch_size x num_pixels
    depth_accuracy_cls = (depth_cls_preds.eq(depth_label).float().mean(dim=0)).mean(dim=0)

    return depth_loss_cls*cfg.MODEL.WEIGHT_LOSS_DEPTH, depth_accuracy_cls

