import importlib
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils

import modeling.attribute_heads as attribute_heads
import modeling.depth_heads as depth_heads
import modeling.normal_heads as normal_heads

logger = logging.getLogger(__name__)



def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)
        # ------------------------------------------------------------------------------------
        #Depth prediction Network
        if cfg.DEPTH.IS_ON:
            self.DepthNet = depth_heads.depth_outputs(self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        #Normal prediction Network
        if cfg.NORMAL.IS_ON:
            self.NormalNet = normal_heads.normal_outputs(self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)
        # ------------------------------------------------------------------------------------

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # -----------------------------------------------------------------------------
        # Attribute Branch
        if cfg.MODEL.ATTRIBUTE_ON:
            self.AttributeNet = attribute_heads.attribute_outputs(self.Box_Head.dim_out)
        # -----------------------------------------------------------------------------

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert self.Mask_Head.res5.state_dict() == self.Box_Head.res5.state_dict()
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert self.Keypoint_Head.res5.state_dict() == self.Box_Head.res5.state_dict()

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def _prep_return_dict(self, return_dict):
        # pytorch0.4 bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
            return_dict['losses'][k] = v.unsqueeze(0)
        for k, v in return_dict['metrics'].items():
            return_dict['metrics'][k] = v.unsqueeze(0)
        return return_dict

    def forward(self, data, im_info, roidb=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, **rpn_kwargs)

    def _splice_rpn_kwargs(self, rpn_kwargs, idx_tensor):
        rpn_kwargs_ret = {}
        for kw in rpn_kwargs:
            rpn_kwargs_ret[kw] = rpn_kwargs[kw][idx_tensor]
        return rpn_kwargs_ret


    def _merge_dict(self, real_dict, syn_dict, real_wt=0.5, syn_wt=0.5):
        total_wt = float(real_wt + syn_wt)
        real_wt = real_wt*1.0/total_wt
        syn_wt = syn_wt*1.0/total_wt

        return_dict = {}
        return_dict['losses'] = {}
        return_dict['metrics'] = {}

        if(len(real_dict) == 0):
            return_dict = syn_dict

        elif(len(syn_dict) == 0):
            return_dict = real_dict  
        
        else:
            for loss_kw in real_dict['losses']:
                return_dict['losses'][loss_kw] = real_wt*real_dict['losses'][loss_kw] + syn_wt*syn_dict['losses'][loss_kw] 

            for metrics_kw in real_dict['metrics']:
                return_dict['metrics'][metrics_kw] = real_wt*real_dict['metrics'][metrics_kw] + syn_wt*syn_dict['metrics'][metrics_kw] 

        return self._prep_return_dict(return_dict)
 
    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        im_data = data
        device_id = im_data.get_device()            
        # -------------------------------------------------------------------------------------------------------------------
        # In case the batch contains any real images
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))
            real_idx = []
            syn_idx = []
            for idx, entry in enumerate(roidb):
                if(entry['gt_is_real'] == 1):
                    real_idx.append(idx)
                else:
                    syn_idx.append(idx)

            real_return_dict = {}
            syn_return_dict = {}
            if(len(real_idx) > 0):
                real_idx_tensor = torch.tensor(real_idx)
                real_im_data = data[real_idx_tensor]
                real_im_info = im_info[real_idx]
                real_roidb = [roidb[idx] for idx in real_idx]
                real_rpn_kwargs = self._splice_rpn_kwargs(rpn_kwargs, real_idx_tensor)
                real_return_dict = self._real_train_forward(real_im_data, real_im_info, real_roidb, **real_rpn_kwargs)

            if(len(syn_idx) > 0):
                syn_idx_tensor = torch.tensor(syn_idx)
                syn_im_data = data[syn_idx_tensor]
                syn_im_info = im_info[syn_idx]
                syn_roidb = [roidb[idx] for idx in syn_idx]
                syn_rpn_kwargs = self._splice_rpn_kwargs(rpn_kwargs, syn_idx_tensor)
                syn_return_dict = self._syn_train_forward(syn_im_data, syn_im_info, syn_roidb, **syn_rpn_kwargs)

            return_dict = self._merge_dict(real_return_dict, syn_return_dict, real_wt=len(real_idx), syn_wt=len(syn_idx))
        # inference
        else:
            return_dict = self._inference_forward(data, im_info, roidb, **rpn_kwargs)

        return return_dict

 # ---------------------------------------------------------------------------------------------

    # called when training
    def _syn_train_forward(self, im_data, im_info, roidb=None, **rpn_kwargs):
        device_id = im_data.get_device()
        blob_conv = self.Conv_Body(im_data) #list of len equal to pyramid level, each containing the level data
        # DEPTH Branch
        if cfg.DEPTH.IS_ON:
            depth_ret = self.DepthNet(blob_conv, im_info, roidb)

        # NORMAL Branch
        if cfg.NORMAL.IS_ON:
            normal_ret = self.NormalNet(blob_conv, im_info, roidb)
        # -----------------------------------------------------------------------------------------------------------------        
        return_dict = {}  # A dict to collect return variables
        return_dict['losses'] = {}
        return_dict['metrics'] = {}

        # -----------------------------------------------------------------------------------------------------------------        
        #Loss computation for Depth
        if cfg.DEPTH.IS_ON:
            if cfg.DEPTH.SOFT_LABEL_ON:
                depth_loss_cls, depth_accuracy_cls = depth_heads.soft_depth_losses(depth_ret['depth_cls_logits'], roidb)
            else:
                depth_loss_cls, depth_accuracy_cls = depth_heads.depth_losses(depth_ret['depth_cls_logits'], roidb)
            return_dict['losses']['depth_loss_cls'] = depth_loss_cls
            return_dict['metrics']['depth_accuracy_cls'] = depth_accuracy_cls

        #Loss computation for Normal
        if cfg.NORMAL.IS_ON:
            if cfg.NORMAL.SOFT_LABEL_ON:
                normal_loss_cls, normal_accuracy_cls = normal_heads.soft_normal_losses(normal_ret['normal_cls_logits'], roidb)
            else:
                normal_loss_cls, normal_accuracy_cls = normal_heads.normal_losses(normal_ret['normal_cls_logits'], roidb)
            return_dict['losses']['normal_loss_cls'] = normal_loss_cls
            return_dict['metrics']['normal_accuracy_cls'] = normal_accuracy_cls
        # ------------------------------------------------------------------------------------------------------------------
        rpn_ret = self.RPN(blob_conv, im_info, roidb) # can ignore here
       
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        box_feat = self.Box_Head(blob_conv, rpn_ret)
        cls_score, bbox_pred = self.Box_Outs(box_feat) #fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)

        if cfg.MODEL.ATTRIBUTE_ON:
            rotation_cls_score = self.AttributeNet(box_feat) #fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)

        # ------------------------------------------------------------------------------------------------------------------
        # rpn loss
        rpn_kwargs.update(dict(
            (k, rpn_ret[k]) for k in rpn_ret.keys()
            if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
        ))
        loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
        if cfg.FPN.FPN_ON:
            for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
        else:
            return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
            return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

        # --------------------------------------------------------------------------
        # bbox loss
        loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
            cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
            rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])

        return_dict['losses']['loss_cls'] = loss_cls
        return_dict['losses']['loss_bbox'] = loss_bbox
        return_dict['metrics']['accuracy_cls'] = accuracy_cls

        if cfg.MODEL.ATTRIBUTE_ON:
            # attribute loss
            rotation_loss_cls, rotation_accuracy_cls = attribute_heads.attribute_losses(
                cls_score,
                rotation_cls_score,
                rpn_ret['labels_int32'],
                rpn_ret['rotation_labels_int32'])

            return_dict['losses']['rotation_loss_cls'] = rotation_loss_cls
            
            return_dict['metrics']['rotation_accuracy_cls'] = rotation_accuracy_cls
        # --------------------------------------------------------------------------

        if cfg.MODEL.MASK_ON:
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                           roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
            else:
                mask_feat = self.Mask_Head(blob_conv, rpn_ret)
            mask_pred = self.Mask_Outs(mask_feat)
            # return_dict['mask_pred'] = mask_pred
            # mask loss
            loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
            # handle nan loss
            is_nan = int(loss_mask != loss_mask) #1 if nan
            if(is_nan == 1):
                print("Mask NaN logged!")
                loss_mask = loss_mask*0
            return_dict['losses']['loss_mask'] = loss_mask

        return return_dict
# ---------------------------------------------------------------------------------------------
#Only called while training
    def _real_train_forward(self, im_data, im_info, roidb, **rpn_kwargs):
        device_id = im_data.get_device()
        return_dict = {}  # A dict to collect return variables
        return_dict['losses'] = {}
        return_dict['metrics'] = {}

        # --------------------------------------------------------------------------------------------------
        blob_conv = self.Conv_Body(im_data) #list of len equal to pyramid level, each containing the level data
        rpn_ret = self.RPN(blob_conv, im_info, roidb) 

        if cfg.FPN.FPN_ON:
            blob_conv = blob_conv[-self.num_roi_levels:]
        box_feat = self.Box_Head(blob_conv, rpn_ret)
        cls_score, bbox_pred = self.Box_Outs(box_feat) #fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)  

        # --------------------------------------------------------------------------------------------
        # rpn loss
        rpn_kwargs.update(dict(
            (k, rpn_ret[k]) for k in rpn_ret.keys()
            if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
        ))
        loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
        if cfg.FPN.FPN_ON:
            for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
        else:
            return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
            return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

        # bbox loss
        loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
            cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
            rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])

        return_dict['losses']['loss_cls'] = loss_cls
        return_dict['losses']['loss_bbox'] = loss_bbox
        return_dict['metrics']['accuracy_cls'] = accuracy_cls

        # --------------------------------------------------------------------------------------------------
        # as we are using coco, we activate the mask loss
        if cfg.MODEL.MASK_ON:
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                           roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
            else:
                mask_feat = self.Mask_Head(blob_conv, rpn_ret)
            mask_pred = self.Mask_Outs(mask_feat)
            # return_dict['mask_pred'] = mask_pred
            # mask loss
            loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
            # handle nan loss
            is_nan = int(loss_mask != loss_mask) #1 if nan
            if(is_nan == 1):
                print("Mask NaN logged!")
                loss_mask = loss_mask*0
            return_dict['losses']['loss_mask'] = loss_mask
        # --------------------------------------------------------------------------------------------------

        # zero some losses, uncomment if needed
        # return_dict['losses']['loss_cls'] = torch.tensor(0.0).cuda(device_id)
        # return_dict['metrics']['accuracy_cls'] = torch.tensor(0.0).cuda(device_id)

        # # ------------------------------------------------------------
        #fake losses        
        # -----------------------------------
        if cfg.DEPTH.IS_ON:
            return_dict['losses']['depth_loss_cls'] = torch.tensor(0.0).cuda(device_id)
            return_dict['metrics']['depth_accuracy_cls'] = torch.tensor(0.0).cuda(device_id)

        if cfg.NORMAL.IS_ON:
            return_dict['losses']['normal_loss_cls'] = torch.tensor(0.0).cuda(device_id)
            return_dict['metrics']['normal_accuracy_cls'] = torch.tensor(0.0).cuda(device_id)
        
        if cfg.MODEL.ATTRIBUTE_ON:
            return_dict['losses']['rotation_loss_cls'] = torch.tensor(0.0).cuda(device_id)
            
            return_dict['metrics']['rotation_accuracy_cls'] = torch.tensor(0.0).cuda(device_id)
        # # ------------------------------------------------------------

        return return_dict
# ---------------------------------------------------------------------------------------------
    def _inference_forward(self, im_data, im_info, roidb=None, **rpn_kwargs):
        device_id = im_data.get_device()        
        return_dict = {}
        blob_conv = self.Conv_Body(im_data) #list of len equal to pyramid level, each containing the level data

        # -------------------------------------------------------
        # Note the depth and normal takes the entire FPN
        if cfg.DEPTH.IS_ON:
            depth_ret = self.DepthNet(blob_conv, im_info, roidb)
            return_dict['depth_cls_score'] = depth_ret['depth_cls_probs'] #this will be after softmax

        if cfg.NORMAL.IS_ON:
            normal_ret = self.NormalNet(blob_conv, im_info, roidb)
            return_dict['normal_cls_score'] = normal_ret['normal_cls_probs'] #this will be after softmax
        # --------------------------------
        rpn_ret = self.RPN(blob_conv, im_info, roidb) # can ignore here
       
        # --------------------------------
        if cfg.FPN.FPN_ON:
            blob_conv = blob_conv[-self.num_roi_levels:]

        # -----------------------------------        
        box_feat = self.Box_Head(blob_conv, rpn_ret)
        cls_score, bbox_pred = self.Box_Outs(box_feat) #fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)

        # -----------------------------------
        # make the return dict, all after softmax is applied
        # mask branch is taken care of separately
        return_dict['rois'] = rpn_ret['rois']
        return_dict['cls_score'] = cls_score
        return_dict['bbox_pred'] = bbox_pred
        return_dict['blob_conv'] = blob_conv
        # -------------------------------------------------------
        if cfg.MODEL.ATTRIBUTE_ON:
            rotation_cls_score = self.AttributeNet(box_feat) #fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)    
            return_dict['rotation_cls_score'] = rotation_cls_score

        return return_dict

# ---------------------------------------------------------------------------------------------

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        if not self.training:
            mask_feat = self.Mask_Head(blob_conv, rpn_blob)
            mask_pred = self.Mask_Outs(mask_feat)
            return mask_pred
        else:
            raise ValueError('You should call this function only on inference.'
                             'Set the network in inference mode by net.eval().')


    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        if not self.training:
            kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
            kps_pred = self.Keypoint_Outs(kps_feat)
            return kps_pred
        else:
            raise ValueError('You should call this function only on inference.'
                             'Set the network in inference mode by net.eval().')

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
