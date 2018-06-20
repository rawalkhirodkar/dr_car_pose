cd ../../../..
# ################################################### SCENE1, Syn #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# # # --------------------------without attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_mix/rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat1_mix/rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------with attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_mix/rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat1_mix/rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate MaskRCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------


# # # --------------------------without attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--vis \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_mix/mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat1_mix/mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------with attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--vis \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_mix/mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat1_mix/mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate MaskRCNN + Depth-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------


# # # --------------------------without attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--vis \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_mix/depth_mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat1_mix/depth_mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------with attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--vis \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_mix/depth_mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat1_mix/depth_mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \
