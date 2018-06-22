cd ../../../..
# ################################################### SCENE2, Mix #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# # # --------------------------without attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

##------multi gpu evaluate------
CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset virat2 \
						--test_data virat2_real_val \
						--cfg configs/virat/rcnn/rcnn_101.yaml \
						--multi-gpu-testing \
						--load_ckpt Outputs/virat2_mix/rcnn_101/baseline_Jun20-02-19-45_bheem_step/ckpt/model_step15999.pth \


# # --------------------------with attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_mix/rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate MaskRCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------


# # # --------------------------without attributes------------------------------

# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_101/baseline_Jun19-22-56-55_bheem_step/ckpt/model_step15800.pth \

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_101/baseline_Jun19-22-56-55_bheem_step/ckpt/model_step15800.pth \


# # --------------------------with attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate MaskRCNN + Depth-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------


# # # --------------------------without attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------with attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \
