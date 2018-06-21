cd ../../../..
# ################################################### SCENE2, Syn #######################################################################################3
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
# 						--load_ckpt Outputs/virat2_syn/rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

##------multi gpu evaluate------
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset virat2 \
						--test_data virat2_real_val \
						--cfg configs/virat/rcnn/rcnn_101.yaml \
						--multi-gpu-testing \
						--load_ckpt Outputs/virat2_syn/rcnn_101/baseline_Jun19-18-12-32_klab-server2_step/ckpt/model_step9999.pth \


# # --------------------------with attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


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
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/baseline_Jun18-18-10-17_bheem_step/ckpt/model_step9999.pth \

# ###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/baseline_Jun18-18-10-17_bheem_step/ckpt/model_step9999.pth \


# # --------------------------with attributes------------------------------

# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_attr_101/baseline_Jun19-01-15-08_bheem_step/ckpt/model_step9999.pth \

###------multi gpu evaluate------
CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset virat2 \
						--test_data virat2_real_val \
						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
						--multi-gpu-testing \
						--load_ckpt Outputs/virat2_syn/mask_rcnn_attr_101/baseline_Jun19-01-15-08_bheem_step/ckpt/model_step29999.pth \


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
# 						--load_ckpt Outputs/virat2_syn/depth_mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/depth_mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------with attributes------------------------------

# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/depth_mask_rcnn_attr_101/baseline_Jun19-17-04-52_bheem_step/ckpt/model_step9999.pth \

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/depth_mask_rcnn_attr_101/baseline_Jun19-17-04-52_bheem_step/ckpt/model_step9999.pth \

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/depth_mask_rcnn_attr_101/baseline_Jun19-17-04-52_bheem_step/ckpt/model_step11528.pth \
