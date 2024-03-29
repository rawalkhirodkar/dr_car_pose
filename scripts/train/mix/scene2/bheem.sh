#per GPU 2 images recommended to fill the whole model

cd ../../../..
################################################### SCENE2, Mix#######################################################################################3
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------Train RCNN-----------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
# 						--iter_size 4 \
# 						--nw 4


# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
						# --iter_size 4 \
# 						--nw 4


# --------------------------with attributes------------------------------

# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
						# --iter_size 4 \
# 						--nw 0


# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
						# --iter_size 4 \
# 						--nw 4

# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------Train MaskRCNN-----------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2_mix \
						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
						--use_tfboard \
						--bs 2 \
						--iter_size 4 \
						--nw 4


# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
						# --iter_size 4 \
# 						--nw 4

# --------------------------with attributes------------------------------

# CUDA_VISIBLE_DEVICES=2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
						# --iter_size 4 \
# 						--nw 0

# CUDA_VISIBLE_DEVICES=0,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
						# --iter_size 4 \
# 						--nw 4

# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
##------------------------------------------------------Train MaskRCNN + Depth-----------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
						# --iter_size 4 \
# 						--nw 0


# CUDA_VISIBLE_DEVICES=0,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
						# --iter_size 4 \
# 						--nw 4

# --------------------------with attributes------------------------------

# CUDA_VISIBLE_DEVICES=2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
						# --iter_size 4 \
# 						--nw 0

# CUDA_VISIBLE_DEVICES=0,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
						# --iter_size 4 \
# 						--nw 4


###################################################Resume Training#######################################################################################3


# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May30-12-18-44_bheem_step/ckpt/model_step3995.pth \
# 						--resume \
# 						--use_tfboard \
# 						--bs 2 \
						# --iter_size 4 \
# 						--nw 0