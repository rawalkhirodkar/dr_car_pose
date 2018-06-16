##########################SCENE1#########################################
##--------------------Train RCNN--------------------------
# 13 images per GPU about 9GB memory!

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/rcnn/scene1_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 20


##--------------------Train MaskRCNN--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/mask_rcnn/scene1_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
# 						--nw 10


##--------------------Train MaskRCNN + Depth--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/depth_mask_rcnn/scene1_depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 12 \
# 						--nw 10


# ##--------------------Train only Normal--------------------------
# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/normal/normal_101.yaml \
# 						--use_tfboard \
# 						--bs 15 \
# 						--nw 20


##--------------------Train only Depth--------------------------

# ###------------debugging ResNet 50------------------
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/e2e_mask_rcnn_R-50-FPN_1x.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 20


# -------------------------------------------------------------------------------------



###################################################SCENE2#######################################################################################3


##--------------------Train RCNN--------------------------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/rcnn/scene2_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4


# cd ..
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4


##--------------------Train MaskRCNN--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2_syn \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4


# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4

# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 12 \
# 						--nw 4

##--------------------Train MaskRCNN + Depth--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_syn \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4

cd ../..
CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_mix \
						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
						--use_tfboard \
						--bs 4 \
						--nw 4

##--------------------Train MaskRCNN + Normal--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_syn \
# 						--cfg configs/virat/normal_mask_rcnn/normal_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4

##--------------------Train MaskRCNN + Depth + Normal--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_syn \
# 						--cfg configs/virat/depth_normal_mask_rcnn/depth_normal_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 4
###################################################Resume Training#######################################################################################3


# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May30-12-18-44_bheem_step/ckpt/model_step3995.pth \
# 						--resume \
# 						--use_tfboard \
# 						--bs 2 \
# 						--nw 0