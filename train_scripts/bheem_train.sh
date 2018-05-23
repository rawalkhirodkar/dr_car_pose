##########################SCENE1#########################################
##--------------------Train RCNN--------------------------
# 13 images per GPU about 9GB memory!

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/rcnn/scene1_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 20

# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/rcnn/scene1_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 39 \
# 						--nw 20


# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat1_real \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 30 \
# 						--nw 10

##--------------------Train MaskRCNN--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/mask_rcnn/scene1_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
# 						--nw 10

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/mask_rcnn/scene1_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 12 \
# 						--nw 10 

##--------------------Train MaskRCNN + Depth--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/depth_mask_rcnn/scene1_depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 12 \
# 						--nw 10

# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/depth_mask_rcnn/scene1_depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 18 \
# 						--nw 20 


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
# 						--bs 6 \
# 						--nw 20

# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/rcnn/scene2_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 39 \
# 						--nw 20


# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat2_syn \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 24 \
# 						--nw 10

cd ..
CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat2_mix \
						--cfg configs/virat/rcnn/rcnn_101.yaml \
						--use_tfboard \
						--bs 9 \
						--nw 10

# cd ..
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset virat2_real \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 8 \
# 						--nw 0

# cd ..
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 2 \
# 						--nw 0


# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 18 \
# 						--nw 10


# python tools/train_net_step.py --dataset coco2017 --cfg configs/e2e_mask_rcnn_R-50-FPN_1x.yaml --use_tfboard --bs 9

##--------------------Train MaskRCNN--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/mask_rcnn/scene2_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 10

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/mask_rcnn/scene2_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 12 \
# 						--nw 10 

##--------------------Train MaskRCNN + Depth--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 0

# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 18 \
# 						--nw 20 

# # ### resume training
# cd ..
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May15-19-43-52_cube3_step/ckpt/model_step2663.pth \
# 						--resume \
# 						--use_tfboard \
# 						--bs 2 \
# 						--nw 10 

# cd ..
# CUDA_VISIBLE_DEVICES=2,1,0 python tools/train_net_step.py --dataset virat2 \
# 						--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May15-19-43-52_cube3_step/ckpt/model_step2663.pth \
# 						--resume \
# 						--use_tfboard \
# 						--bs 15 \
# 						--nw 10 

###################################################Resume Training#######################################################################################3

# cd ..
# CUDA_VISIBLE_DEVICES=1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/scene1/mask_rcnn/scene1_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/scene1_mask_rcnn_101/May15-19-17-46_bheem_step/ckpt/model_step8060.pth \
# 						--resume \
# 						--use_tfboard \
# 						--bs 12 \
# 						--nw 10 