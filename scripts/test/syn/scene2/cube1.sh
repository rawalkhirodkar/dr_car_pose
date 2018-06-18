
##########################SCENE1#########################################
##--------------------Test RCNN--------------------------

# # ----Synthetic Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-101-FPN_2x/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 							--image_dir data/test/virat1/\
# 							--output_dir rcnn_syn_virat1\


# # # ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x/May13-17-19-59_bheem_step/ckpt/model_step2999.pth\
# 							--image_dir data/test/real/virat1\
# 							--output_dir rcnn_real_virat1\


##--------------------Test MaskRCNN--------------------------

# ----Synthetic Visualise------
cd ..
CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
							--cfg configs/mask_rcnn/mask_rcnn_101.yaml\
							--load_ckpt Outputs/mask_rcnn_101/May14-13-47-00_klab-server1_step/ckpt/model_step1676.pth\
							--image_dir data/test/virat1/\
							--output_dir mask_rcnn_syn_virat1\

# # # ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/mask_rcnn/mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/mask_rcnn_101/May14-13-47-00_klab-server1_step/ckpt/model_step1676.pth\
# 							--image_dir data/test/real/virat1\
# 							--output_dir mask_rcnn_real_virat1\

# ##--------------------Test only Normal--------------------------
# cd ..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/normal/normal_101.yaml \
# 						--use_tfboard \
# 						--bs 15 \
# 						--nw 20


##--------------------Test only Depth--------------------------

# ###------------debugging ResNet 50------------------
# CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset virat1 \
# 						--cfg configs/e2e_mask_rcnn_R-50-FPN_1x.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--nw 20



##########################SCENE2#########################################

#-----------------1. Depth Only-------------------
# # ------resnet50-------
# CUDA_VISIBLE_DEVICES=0 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/depth/resnet50.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 20

# # ------resnet101-------
# CUDA_VISIBLE_DEVICES=0,1,2,4 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/depth/resnet101.yaml \
# 						--use_tfboard \
# 						--bs 24 \
# 						--nw 20

#-----------------2. Normal Only-------------------
# # ------resnet50-------
# CUDA_VISIBLE_DEVICES=0 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/normal/resnet50.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 20

# # ------resnet101-------
# CUDA_VISIBLE_DEVICES=0,1,2,4 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/normal/resnet101.yaml \
# 						--use_tfboard \
# 						--bs 18 \
# 						--nw 20



###################################################################



# -----------------------------------------------------------------------------------------------------


