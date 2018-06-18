
##########################SCENE1#########################################


##########################SCENE2#########################################




# ------------------------------Virat Syn MaskRCNN----------------------------------------------

# --------------------------------------------------
# ---------Syn with Rotation, Color-------------
# --------------------------------------------------

# ###Virat evaluate
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# ##Virat evaluate with visualization
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \



# ###Virat evaluate multi gpu
# cd ../..
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step17999.pth \

# --------------------------------------------------
# ---------Syn without Rotation, Color-------------
# --------------------------------------------------

# ###Virat evaluate
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/Jun18-15-18-09_bheem_step/ckpt/model_step4719.pth \


# ##Virat evaluate with visualization
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/Jun18-15-18-09_bheem_step/ckpt/model_step4719.pth \



###Virat evaluate multi gpu
cd ../..
CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
						--test_data virat2_real_val \
						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
						--multi-gpu-testing \
						--load_ckpt Outputs/virat2_syn/mask_rcnn_101/Jun18-15-18-09_bheem_step/ckpt/model_step4719.pth \




# ------------------------------COCO trained model----------------------------------------------

# ###Virat evaluate
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 \
# 						--test_data virat2_real_val \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


# ##Virat evaluate with visualization
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \



# ###Virat evaluate multi gpu
# cd ../..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset coco2017 \
# 						--test_data virat2_real_val \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


######################################COCO###################################################3
# #normal coco test
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 \
# 						--test_data coco_2017_val \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


# #normal coco test multi gpu
# cd ../..
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset coco2017 \
# 						--test_data coco_2017_val \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \

# -----------------------------------------------------------------------------------------------------


