
##########################SCENE1#########################################


##########################SCENE2#########################################










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


#######################################################################################################3
# #normal coco test
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 \
# 						--test_data coco_2017_val \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


#normal coco test multi gpu
cd ../..
CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset coco2017 \
						--test_data coco_2017_val \
						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
						--multi-gpu-testing \
						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \

# -----------------------------------------------------------------------------------------------------


