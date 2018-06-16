
##########################SCENE1#########################################


##########################SCENE2#########################################
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \

# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2_real_test \
# 						--model_type coco \
# 						--vis \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


# #normal coco test
# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset coco2017 \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


#normal coco test multi gpu
cd ../..
CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset coco2017 \
						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
						--multi-gpu-testing \
						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


# cd ../..
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--multi-gpu-testing \
# 						--vis \
# 						--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \


# -----------------------------------------------------------------------------------------------------


