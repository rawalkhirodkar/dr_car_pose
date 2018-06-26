cd ../../..

##########################SCENE1#########################################

# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset coco \
# 							--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 							--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \
# 							--image_dir data/test/real/small_virat1/ \
# 							--output_dir results/coco_virat1\



##########################SCENE2#########################################

# CUDA_VISIBLE_DEVICES=1 python tools/infer_simple.py --dataset coco \
# 							--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 							--load_ckpt Outputs/coco/baseline_Jun24-14-58-56_bheem_step/ckpt/model_step60000.pth \
# 							--image_dir data/test/real/small_virat2/ \
# 							--output_dir results/coco_virat2\


# ### visualise
# CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --dataset coco2017 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--load_ckpt Outputs/coco/baseline_Jun24-14-58-56_bheem_step/ckpt/model_step60000.pth \


# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset coco2017 \
# 						--test_data virat2_real_val \
# 						--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/coco/baseline_Jun14-02-33-53_cube3_step/ckpt/model_step30999.pth \


###################################################################



# -----------------------------------------------------------------------------------------------------


