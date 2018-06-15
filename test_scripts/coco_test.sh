
##########################SCENE1#########################################

cd ..
CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset coco \
							--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
							--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \
							--image_dir data/test/real/small_virat1/ \
							--output_dir results/coco_virat1\



##########################SCENE2#########################################

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset coco \
# 							--cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
# 							--load_ckpt Outputs/coco/Jun14-02-33-53_cube3_step/ckpt/model_step35999.pth \
# 							--image_dir data/test/real/small_virat2/ \
# 							--output_dir results/coco_virat2\


###################################################################



# -----------------------------------------------------------------------------------------------------


