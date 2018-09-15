cd ../../../..



# ################################################### VIRAT SCENE1 #######################################################################################3

# # --------------------------with attributes------------------------------

# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep08-21-34-54_cube3_step/ckpt/model_step15101.pth

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--vis \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# 						--test_data virat1_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\



# ################################################### VIRAT SCENE2 #######################################################################################3

# # --------------------------with attributes------------------------------

# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep08-21-34-54_cube3_step/ckpt/model_step15101.pth

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--vis \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\



# # ################################################### EPFL #######################################################################################3

# # # --------------------------with attributes------------------------------

# # ## --------evaluate with visualisation------
# # CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# # 						--test_data virat2_real_val \
# # 						--vis \
# # 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# # 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep08-21-34-54_cube3_step/ckpt/model_step15101.pth


# # ##------multi gpu evaluate------
# # CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# # 						--test_data epfl_real_val \
# # 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# # 						--vis \
# # 						--multi-gpu-testing \
# # 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\

# ################################################### UADETRAC #######################################################################################3

# # --------------------------with attributes------------------------------

##------single gpu visualize------
CUDA_VISIBLE_DEVICES=0, python tools/test_net.py --dataset car_coco \
						--test_data uadetrac1_real_val \
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--vis \
						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\


# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--vis \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\

# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset car_coco \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth\
