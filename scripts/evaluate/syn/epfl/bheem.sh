cd ../../../..
# ################################################### SCENE2, Syn #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# # # --------------------------without attributes------------------------------

### --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \

###------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=1,2 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/rcnn_101/Jun17-23-01-31_bheem_step/ckpt/model_step39999.pth \


# # --------------------------with attributes------------------------------

# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep08-21-34-54_cube3_step/ckpt/model_step15101.pth

##------multi gpu evaluate------
CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset epfl \
						--test_data epfl_real_val \
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--vis \
						--multi-gpu-testing \
						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep11-18-34-05_bheem_step/ckpt/model_step15000.pth\

