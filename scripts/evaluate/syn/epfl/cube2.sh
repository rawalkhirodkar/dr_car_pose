cd ../../../..
# ################################################### SCENE2, Syn #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Evaluate RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# # --------------------------with attributes------------------------------

# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep08-21-34-54_cube3_step/ckpt/model_step15101.pth

CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --dataset epfl \
						--test_data epfl_real_val \
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep13-01-44-52_klab-server2_step/ckpt/model_step12326.pth\


###from scratch
# CUDA_VISIBLE_DEVICES=1 python tools/test_net.py --dataset epfl \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep14-02-50-07_klab-server2_step/ckpt/model_step14509.pth\



# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset epfl \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--vis \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep13-01-44-52_klab-server2_step/ckpt/model_step12326.pth\


# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,2,3 python tools/test_net.py --dataset epfl \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep13-01-44-52_klab-server2_step/ckpt/model_step12326.pth\



###################################### from Scratch ###########################################################################

##------single gpu visualize ------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset epfl \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--vis \
# 						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep14-02-50-07_klab-server2_step/ckpt/model_step14509.pth\



# ##------multi gpu evaluate ------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset epfl \
# 						--test_data epfl_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep14-02-50-07_klab-server2_step/ckpt/model_step14509.pth\
