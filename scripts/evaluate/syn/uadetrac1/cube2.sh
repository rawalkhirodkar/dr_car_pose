cd ../../../..
# ################################################### UADETRAC SCENE1, Syn #######################################################################################3
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


# ##------single gpu visualize------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset uadetrac1 \
# 						--test_data uadetrac1_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--vis \
# 						--load_ckpt Outputs/uadetrac1_syn/rcnn_attr_101/Sep14-02-45-37_klab-server2_step/ckpt/model_step11104.pth\

# #------single gpu visualize------
# CUDA_VISIBLE_DEVICES=2 python tools/test_net.py --dataset uadetrac1 \
# 						--test_data uadetrac1_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/uadetrac1_syn/rcnn_attr_101/Sep15-00-00-27_klab-server2_step/ckpt/model_step15000.pth\


#------single gpu visualize------
CUDA_VISIBLE_DEVICES=2 python tools/test_net.py --dataset uadetrac1 \
						--test_data uadetrac1_real_val \
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--load_ckpt Outputs/uadetrac1_syn/rcnn_attr_101/Sep14-02-45-37_klab-server2_step/ckpt/model_step11104.pth\



# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset uadetrac1 \
# 						--test_data uadetrac1_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/uadetrac1_syn/rcnn_attr_101/Sep14-02-45-37_klab-server2_step/ckpt/model_step11104.pth\



#################################### from Scratch #########################################


# ##------multi gpu evaluate------
# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_net.py --dataset virat2 \
# 						--test_data virat2_real_val \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--multi-gpu-testing \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep14-02-48-47_klab-server2_step/ckpt/model_step15316.pth\




