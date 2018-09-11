cd ../../../..
# ################################################### SCENE1, Syn #######################################################################################3
# ## --------evaluate with visualisation------
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset virat1 \
# 						--test_data virat1_real_val \
# 						--vis \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/rcnn_attr_101/Sep11-00-18-06_bheem_step/ckpt/model_step24401.pth\

##------multi gpu evaluate------
CUDA_VISIBLE_DEVICES=0,1,2 python tools/test_net.py --dataset virat1 \
						--test_data virat1_real_val \
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--multi-gpu-testing \
						--load_ckpt Outputs/virat1_syn/rcnn_attr_101/Sep11-00-18-06_bheem_step/ckpt/model_step24401.pth\