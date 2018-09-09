#per GPU 2 images recommended to fill the whole model

cd ../../../..



# CUDA_VISIBLE_DEVICES=0,3 python tools/train_net_step.py --dataset virat2_syn \
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--iter_size 4 \
# 						--nw 0


# #---------------------- with pose-----------------------------------
# CUDA_VISIBLE_DEVICES=0,3 python tools/train_net_step.py --dataset virat2_syn \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--use_tfboard \
# 						--bs 4 \
# 						--iter_size 4 \
# 						--nw 0

CUDA_VISIBLE_DEVICES=2 python tools/train_net_step.py --dataset virat2_syn \
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--use_tfboard \
						--bs 4 \
						--iter_size 4 \
						--nw 0



###################################################Resume Training#######################################################################################3


# CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset virat2_mix \
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May30-12-18-44_bheem_step/ckpt/model_step3995.pth \
# 						--resume \
# 						--use_tfboard \
# 						--bs 2 \
						# --iter_size 4 \
# 						--nw 0