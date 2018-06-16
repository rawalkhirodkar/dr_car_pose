
###--------------------------Best Only Normal ResNet 50----------------------------------------------------

# # ----------------------Synthetic Visualise--------------------------------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/e2e_mask_rcnn_R-101-FPN_2x.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-101-FPN_2x/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 							--image_dir data/test/virat1/\
# 							--output_dir infer_outputs\


# # # ----------------------Real Visualise--------------------------------
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/e2e_mask_rcnn_R-50-FPN_1x.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x/May13-17-19-59_bheem_step/ckpt/model_step2999.pth\
# 							--image_dir data/test/real/virat1\
# 							--output_dir real_infer_outputs\




####--------------------------Best Only Depth ResNet 101----------------------------------------------------

# # ----------------------Synthetic Visualise--------------------------------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/e2e_mask_rcnn_R-101-FPN_2x.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-101-FPN_2x/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 							--image_dir data/test/virat1/\
# 							--output_dir infer_outputs\


# # # ----------------------Real Visualise--------------------------------
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/e2e_mask_rcnn_R-101-FPN_2x.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-101-FPN_2x/May12-21-18-52_bheem_step/ckpt/model_step16649.pth\
# 							--image_dir data/test/real/virat1\
# 							--output_dir real_infer_outputs\


# -------------------------------------------------------------------------------------


#--------------------------ResNet 50----------------------------------------------------

# # ----------------------Synthetic Visualise--------------------------------
# CUDA_VISIBLE_DEVICES=1 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/e2e_mask_rcnn_R-50-FPN_1x.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x/May05-17-54-00_bheem_step/ckpt/model_step328.pth\
# 							--image_dir data/virat1/images/train2018\
# 							--output_dir infer_outputs\


# # # ----------------------Real Visualise--------------------------------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/e2e_mask_rcnn_R-50-FPN_1x.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x/May10-22-36-57_bheem_step/ckpt/model_step1861.pth\
# 							--image_dir data/real/virat1\
# 							--output_dir real_infer_outputs\


# -------------------------------------------------------------------------------------







##---------------------Scene1-----------------------------

#-----------------1. Depth Only-------------------
# # ------resnet50-------
# CUDA_VISIBLE_DEVICES=0 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/depth/resnet50.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 20

# # ------resnet101-------
# CUDA_VISIBLE_DEVICES=0,1,2,4 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/depth/resnet101.yaml \
# 						--use_tfboard \
# 						--bs 18 \
# 						--nw 20

#-----------------2. Normal Only-------------------
# # # ------syn resnet101-------
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 						--cfg ../configs/normal/normal_101.yaml \
# 						--load_ckpt Outputs/resnet101/May13-19-39-34_bheem_step/ckpt/model_step320.pth\
# 						--image_dir data/test/virat1/\
# 						--output_dir normal_syn\
# # # ------real resnet101-------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 						--cfg configs/normal/normal_101.yaml \
# 						--load_ckpt Outputs/resnet101/May13-19-39-34_bheem_step/ckpt/model_step320.pth\
# 						--image_dir data/test/real/virat1\
# 						--output_dir normal_real\


###################################################################



##---------------------Scene2-----------------------------

#-----------------1. Depth Only-------------------
# # ------resnet50-------
# CUDA_VISIBLE_DEVICES=0 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/depth/resnet50.yaml \
# 						--use_tfboard \
# 						--bs 6 \
# 						--nw 20

# # ------resnet101-------
# CUDA_VISIBLE_DEVICES=0,1,2,4 python ../tools/train_net_step.py --dataset virat2 \
# 						--cfg ../configs/depth/resnet101.yaml \
# 						--use_tfboard \
# 						--bs 18 \
# 						--nw 20


cd ..
CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
						--cfg configs/normal/normal_101.yaml \
						--load_ckpt Outputs/resnet101/May13-19-39-34_bheem_step/ckpt/model_step320.pth\
						--image_dir data/test/real/virat1\
						--output_dir normal_real\

#-----------------2. Normal Only-------------------
# # # ------syn resnet101-------
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 						--cfg ../configs/normal/normal_101.yaml \
# 						--load_ckpt Outputs/resnet101/May13-19-39-34_bheem_step/ckpt/model_step320.pth\
# 						--image_dir data/test/virat1/\
# 						--output_dir normal_syn\
# # # ------real resnet101-------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 						--cfg configs/normal/normal_101.yaml \
# 						--load_ckpt Outputs/resnet101/May13-19-39-34_bheem_step/ckpt/model_step320.pth\
# 						--image_dir data/test/real/virat1\
# 						--output_dir normal_real\


###################################################################