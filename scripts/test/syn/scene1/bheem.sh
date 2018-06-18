##########################SCENE2#########################################

# ##--------------------Test RCNN Real--------------------------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_real\
# 							--cfg configs/virat/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/virat2_real/rcnn_101/May20-22-11-49_bheem_step/ckpt/model_step469.pth \
# 							--image_dir data/test/real/small_virat2/\
# 							--output_dir results/rcnn_real_virat2\

##--------------------Test RCNN Mix on Real--------------------------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 							--cfg configs/virat/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/virat2_mix/rcnn_101/May21-20-37-37_bheem_step/ckpt/model_step2752.pth \
# 							--image_dir data/test/real/small_virat2/\
# 							--output_dir results/rcnn_real_virat2\

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 							--cfg configs/virat/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/virat2_syn/rcnn_101/May26-00-23-58_bheem_step/ckpt/model_step1167.pth \
# 							--image_dir data/test/real/small_virat2/\
# 							--output_dir results/rcnn_real_virat2\

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 							--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 							--load_ckpt Outputs/virat2_mix/mask_rcnn_101/Jun07-16-09-01_bheem_step/ckpt/model_step2499.pth \
# 							--image_dir data/test/real/small_virat2/\
# 							--output_dir results/mask_rcnn_real_virat2\


# #--------------------Test RCNN Mix on Syn--------------------------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 							--cfg configs/virat/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/virat2_mix/rcnn_101/May21-20-37-37_bheem_step/ckpt/model_step998.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/rcnn_syn_virat2\


# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 							--cfg configs/virat/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/virat2_mix/rcnn_101/May22-01-03-45_bheem_step/ckpt/model_step11654.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/rcnn_syn_virat2\

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 							--cfg configs/virat/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/virat2_syn/rcnn_101/May27-14-32-17_bheem_step/ckpt/model_step665.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/rcnn_syn_virat2\


# ##--------------------Test RCNN Syn--------------------------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_real\
# 							--cfg configs/virat/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/virat2_syn/rcnn_101/May19-16-14-58_bheem_step/ckpt/model_step311.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/rcnn_syn_virat2\


# ##--------------------Test Mask RCNN Syn--------------------------

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 							--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 							--load_ckpt Outputs/virat2_syn/mask_rcnn_101/May27-20-42-18_bheem_step/ckpt/model_step491.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/mask_rcnn_syn_virat2\

# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 							--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 							--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May28-18-55-16_bheem_step/ckpt/model_step12291.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/mask_rcnn_mix_virat2\


# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 							--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 							--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May31-23-47-32_bheem_step/ckpt/model_step12320.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/mask_rcnn_syn_virat2\

##--------------------Test MaskRCNN + Depth--------------------------

# # ----Synthetic Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2\
# 							--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May16-21-08-47_bheem_step/ckpt/model_step5009.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/depth_mask_rcnn_syn_virat2\


# ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat2\
# 							--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May15-19-43-52_cube3_step/ckpt/model_step2663.pth \
# 							--image_dir data/test/real/virat2\
# 							--output_dir results/depth_mask_rcnn_real_virat2\

cd ..
CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
							--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
							--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_101/Jun08-02-41-12_bheem_step/ckpt/model_step28999.pth \
							--image_dir data/test/real/small_virat2\
							--output_dir results/depth_mask_rcnn_real_virat2_logs\

# # ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat2\
# 							--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May16-21-08-47_bheem_step/ckpt/model_step5009.pth \
# 							--image_dir data/test/real/small_virat2\
# 							--output_dir results/depth_mask_rcnn_real_virat2\


###################################################################



# -----------------------------------------------------------------------------------------------------


