
##########################SCENE1#########################################
##--------------------Test RCNN--------------------------

# # ----Synthetic Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-101-FPN_2x/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 							--image_dir data/test/virat1/\
# 							--output_dir results/rcnn_syn_virat1\


# # # ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/rcnn/rcnn_101.yaml\
# 							--load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x/May13-17-19-59_bheem_step/ckpt/model_step2999.pth\
# 							--image_dir data/test/real/virat1\
# 							--output_dir results/rcnn_real_virat1\


##--------------------Test MaskRCNN--------------------------

# # ----Synthetic Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/mask_rcnn/mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/mask_rcnn_101/May14-13-47-00_klab-server1_step/ckpt/model_step1676.pth\
# 							--image_dir data/test/virat1/\
# 							--output_dir results/mask_rcnn_syn_virat1\

# # # ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/mask_rcnn/mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/mask_rcnn_101/May14-13-47-00_klab-server1_step/ckpt/model_step1676.pth\
# 							--image_dir data/test/real/virat1\
# 							--output_dir results/mask_rcnn_real_virat1\

##--------------------Test MaskRCNN + Depth--------------------------

# # ----Synthetic Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/scene1/depth_mask_rcnn/scene1_depth_mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/scene1_depth_mask_rcnn_101/May15-01-12-58_bheem_step/ckpt/model_step10397.pth \
# 							--image_dir data/test/syn/virat1/\
# 							--output_dir results/depth_mask_rcnn_syn_virat1\

# # ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat1\
# 							--cfg configs/scene1/depth_mask_rcnn/scene1_depth_mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/scene1_depth_mask_rcnn_101/May15-01-12-58_bheem_step/ckpt/model_step10397.pth \
# 							--image_dir data/test/real/virat1\
# 							--output_dir results/depth_mask_rcnn_real_virat1\



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

cd ..
CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
							--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
							--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May28-18-55-16_bheem_step/ckpt/model_step12291.pth \
							--image_dir data/test/real/small_virat2/\
							--output_dir results/mask_rcnn_real_virat2\

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

# # ----Real Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat2\
# 							--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May16-21-08-47_bheem_step/ckpt/model_step5009.pth \
# 							--image_dir data/test/real/small_virat2\
# 							--output_dir results/depth_mask_rcnn_real_virat2\


###################################################################



# -----------------------------------------------------------------------------------------------------


