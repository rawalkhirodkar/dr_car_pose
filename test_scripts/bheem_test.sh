
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

##--------------------Test MaskRCNN + Depth--------------------------

# # ----Synthetic Visualise------
# cd ..
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2\
# 							--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml\
# 							--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May15-19-43-52_cube3_step/ckpt/model_step2663.pth \
# 							--image_dir data/test/syn/virat2/\
# 							--output_dir results/depth_mask_rcnn_syn_virat2\

# ----Real Visualise------
cd ..
CUDA_VISIBLE_DEVICES=2 python tools/infer_simple.py --dataset virat2\
							--cfg configs/scene2/depth_mask_rcnn/scene2_depth_mask_rcnn_101.yaml\
							--load_ckpt Outputs/scene2_depth_mask_rcnn_101/May15-19-43-52_cube3_step/ckpt/model_step2663.pth \
							--image_dir data/test/real/virat2\
							--output_dir results/depth_mask_rcnn_real_virat2\



###################################################################



# -----------------------------------------------------------------------------------------------------


