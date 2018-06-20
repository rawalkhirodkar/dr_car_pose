#per GPU 2 images recommended to fill the whole model

cd ../../../..
# ################################################### SCENE2, Mix #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Test RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# # # --------------------------without attributes------------------------------

# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_mix/rcnn/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_mix/rcnn/real_visualise



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_mix/rcnn_attr/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_mix/rcnn_attr/real_visualise

# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Test MaskRCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------


# # --------------------------without attributes------------------------------

# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_mix/mask_rcnn/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_mix/mask_rcnn/real_visualise



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/mask_rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_mix/mask_rcnn_attr/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_mix/mask_rcnn_attr/real_visualise

# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Train MaskRCNN + Depth-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_mix/depth_mask_rcnn/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_mix/depth_mask_rcnn/real_visualise



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_mix/depth_mask_rcnn_attr/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_mix\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_mix/depth_mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_mix/depth_mask_rcnn_attr/real_visualise

