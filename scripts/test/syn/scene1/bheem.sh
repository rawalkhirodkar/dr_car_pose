#per GPU 2 images recommended to fill the whole model

cd ../../../..
# ################################################### SCENE1, Syn #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Test RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# # # --------------------------without attributes------------------------------

# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat1/\
# 						--output_dir results/virat1_syn/rcnn/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat1/\
# 						--output_dir results/virat1_syn/rcnn/real_visualise



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat1/\
# 						--output_dir results/virat1_syn/rcnn_attr/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat1/\
# 						--output_dir results/virat1_syn/rcnn_attr/real_visualise

# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Test MaskRCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------


# # --------------------------without attributes------------------------------

# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat1/\
# 						--output_dir results/virat1_syn/mask_rcnn/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat1/\
# 						--output_dir results/virat1_syn/mask_rcnn/real_visualise



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/mask_rcnn/mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/mask_rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat1/\
# 						--output_dir results/virat1_syn/mask_rcnn_attr/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat1/\
# 						--output_dir results/virat1_syn/mask_rcnn_attr/real_visualise

# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Test MaskRCNN + Depth-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/depth_mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat1/\
# 						--output_dir results/virat1_syn/depth_mask_rcnn/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/depth_mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat1/\
# 						--output_dir results/virat1_syn/depth_mask_rcnn/real_visualise



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/depth_mask_rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat1/\
# 						--output_dir results/virat1_syn/depth_mask_rcnn_attr/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/depth_mask_rcnn/depth_mask_rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/depth_mask_rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat1/\
# 						--output_dir results/virat1_syn/depth_mask_rcnn_attr/real_visualise

