#per GPU 2 images recommended to fill the whole model

cd ../../../..
# ################################################### SCENE2, Syn #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Test RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------

# # # --------------------------without attributes------------------------------

# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_syn/rcnn/syn_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_syn/rcnn/real_visualise



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep08-21-34-54_cube3_step/ckpt/model_step15101.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_syn/rcnn_attr/syn_visualise


CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep09-22-54-01_cube3_step/ckpt/model_step16994.pth\
						--image_dir data/test/real/small_virat2/\
						--output_dir results/virat2_syn/rcnn_attr/real_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_syn/rcnn_attr/real_visualise

