#per GPU 2 images recommended to fill the whole model

cd ../../../..
# ################################################### SCENE2, Syn #######################################################################################3
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------Test RCNN-----------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------------



# # --------------------------with attributes------------------------------
# ### ----Synthetic Visualise------
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep08-21-34-54_cube3_step/ckpt/model_step15101.pth\
# 						--image_dir data/test/syn/virat2/\
# 						--output_dir results/virat2_syn/rcnn_attr/syn_visualise


CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset epfl_syn\
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep11-18-34-05_bheem_step/ckpt/model_step15000.pth\
						--image_dir data/test/real/small_epfl/\
						--output_dir results/epfl_syn/rcnn_attr/real_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_syn/rcnn_attr/real_visualise

