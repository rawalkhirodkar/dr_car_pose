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


# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset car_coco\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step7103.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/car_coco/rcnn_attr/real_visualise


# ### ---- Real Visualise------
# CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/May12-21-18-52_bheem_step/ckpt/model_step15983.pth\
# 						--image_dir data/test/real/small_virat2/\
# 						--output_dir results/virat2_syn/rcnn_attr/real_visualise



#################################### from Scratch #########################################




CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset car_coco \
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step7103.pth \
						--image_dir data/kitti/images/\
						--output_dir results/kitti/car_coco/real_visualise