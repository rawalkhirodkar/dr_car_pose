#per GPU 2 images recommended to fill the whole model

cd ../../
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




# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset car_coco \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step7103.pth \
# 						--image_dir data/kitti/images/\
# 						--output_dir results/kitti/car_coco/real_visualise


# # ------------------------------------------------------------------------------------------------
# # # #COCO
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset car_coco \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth \
# 						--image_dir data/wacv/virat1/\
# 						--output_dir results/wacv/coco/virat1

# # # # ------------------------------------------------------------------------------------------------
# # # ## DR scene1
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat1_syn/rcnn_attr_101/Sep13-01-43-56_klab-server2_step/ckpt/model_step12265.pth\
# 						--image_dir data/wacv/virat1/\
# 						--output_dir results/wacv/dr_virat1/virat1

# # ------------------------------------------------------------------------------------------------
# ## DR scene1
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat1_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep09-22-54-01_cube3_step/ckpt/model_step16994.pth\
# 						--image_dir data/wacv/virat1/\
# 						--output_dir results/wacv/ss_virat1/virat1

#####################################################################################################

# # # #COCO
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset car_coco \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth \
# 						--image_dir data/wacv/virat2/\
# 						--output_dir results/wacv/coco/virat2

# # # # # ------------------------------------------------------------------------------------------------
# # # ## DR scene2
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep13-01-44-07_klab-server2_step/ckpt/model_step11558.pth\
# 						--image_dir data/wacv/virat2/\
# 						--output_dir results/wacv/dr_virat2/virat2

# # # # # ------------------------------------------------------------------------------------------------
# # # ## SS scene2
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep09-22-54-01_cube3_step/ckpt/model_step16994.pth\
# 						--image_dir data/wacv/virat2/\
# 						--output_dir results/wacv/ss_virat2/virat2





# # # ## SS scene2
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_101/baseline/ckpt/model_step15877.pth\
# 						--image_dir data/wacv/virat2/\
# 						--output_dir results/wacv/ss_virat2/virat2


#####################################################################################################

# # # #COCO
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset car_coco \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth \
# 						--image_dir data/wacv/epfl/\
# 						--output_dir results/wacv/coco/epfl

# # # # # # ------------------------------------------------------------------------------------------------
# # # # ## epfl
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/epfl_syn/rcnn_attr_101/Sep13-01-44-52_klab-server2_step/ckpt/model_step12326.pth\
# 						--image_dir data/wacv/epfl/\
# 						--output_dir results/wacv/dr_epfl

# # # # ------------------------------------------------------------------------------------------------
# # # ## SS scene2
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/virat2_syn/rcnn_101/baseline_Jun19-18-12-32_klab-server2_step/ckpt/model_step15877.pth\
# 						--image_dir data/wacv/virat2/\
# 						--output_dir results/wacv/ss_virat2/virat2


#####################################################################################################

# # # #COCO
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset car_coco \
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/car_coco/rcnn_attr_101/Sep12-16-11-36_klab-server2_step/ckpt/model_step6000.pth \
# 						--image_dir data/wacv/uadetrac1/\
# 						--output_dir results/wacv/coco/uadetrac1

# # # # # ------------------------------------------------------------------------------------------------
# # # # ## epfl
# CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
# 						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
# 						--load_ckpt Outputs/uadetrac1_syn/rcnn_attr_101/Sep14-02-45-37_klab-server2_step/ckpt/model_step11104.pth\
# 						--image_dir data/wacv/uadetrac1/\
# 						--output_dir results/wacv/dr_uadetrac1

# # # # ------------------------------------------------------------------------------------------------
# # ## SS scene2
CUDA_VISIBLE_DEVICES=3 python tools/infer_simple.py --dataset virat2_syn\
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--load_ckpt Outputs/uadetrac1_syn/rcnn_attr_101/Sep15-00-00-27_klab-server2_step/ckpt/model_step16676.pth\
						--image_dir data/wacv/uadetrac1/\
						--output_dir results/wacv/ss_uadetrac1