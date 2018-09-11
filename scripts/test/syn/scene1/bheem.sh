#per GPU 2 images recommended to fill the whole model

cd ../../../..
# ################################################### SCENE1#######################################################################################3
CUDA_VISIBLE_DEVICES=0 python tools/infer_simple.py --dataset virat1_syn\
						--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
						--load_ckpt Outputs/virat1_syn/rcnn_attr_101/Sep11-00-18-06_bheem_step/ckpt/model_step24401.pth\
						--image_dir data/test/real/small_virat1/\
						--output_dir results/virat1_syn/rcnn_attr/real_visualise

