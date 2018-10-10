


cd ../../../..




# -----------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python tools/render_infer_simple.py --dataset virat2_syn\
							--cfg configs/virat/rcnn/rcnn_attr_101.yaml \
							--load_ckpt Outputs/virat2_syn/rcnn_attr_101/Sep13-01-44-07_klab-server2_step/ckpt/model_step11558.pth\
							--image_dir data/test/real/small_virat2/\
							--output_dir results/virat2_syn/render



###################################################################



# -----------------------------------------------------------------------------------------------------


