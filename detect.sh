export CUDA_VISIBLE_DEVICES=4

python test.py --weight rope_3d/3d_train5/weights/best.pt \
	--source data/Baidu_Rope3d_mini/validation --img-size 960 \
	--project rope_3d --name detect \
	--no-trace --save-txt --conf-thres 0.25 --iou-thres 0.45