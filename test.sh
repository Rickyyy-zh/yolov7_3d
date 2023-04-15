export CUDA_VISIBLE_DEVICES=4

python test.py --weight rope_3d/3d_train5/weights/best.pt \
	--data data/rope3d.yaml --batch-size 4 --img-size 960 \
	--project rope_3d --name validation \
	--task val --no-trace --save-txt --conf-thres 0.1 --iou-thres 0.6