export CUDA_VISIBLE_DEVICES=1,2

python -m torch.distributed.launch \
	--nproc_per_node 2 \
	--master_port 1230 \
	train.py --hyp data/hyp.scratch.rope3d_v2.yaml --weight /data/zhangqi/yolov7/rope_3d/train22/weights/best_220.pt \
	--data data/rope3d.yaml --epochs 100 --batch-size 64 --img 960 960 \
	--project rope_3d --name train --rect --noautoanchor \
	--cfg cfg/training/yolov7_5head_v2.yaml \