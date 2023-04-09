export CUDA_VISIBLE_DEVICES=1,4

python -m torch.distributed.launch \
	--nproc_per_node 2 \
	--master_port 1230 \
	train.py --hyp data/hyp.scratch.rope3d_v2.yaml --weight none \
	--data data/rope3d.yaml --epochs 300 --batch-size 32 --img 960 960 \
	--project rope_3d --name 3d_train --noautoanchor \
	--cfg cfg/training/yolov7_5head_v2.yaml \
	--resume rope_3d/3d_train5/weights/last.pt