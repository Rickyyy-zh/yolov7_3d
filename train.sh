export CUDA_VISIBLE_DEVICES=1

python train.py --hyp data/hyp.scratch.rope3d.yaml --weight none \
	--cfg cfg/training/yolov7_5head.yaml \
	--data data/rope3d.yaml --epochs 300 --batch-size 16 --img 800 800 --noautoanchor \
	--project rope_3d --name train \