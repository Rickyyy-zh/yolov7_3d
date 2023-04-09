export CUDA_VISIBLE_DEVICES=4

python train.py --hyp data/hyp.scratch.rope3d_v2.yaml --weight none \
	--cfg cfg/training/yolov7_5head_v2.yaml \
	--data data/rope3d.yaml --epochs 300 --batch-size 16 --img 960 960 --noautoanchor \
	--project rope_3d --name 3d_train \
	--resume rope_3d/3d_train5/weights/last.pt