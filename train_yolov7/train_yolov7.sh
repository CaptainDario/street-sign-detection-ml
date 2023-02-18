#! /bin/bash

echo 'train p5-models';
cp utils/loss_p5.py utils/loss.py; 
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 9527  train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --epochs 450 --batch-size 32 --data data/streetsign_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name tiny --hyp data/hyp.scratch.custom.yaml;
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 9527  train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --epochs 450 --batch-size 32 --data data/streetsign_data.yaml --img 1280 1280 --cfg cfg/training/yolov7.yaml --weights '' --name normal --hyp data/hyp.scratch.custom.yaml;
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 9527  train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --epochs 450 --batch-size 24 --data data/streetsign_data.yaml --img 1280 1280 --cfg cfg/training/yolov7x.yaml --weights '' --name x --hyp data/hyp.scratch.custom.yaml; 
echo 'train-models'; 
rm utils/loss.py;
cp utils/loss_p6.py utils/loss.py;
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 9527  train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --epochs 450 --batch-size 32 --data data/streetsign_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name w6 --hyp data/hyp.scratch.custom.yaml; 
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 9527  train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --epochs 450 --batch-size 32 --data data/streetsign_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6.yaml --weights '' --name e6 --hyp data/hyp.scratch.custom.yaml; 
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 9527  train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --epochs 450 --batch-size 24 --data data/streetsign_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-d6.yaml --weights '' --name d6 --hyp data/hyp.scratch.custom.yaml; 
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 9527  train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --epochs 450 --batch-size 16 --data data/streetsign_data.yaml --img 1280 1280 --cfg cfg/training/yolov7-e6e.yaml --weights '' --name e6e --hyp data/hyp.scratch.custom.yaml; 
