



..\.venv\Scripts\python.exe ..\yolov5\train.py ^
    --data data.yaml ^
    --epochs 300 ^
    --weights yolov5l.pt ^
    --batch -1 ^
    --cache ram ^
    --device 0