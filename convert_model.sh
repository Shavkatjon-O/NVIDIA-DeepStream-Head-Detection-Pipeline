#!/bin/bash

echo "Converting head_detection.pt to TensorRT engine..."

python3 << 'EOF'
from ultralytics import YOLO

model = YOLO('model/head_detection.pt')
model.export(format='engine', device=0, half=False, imgsz=640, batch=1)
print("Conversion complete!")
EOF
