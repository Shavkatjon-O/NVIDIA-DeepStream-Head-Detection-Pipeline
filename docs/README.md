# DeepStream Head Detection Pipeline Documentation

## Overview

This project implements a real-time head detection pipeline using NVIDIA DeepStream SDK. It processes video streams from an RTSP source, performs AI inference using a YOLO-based model to detect heads, and outputs the annotated video to a file.

The pipeline leverages:
- **DeepStream SDK** for optimized video processing and inference
- **YOLO (You Only Look Once)** model for object detection
- **TensorRT** for accelerated inference on GPU
- **ONNX** format for model compatibility

## Project Structure

```
deepstream_pipeline_test/
├── config_infer_primary.txt    # Primary inference engine configuration
├── convert_model.sh           # Script to convert PyTorch model to TensorRT
├── deepstream_pipeline_config.txt  # Main pipeline configuration
├── labels.txt                 # Class labels for detection
├── LICENSE                    # Project license
├── README.md                  # Root README
├── requirements.txt           # Python dependencies
├── docs/                      # Documentation folder
│   └── README.md             # This documentation
├── model/                     # Model files
│   ├── head_detection.onnx   # ONNX model for inference
│   ├── head_detection.pt     # Original PyTorch model
│   └── head_detection.engine # TensorRT engine (generated)
└── output_data/               # Output directory for processed videos
```

## Prerequisites

- **NVIDIA GPU** with CUDA support (compute capability 6.0 or higher)
- **Ubuntu 18.04/20.04/22.04** or similar Linux distribution
- **DeepStream SDK 6.0+** installed
- **Python 3.8+**
- **CUDA Toolkit 11.0+**
- **TensorRT 8.0+**

## Installation

1. **Clone or download** this repository to your local machine.

2. **Set up Python environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install DeepStream SDK** following NVIDIA's official documentation.

4. **Verify GPU and CUDA**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

## Model Preparation

The project includes a pre-trained YOLO model for head detection. If you need to convert or update the model:

1. **Convert PyTorch to TensorRT**:
   ```bash
   chmod +x convert_model.sh
   ./convert_model.sh
   ```
   This generates `model/head_detection.engine` for optimized inference.

2. **Model Details**:
   - Input size: 640x640
   - Classes: 1 (head)
   - Format: ONNX with TensorRT optimization

## Configuration

### Pipeline Configuration (`deepstream_pipeline_config.txt`)

Key settings:
- **Source**: RTSP stream from IP camera
- **Resolution**: 1280x720
- **Output**: MP4 video file
- **Display**: Tiled display with OSD overlays

### Inference Configuration (`config_infer_primary.txt`)

Key settings:
- **Model**: YOLO-based head detection
- **Thresholds**: NMS IoU 0.45, pre-cluster 0.25
- **Input dimensions**: 3x640x640
- **Batch size**: 1

### Customization

- **Change RTSP source**: Edit `uri` in `[source0]` section
- **Adjust detection sensitivity**: Modify `pre-cluster-threshold` and `nms-iou-threshold`
- **Change output resolution**: Update `width` and `height` in `[streammux]` and `[tiled-display]`

## Running the Pipeline

1. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Ensure model is converted** (if not already):
   ```bash
   ./convert_model.sh
   ```

3. **Run the pipeline**:
   ```bash
   deepstream-app -c deepstream_pipeline_config.txt
   ```

4. **Monitor output**:
   - Processed video saved to `output_data/video.mp4`
   - Performance metrics displayed in terminal
   - Real-time display (if enabled)

## Output

- **Video File**: Annotated MP4 with bounding boxes around detected heads
- **Performance Metrics**: FPS, latency, and throughput statistics
- **Logs**: Detailed inference and processing information

## Troubleshooting

### Common Issues

1. **CUDA/GPU Errors**:
   - Ensure GPU drivers are installed
   - Check CUDA compatibility with DeepStream version

2. **Model Loading Failures**:
   - Verify `head_detection.engine` exists
   - Run `./convert_model.sh` to regenerate engine

3. **RTSP Stream Issues**:
   - Check network connectivity
   - Verify camera credentials and URL
   - Test stream with VLC or similar player

4. **Performance Issues**:
   - Reduce batch size or resolution
   - Enable TensorRT optimization
   - Monitor GPU memory usage

### Debug Mode

Enable verbose logging by adding to pipeline config:
```
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1
```

## Performance Optimization

- Use TensorRT engine for maximum throughput
- Adjust batch size based on GPU memory
- Optimize input resolution vs. accuracy trade-off
- Enable hardware decoding/encoding

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

See LICENSE file for details.

## Support

For DeepStream-specific issues, refer to NVIDIA's documentation and forums.