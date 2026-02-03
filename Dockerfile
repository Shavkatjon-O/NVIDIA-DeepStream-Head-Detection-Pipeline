FROM nvcr.io/nvidia/deepstream:8.0-gc-triton-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV DS_SDK_ROOT=/opt/nvidia/deepstream/deepstream
ENV LD_LIBRARY_PATH=${DS_SDK_ROOT}/lib:$LD_LIBRARY_PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-rtsp \
    libgstrtspserver-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY deepstream_pipeline_config.txt .

COPY config_infer_primary.txt .

COPY labels.txt .

COPY model/ ./model/

COPY libnvdsinfer_custom_impl_Yolo.so .

COPY convert_model.sh .

RUN chmod +x convert_model.sh

RUN mkdir -p /app/output_data

EXPOSE 8554

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gst-inspect-1.0 nvstreammux > /dev/null 2>&1 || exit 1

CMD ["deepstream-app", "-c", "deepstream_pipeline_config.txt"]
