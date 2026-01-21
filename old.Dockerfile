FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    git build-essential cmake pkg-config \
    libgstreamer1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools \
    libglib2.0-dev libusb-1.0-0-dev libudev-dev libv4l-dev \
    libxml2-dev libjson-glib-dev \
    gobject-introspection libgirepository1.0-dev \
    lsb-release sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN apt-get update && apt-get install -y wget git sudo build-essential cmake pkg-config
# Install tzdata
RUN apt-get update \
    && apt-get install -y tzdata \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata
# Build tiscamera
RUN cd /tmp \
    && git clone https://github.com/TheImagingSource/tiscamera.git \
    && cd tiscamera \
    && git checkout v-tiscamera-1.1.1 \
    && sed -i 's/tags\[\] = {"id", "val", NULL}/tags\[\] = {NULL}/g' \
    libs/tcam-property/src/gst/meta/gstmetatcamstatistics.cpp \
    && sudo ./scripts/dependency-manager install -y \
    && mkdir build \
    && cd build \
    # Sphinx would produce error, so disabled TCAM_BUILD_DOCUMENTATION
    && cmake -DTCAM_BUILD_DOCUMENTATION=OFF .. \
    && make -j$(nproc) \
    && sudo make install \
    && rm -rf /tmp/tiscamera


WORKDIR /opt/bench
COPY benchmark.py .
COPY yolov8s_int8_mtk-mdla3.0.dla .
COPY yolov8s_int8_mtk.tflite .
COPY input.bin .

CMD ["/bin/bash"]