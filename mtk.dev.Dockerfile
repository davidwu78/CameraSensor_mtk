FROM ubuntu:22.04

# 1. User setup
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=mpc
ARG MAMBA_ENV_NAME=camerasensor
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

SHELL ["/bin/bash", "-c"]

# 2. Locales, Timezone
RUN apt-get update \
    && apt-get install -y locales fonts-noto-cjk \
    && locale-gen en_US.UTF-8 \
    && dpkg-reconfigure locales \
    && update-locale LANG=en_US.UTF-8

ENV LC_ALL=en_US.UTF-8

RUN apt-get update \
    && apt-get install -y tzdata \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata

 # 3. sudo, video group

RUN apt-get update \
&& apt-get install -y sudo \
&& groupadd --gid $USER_GID $USERNAME \
&& useradd --uid $USER_UID --gid $USER_GID --groups sudo,video -m -s /bin/bash $USERNAME \
&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
&& chmod 0440 /etc/sudoers.d/$USERNAME 

# 4. GStreamer & Basic Tools
# 加入 software-properties-common 以支援 add-apt-repository
RUN apt-get update && apt-get install -y \
    git vim curl wget usbutils lsb-release pkg-config build-essential cmake \
    software-properties-common gnupg \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools \
    gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstrtspserver-1.0-dev \
    gstreamer1.0-plugins-base-apps \
    libglib2.0-dev libusb-1.0-0-dev libudev-dev libv4l-dev \
    libxml2-dev libjson-glib-dev gobject-introspection libgirepository1.0-dev \
    python3 python3-pip

# 5. Build tiscamera 
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

# 6. Others
RUN apt-get install -y freeglut3 freeglut3-dev mosquitto mosquitto-clients \
    libxml2 libxml2-dev libzip-dev qtbase5-dev \
    qtdeclarative5-dev libudev-dev python3-sphinx python3-pyqt5 libpulse-dev \
    libturbojpeg libcap-dev python3-dev libhdf5-dev \
    python3-pyqt5.qtsvg python3-pyqt5.qtwebkit python3-pyqt5.qtmultimedia \
    libqt5multimedia5-plugins \
    ffmpeg gfortran liblapack-dev liblapacke-dev \
    libopenblas-dev libeigen3-dev libtbb2 libtbb-dev v4l-utils \
    qv4l2 gpac x264 libx264-dev libgtk-3-dev python3-vtk9 python3-gi \
    python3-gi-cairo gir1.2-gtk-3.0 libffi-dev

# ==========================================
# 6.5 Genio PPA & NeuroPilot Libraries
# ==========================================
# 這裡加入 PPA 並安裝 Runtime Libraries (不含 Firmware/Drivers)
RUN add-apt-repository -y ppa:mediatek-genio/genio-public \
    && apt-add-repository -y ppa:mediatek-genio/genio-proposed \
    && apt-get update \
    && apt install -y mediatek-apusys-firmware-genio700 \
    && apt-get install -y mediatek-libneuron mediatek-neuron-utils mediatek-libneuron-dev

# switch user, setting environment
USER ${USERNAME}

# 7. Micromamba
RUN cd ~ \
    && curl -L micro.mamba.pm/install.sh -o install-micromamba.sh \
    && bash install-micromamba.sh \
    && rm install-micromamba.sh

ARG MAMBA_EXE=/home/${USERNAME}/.local/bin/micromamba
ARG MAMBA_ROOT_PREFIX=/home/${USERNAME}/micromamba
ENV PATH=/home/${USERNAME}/.local/bin:$PATH

RUN micromamba create -n ${MAMBA_ENV_NAME} python=3.10 -y \
    && echo "micromamba activate ${MAMBA_ENV_NAME}" >> ~/.bashrc \
    && eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && echo "/usr/lib/python3/dist-packages" > ${MAMBA_ROOT_PREFIX}/envs/${MAMBA_ENV_NAME}/lib/python3.10/site-packages/aptlibs.pth

# 8. uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && sudo apt-get install -y libgirepository1.0-dev libcairo2-dev \
    && pip install "setuptools<69" wheel \
    && pip install "playsound==1.2.2" \
    && pip install "setuptools==80.1.0" \
    && uv pip install --system --no-deps pyvista pyvistaqt qtpy scooby \
       torchvision==0.16.2 torchaudio psutil

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib

RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && echo "/usr/lib/python3.10/dist-packages" > "$(python -c "import site; print(site.getsitepackages()[0])")/_opencv_sys.pth" \
    && sudo mkdir -p /run/user/1000 \
    && sudo chown ${USERNAME}:${USERNAME} /run/user/1000 \
    && sudo chmod 700 /run/user/1000

RUN unset DEBIAN_FRONTEND

CMD ["/bin/bash"]
