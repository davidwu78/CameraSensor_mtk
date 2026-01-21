FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0
 
ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=nol
ARG MAMBA_ENV_NAME=camerasensor
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y locales fonts-noto-cjk\
    && locale-gen en_US.UTF-8 \
    && dpkg-reconfigure locales \
    && update-locale LANG=en_US.UTF-8

ENV LC_ALL=en_US.UTF-8

# Create the user
RUN apt-get update \
    && apt-get install -y sudo \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --groups sudo,video -m -s \
    /bin/bash $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install tzdata
RUN apt-get update \
    && apt-get install -y tzdata \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata

# Install common tools
RUN apt-get install -y git vim curl wget usbutils lsb-release pkg-config

# Install gstreamer
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools \
    gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstrtspserver-1.0-dev \
    gstreamer1.0-plugins-base-apps

# Build tiscamera
RUN cd /tmp \
    && git clone https://github.com/TheImagingSource/tiscamera.git \
    && cd tiscamera \
    && git checkout v-tiscamera-1.0.0 \
    && sed -i 's/tags\[\] = {"id", "val", NULL}/tags\[\] = {NULL}/g' \
    src/gstreamer-1.0/tcamsrc/gstmetatcamstatistics.cpp \
    && cd dependencies/ \
    && wget https://raw.githubusercontent.com/TheImagingSource/tiscamera/\
master/dependencies/ubuntu_2204.dep \
    && cd .. \
    && sudo ./scripts/dependency-manager install -y \
    && mkdir build \
    && cd build \
    # Sphinx would produce error, so disabled TCAM_BUILD_DOCUMENTATION
    && cmake -DTCAM_BUILD_DOCUMENTATION=OFF .. \
    && make -j$(nproc) \
    && sudo make install \
    && rm -rf /tmp/tiscamera

# environment packages
RUN apt-get install -y freeglut3 freeglut3-dev mosquitto mosquitto-clients \
    libxml2 libxml2-dev libusb-1.0-0-dev libzip-dev qtbase5-dev \
    qtdeclarative5-dev libudev-dev python3-sphinx python3-pyqt5 libpulse-dev \
    libturbojpeg libcap-dev build-essential python3-dev libhdf5-dev \
    lsb-release cmake pkg-config python3-pip python3-pyqt5.qtsvg \ 
    python3-pyqt5.qtwebkit python3-pyqt5.qtmultimedia libqt5multimedia5-plugins

# for opencv?
RUN apt-get install -y ffmpeg gfortran liblapack-dev liblapacke-dev \
    libopenblas-dev libeigen3-dev libtbb2 libtbb-dev libv4l-dev v4l-utils \
    qv4l2 gpac x264 libx264-dev libgtk-3-dev python3-vtk9 python3-gi \
    python3-gi-cairo gir1.2-gtk-3.0 libffi7

USER ${USERNAME}

# Install micromamba
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
    && echo "/usr/lib/python3/dist-packages" \
    > ${MAMBA_ROOT_PREFIX}/envs/${MAMBA_ENV_NAME}/lib/python3.10/\
site-packages/aptlibs.pth

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install pytorch from jetson zoo
RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && wget -P ~ https://developer.download.nvidia.cn/compute/redist/jp/v60dp/\
pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl \
    && mv ~/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl \
    ~/torch-2.2.0a0+6a974be-cp310-cp310-linux_aarch64.whl \
    && uv pip install --system \
    ~/torch-2.2.0a0+6a974be-cp310-cp310-linux_aarch64.whl \
    && rm ~/torch-2.2.0a0+6a974be-cp310-cp310-linux_aarch64.whl \
    && micromamba clean -y --all

ENV LD_LIBRARY_PATH=${MAMBA_ROOT_PREFIX}/envs/${MAMBA_ENV_NAME}/lib:\
${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}

# Install nvidia-pyindex packages
RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && mkdir -p /home/${USERNAME}/.config/pip \
    && touch /home/${USERNAME}/.config/pip/pip.conf \
    && grep -q '^\[global\]' /home/${USERNAME}/.config/pip/pip.conf || \
    echo '[global]' >> /home/${USERNAME}/.config/pip/pip.conf \
    && sed -i '/^index-url/d' /home/${USERNAME}/.config/pip/pip.conf \
    && echo 'index-url = https://pypi.ngc.nvidia.com' >> \
    /home/${USERNAME}/.config/pip/pip.conf \
    && sed -i '/^extra-index-url/d' /home/${USERNAME}/.config/pip/pip.conf \
    && echo 'extra-index-url = https://pypi.org/simple' >> \
    /home/${USERNAME}/.config/pip/pip.conf \
    && sed -i '/^trusted-host/d' /home/${USERNAME}/.config/pip/pip.conf \
    && echo 'trusted-host = pypi.ngc.nvidia.com' >> \
    /home/${USERNAME}/.config/pip/pip.conf

# Install packages in requirements_l4t.txt
COPY requirements_l4t.txt /home/${USERNAME}/requirements_l4t.txt
RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && sudo apt-get install -y libgirepository1.0-dev \
    && pip install "setuptools<69" \
    && pip install playsound \
    && pip install "setuptools==80.1.0" \
    && uv pip install --system --no-cache-dir -r ~/requirements_l4t.txt \
    && uv pip install --system --no-deps pyvista pyvistaqt qtpy scooby \
    torchvision==0.16.2 torchaudio psutil

ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && sudo mkdir -p /run/user/1000 \
    && sudo chown ${USERNAME}:${USERNAME} /run/user/1000 \
    && sudo chmod 700 /run/user/1000
    
RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && echo "/usr/lib/python3.10/dist-packages" > "$(python -c "import site; print(site.getsitepackages()[0])")/_opencv_sys.pth"

RUN eval "$(micromamba shell hook --shell bash)" \
    && micromamba activate ${MAMBA_ENV_NAME} \
    && uv pip install --system --no-cache-dir onnx

ENV PATH=/usr/src/tensorrt/bin:$PATH

RUN unset DEBIAN_FRONTEND