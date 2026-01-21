xhost +local:
docker run -it --rm \
--privileged \
--net=host \
--name camerasensor_dev_env \
\
--device /dev/apusys \
--device /dev/apusys_reviser \
--device /dev/mali0 \
--device=/dev/video5:/dev/video5 \
--device=/dev/video6:/dev/video6 \
--device=/dev/media1:/dev/media1 \
\
-v /usr/bin/ncc-tflite:/usr/bin/ncc-tflite \
-v /usr/bin/neuronrt:/usr/bin/neuronrt \
-v /usr/lib:/host/usr/lib \
-v /lib/firmware:/lib/firmware \
-v /etc:/etc \
\
-e LD_LIBRARY_PATH=/host/usr/lib:/usr/lib:$LD_LIBRARY_PATH \
\
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/home/mpc/.Xauthority \
\
-v $(pwd):/home/mpc/workspace \
\
camerasensor_dev
    # /bin/bash -c "export LD_LIBRARY_PATH=/usr/lib:\$LD_LIBRARY_PATH && exec bash"
    # -e LD_LIBRARY_PATH=/home/mpc/micromamba/envs/camerasensor/lib:/usr/lib/aarch64-linux-gnu:/usr/lib:/host/usr/lib \
    # ncc-tflite --arch=?
