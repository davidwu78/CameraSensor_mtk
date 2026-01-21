xhost +local:
docker run -it --rm \
--privileged \
--device /dev/apusys \
--device /dev/apusys_reviser \
--device /dev/mali0 \
--device=/dev/video5:/dev/video5 \
--device=/dev/video6:/dev/video6 \
--device=/dev/media1:/dev/media1 \
-e LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH \
-e DISPLAY=$DISPLAY \
-v $HOME/.Xauthority:/root/.Xauthority \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /usr/bin/ncc-tflite:/usr/bin/ncc-tflite \
-v /usr/bin/neuronrt:/usr/bin/neuronrt \
-v /usr/lib:/usr/lib \
-v /etc:/etc \
-v /lib/firmware:/lib/firmware \
-v /home/ubuntu/Desktop/mdla_benchmark_test:/home/mpc/mdla_benchmark_test \
camerasensor_mtk \
/bin/bash -c "export LD_LIBRARY_PATH=/usr/lib:\$LD_LIBRARY_PATH && exec bash"
