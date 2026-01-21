xhost +local:
docker run -it --rm \
--privileged \
--name camerasensor_dev_env \
--group-add video \
--group-add render \
--device /dev/apusys \
--device /dev/apusys_reviser \
--device /dev/mali0 \
--device=/dev/video5:/dev/video5 \
--device=/dev/video6:/dev/video6 \
--device=/dev/media1:/dev/media1 \
-v /lib/firmware:/lib/firmware \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/home/mpc/.Xauthority \
-v /home/ubuntu/Desktop/mdla_benchmark_test:/home/mpc/mdla_benchmark_test \
-v /home/ubuntu/Desktop/camerasensor:/home/mpc/camerasensor \
camerasensor_mtk \
