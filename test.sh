xhost +local:
docker run -it --rm \
--privileged \
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
camerasensor_mtk \
