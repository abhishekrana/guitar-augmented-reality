REPO_NAME="aug-rel"
REPO_VERSION="0.1"


CONTAINER_NAME="${REPO_NAME//[\/]/_}"_"cnt_pred"
PWD_NAME=` pwd | awk -F'/' '{print $NF}' `

echo $CONTAINER_NAME
echo $PWD_NAME

rm -f .bash_history .viminfo

# Build docker image
# docker build -t "$REPO_NAME":"$REPO_VERSION" .

# Stop and remove container
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

# Create container
xhost +
# sudo apt-get install x11-xserver-utils
docker run -it \
	--runtime=nvidia \
	-e DISPLAY=$DISPLAY \
	--ipc=host \
	--device=/dev/video0:/dev/video0 \
	-v /home:/home \
	-v /media/abhishek/OS1/Abhishek/gar2/:/media/abhishek/OS1/Abhishek/gar2 \
	-v /mnt:/mnt \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ~/.config/pudb:/root/.config/pudb \
	--name $CONTAINER_NAME \
	"$REPO_NAME":"$REPO_VERSION" \
	bash -c "cd `pwd` && bash"

