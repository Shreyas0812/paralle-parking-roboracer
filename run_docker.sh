IMAGE_NAME="ese615" # Replace with your image name

rocker --nvidia --x11 \
    --name ${IMAGE_NAME} \
    --volume ./src:/sim_ws/src \
    -- ${IMAGE_NAME}