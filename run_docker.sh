IMAGE_NAME="ese615" # Replace with your image name

rocker --nvidia --x11 \
    --volume ./src:/sim_ws/src \
    -- ${IMAGE_NAME}