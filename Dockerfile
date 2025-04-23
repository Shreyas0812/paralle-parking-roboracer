# ROS2 Humble + F1TENTH Gym + NVIDIA CUDA 11.8

FROM ros:humble

SHELL ["/bin/bash", "-c"]

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       nano \
                       vim \
                       wget \
                       python3-pip3 \
                       libeigen3-dev \
                       tmux \
                       ros-humble-rviz2

# Add NVIDIA CUDA GPG key and network repo
RUN apt-get update && apt-get install -y gnupg2 curl && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update

                   
# Install CUDA 12.1 + cuDNN 9
RUN apt-get install -y \
    cuda-toolkit-12-1 \
    cuda-compiler-12-1 \
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    libnccl2 libnccl-dev \
    libnvinfer8 libnvinfer-dev \
    libnvonnxparsers8 \
    libnvparsers8 \
    libnvinfer-plugin8 \
    python3-libnvinfer && \
    rm -rf /var/lib/apt/lists/*

# Set environment paths for CUDA 12.1
ENV PATH="/usr/local/cuda-12.1/bin:${PATH}"


RUN apt-get -y dist-upgrade
RUN pip3 install transforms3d

# f1tenth gym
# Clone the repository and checkout a specific commit, branch, or tag
COPY src/f1tenth_gym_ros/f1tenth_gym /f1tenth_gym
RUN cd /f1tenth_gym && \
    pip3 install -e .


# ros2 gym bridge
RUN mkdir -p sim_ws/src/f1tenth_gym_ros
COPY . /sim_ws/src/f1tenth_gym_ros
RUN source /opt/ros/humble/setup.bash && \
    cd sim_ws/ && \
    apt-get update --fix-missing && \
    rosdep install -i --from-path src --rosdistro humble -y

RUN apt update && apt install -y ros-humble-tf-transformations
RUN apt-get update && apt-get install -y python3-matplotlib \
                                                libprotoc-dev \
                                                protobuf-compiler

# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# RUN pip3 install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.7+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl \
#     && pip3 install jax==0.4.7
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade "jax[cuda12]" && pip3 install flax


# Create JAX cache directory and ensure it's writable
RUN mkdir -p /cache/jax && chmod -R 777 /cache/jax

RUN pip3 install numpy==1.25
RUN pip3 install matplotlib

WORKDIR '/sim_ws'
RUN source /opt/ros/humble/setup.bash && \
    colcon build && \
    source install/setup.bash && \
    echo "source /sim_ws/install/setup.bash" >> ~/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/bin/bash"]
