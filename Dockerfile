# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ros:humble

SHELL ["/bin/bash", "-c"]

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       nano \
                       vim \
                       wget \
                       python3-pip \
                       libeigen3-dev \
                       tmux \
                       ros-humble-rviz2

# Add NVIDIA CUDA GPG key and network repo
RUN apt-get update && apt-get install -y gnupg2 curl && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update

                   
RUN apt-get install -y \
    cuda-toolkit-11-8 \
    libcudnn8 libcudnn8-dev \
    libnvinfer8 libnvinfer-dev \
    libnvonnxparsers8 \
    libnvparsers8 \
    libnvinfer-plugin8 \
    python3-libnvinfer \
    && rm -rf /var/lib/apt/lists/*


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
    rosdep install -i --from-path src --rosdistro humble -y && \
    colcon build

RUN apt update && apt install -y ros-humble-tf-transformations
RUN apt-get update && apt-get install -y python3-matplotlib \
                                                libprotoc-dev \
                                                protobuf-compiler

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


RUN pip install numpy==1.24

WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/bash"]
