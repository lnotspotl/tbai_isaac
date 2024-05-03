FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Setup env variables
SHELL ["/bin/bash", "-c"]
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Update system
RUN apt update && apt upgrade -y

# Install python3 and pip
RUN apt install python3 python3-pip -y

# Install torch
RUN pip3 --no-cache-dir install networkx==3.1  # This is because otherwise torch installs the newest networx version which only supports python >=3.10
RUN pip3 install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir numpy wandb tensorboard numpy==1.20.3

# Install ROS
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt install curl -y
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update
RUN apt install ros-noetic-ros-base -y

ENV HOME=/home/tbai
RUN mkdir -p ${HOME}
RUN echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && source ${HOME}/.bashrc

# Install isaacgym
COPY ./isaacgym ${HOME}/isaacgym
RUN pip3 install -e ${HOME}/isaacgym/python

# Download libtorch C++ API
RUN apt install wget -y
RUN cd ${HOME} && wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-static-with-deps-2.0.1%2Bcu118.zip
RUN apt install unzip -y
RUN cd ${HOME} && unzip libtorch-cxx11-abi-static-with-deps-2.0.1+cu118.zip
RUN cd ${HOME} && rm libtorch-cxx11-abi-static-with-deps-2.0.1+cu118.zip


# Install tbai_bindings TODO: Refactor this mess
RUN pip3 install --no-cache-dir catkin_tools
RUN echo "echo hello" >> ${HOME}/.bashrc
ENV EXTRA_TORCH_LIBS=${HOME}/libtorch/libtorch_python.so
RUN echo "export CMAKE_PREFIX_PATH=\$CMAKE_PREFIX_PATH:${HOME}/libtorch" >> ${HOME}/.bashrc && source ${HOME}/.bashrc
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${HOME}/libtorch/lib" >> ${HOME}/.bashrc && source ${HOME}/.bashrc
RUN mkdir -p ${HOME}/tbai_bindings/src && cd ${HOME}/tbai_bindings && catkin init && catkin config -DCMAKE_BUILD_TYPE=Release
RUN apt install -y libglpk-dev ros-noetic-pybind11-catkin libeigen3-dev ros-noetic-pinocchio git ros-noetic-interactive-markers ros-noetic-tf ros-noetic-urdf ros-noetic-kdl-parser ros-noetic-robot-state-publisher vim
COPY ./tbai_bindings ${HOME}/tbai_bindings/src/tbai_bindings
RUN pip3 install --no-cache-dir pybind11 
RUN apt install python3-catkin-tools python3-catkin -y
RUN source ${HOME}/.bashrc && cd ${HOME}/tbai_bindings && source /opt/ros/noetic/setup.bash && export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/root/libtorch && catkin build tbai_bindings 

# Source tbai_bindings
RUN echo "source ${HOME}/tbai_bindings/devel/setup.bash" >> ${HOME}/.bashrc && source ${HOME}/.bashrc
RUN echo "export HOME=${HOME}" >> ${HOME}/.bashrc && source ${HOME}/.bashrc

# Placeholders for singularity
RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

# FOLDERS for 

RUN mkdir -p ${HOME}/wormhole # persistent folder

# Set initial working directory
WORKDIR ${HOME}