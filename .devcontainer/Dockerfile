FROM ros:humble

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libgl1-mesa-dev \
    libglew-dev \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    libegl1-mesa-dev \
    libpng-dev \
    libjpeg-dev \
    libboost-dev \
    libboost-serialization-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    python3-pip \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

WORKDIR /workspace
RUN git clone https://github.com/stevenlovegrove/Pangolin.git --branch v0.8
WORKDIR /workspace/Pangolin
RUN mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF .. && cmake --build . -j$(nproc) && make install

WORKDIR /workspace
RUN mkdir ORB_SLAM3
COPY ../ /workspace/ORB_SLAM3

WORKDIR /workspace/ORB_SLAM3
# RUN chmod +x build.sh && ./build.sh
    
RUN pip3 install catkin_pkg

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

WORKDIR /workspace

CMD ["bash"]