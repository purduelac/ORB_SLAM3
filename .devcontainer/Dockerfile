# FROM ros:humble
FROM osrf/ros:humble-desktop-full

RUN echo "keyboard-configuration keyboard-configuration/layoutcode string us" | debconf-set-selections

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
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
    xauth \
    xvfb \
    x11-apps \
    x11-xserver-utils \
    xorg \
    qtbase5-dev \
    qtchooser \
    qttools5-dev-tools \
    libxcb-xinerama0 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-util1 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-randr0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    x11-apps \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Qt libraries and X11 dependencies
RUN apt-get update && apt-get install -y \
    libxcb1 libx11-xcb1 libgl1-mesa-glx \
    qt5-qmake qtbase5-dev qtchooser qtbase5-dev-tools \
    libqt5x11extras5 libqt5core5a libqt5gui5

RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

WORKDIR /workspace
RUN git clone https://github.com/stevenlovegrove/Pangolin.git --branch v0.8
# RUN git clone https://github.com/purduelac/LunarAutonomyChallenge.git
WORKDIR /workspace/Pangolin
RUN mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF .. && cmake --build . -j$(nproc) && make install

WORKDIR /workspace
RUN mkdir ORB_SLAM3
COPY ../ /workspace/ORB_SLAM3

WORKDIR /workspace/ORB_SLAM3
# RUN chmod +x build.sh && ./build.sh
    
RUN pip3 install \
    catkin_pkg \
    torch \
    ultralytics \
    open3d
RUN pip install pybind11>=2.12

# Copy X11 authentication file
# COPY /tmp/.Xauthority /root/.Xauthority
ENV DISPLAY=:1

# ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

WORKDIR /workspace

# GUI support
# # RUN Xvfb :2 -screen 0 1024x768x24 &
# RUN export DISPLAY=:2

CMD ["bash"]