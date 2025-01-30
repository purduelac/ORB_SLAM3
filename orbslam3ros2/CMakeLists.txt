cmake_minimum_required(VERSION 3.16)
project(orbslam3ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Find ROS 2 packages
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Pangolin REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)
link_directories(${OpenCV_LIBRARY_DIRS})

# ORB_SLAM3 Setup (Ensure the path is correct)
set(ORB_SLAM3_DIR /workspace/ORB_SLAM3)
set(ORB_SLAM3_LIB_DIR /workspace/ORB_SLAM3/lib)
set(ORB_SLAM3_INCLUDE_DIR /workspace/ORB_SLAM3/include)

# Include directories
include_directories(
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_INCLUDE_DIR}
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${OpenCV_INCLUDE_DIRS}
    ${rclcpp_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
)

add_definitions(-DPANGO_HAS_GL_COPY_IMAGE=0)

# Link directories (if needed, but it's better to link OpenCV through the proper `find_package` step)
link_directories(${ORB_SLAM3_LIB_DIR} /opt/ros/humble/lib)

# Set source files
set(SOURCES_MONO src/ros_mono_inertial.cc)
set(SOURCES_STEREO src/ros_stereo_inertial.cc)

add_executable(ros_mono_inertial ${SOURCES_MONO})

# Link dependencies using ament_target_dependencies for ROS 2 dependencies
ament_target_dependencies(ros_mono_inertial
    rclcpp
    sensor_msgs
    cv_bridge
)


# Link OpenCV and ORB_SLAM3
target_link_libraries(ros_mono_inertial
    ${OpenCV_LIBS}                
    ${ORB_SLAM3_LIB_DIR}/libORB_SLAM3.so
    ${Pangolin_LIBRARIES}         
    pthread
    boost_system
    boost_filesystem
)

target_link_options(ros_mono_inertial PRIVATE 
    "-Wl,-rpath,${ORB_SLAM3_LIB_DIR}"
    "-Wl,-rpath,${Pangolin_LIBRARY_DIRS}"    
)

# Create the executable for ros_stereo_inertial
add_executable(ros_stereo_inertial ${SOURCES_STEREO})

# Link dependencies for ros_stereo_inertial
ament_target_dependencies(ros_stereo_inertial
    rclcpp
    sensor_msgs
    cv_bridge
)

# Link OpenCV and ORB_SLAM3
target_link_libraries(ros_stereo_inertial
    ${OpenCV_LIBS}                # This links OpenCV
    ${ORB_SLAM3_LIB_DIR}/libORB_SLAM3.so
    ${Pangolin_LIBRARIES}         # Add Pangolin library linking
    pthread
    boost_system
    boost_filesystem
)

target_link_options(ros_stereo_inertial PRIVATE 
    "-Wl,-rpath,${ORB_SLAM3_LIB_DIR}"
    "-Wl,-rpath,${Pangolin_LIBRARY_DIRS}"    
)

# Install targets
install(TARGETS
    ros_mono_inertial
    ros_stereo_inertial
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
