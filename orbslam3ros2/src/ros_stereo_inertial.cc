#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core/core.hpp>

#include "System.h"
#include "ImuTypes.h"
#include "lac_interfaces/msg/stereo_imu.hpp"  // Include the StereoIMU message header
#include "lac_interfaces/msg/image_pair.hpp"

using namespace std;

class ImageGrabber : public rclcpp::Node
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, bool bRect, bool bClahe) 
        : Node("image_grabber"), mpSLAM(pSLAM), do_rectify(bRect), mbClahe(bClahe)
    {
        mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    }

    // Callback function to process both images and IMU data synchronously
    void SyncStereoImu(const lac_interfaces::msg::StereoIMU::SharedPtr stereo_imu_msg)
    {
        // Convert images
        cv::Mat imLeft = GetImage(std::make_shared<sensor_msgs::msg::Image>(stereo_imu_msg->img_pair.left));
        cv::Mat imRight = GetImage(std::make_shared<sensor_msgs::msg::Image>(stereo_imu_msg->img_pair.right));

        std::cout << " have img " << std::endl;

        // IMU Data processing
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        if (stereo_imu_msg->imu.linear_acceleration.x != 0.0 || stereo_imu_msg->imu.angular_velocity.x != 0.0) {
            std::cout << " have imu pose " << std::endl;
            cv::Point3f acc(stereo_imu_msg->imu.linear_acceleration.x, 
                            stereo_imu_msg->imu.linear_acceleration.y, 
                            stereo_imu_msg->imu.linear_acceleration.z);
            cv::Point3f gyr(stereo_imu_msg->imu.angular_velocity.x, 
                            stereo_imu_msg->imu.angular_velocity.y, 
                            stereo_imu_msg->imu.angular_velocity.z);
            vImuMeas.emplace_back(acc, gyr, rclcpp::Time(stereo_imu_msg->img_pair.left.header.stamp).seconds());
        }

        // Optionally apply CLAHE
        if(mbClahe) {
            mClahe->apply(imLeft, imLeft);
            mClahe->apply(imRight, imRight);
        }

        // Optionally apply Rectification
        if(do_rectify) {
            cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
            cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
        }

        // Update the SLAM system with the stereo pair and IMU data
        double tImLeft = rclcpp::Time(stereo_imu_msg->img_pair.left.header.stamp).seconds();
        mpSLAM->TrackStereo(imLeft, imRight, tImLeft, vImuMeas);
    }

    cv::Mat GetImage(const sensor_msgs::msg::Image::SharedPtr &img_msg)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
        return cv_ptr->image;
    }

    ORB_SLAM3::System* mpSLAM;
    bool do_rectify;
    bool mbClahe;
    cv::Mat M1l, M2l, M1r, M2r;
    cv::Ptr<cv::CLAHE> mClahe;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    auto node = rclcpp::Node::make_shared("orb_slam3");
    node->declare_parameter("vocabulary", argv[1]);
    node->declare_parameter("settings", argv[2]);
    node->declare_parameter("do_rectify", false);
    node->declare_parameter("do_equalize", false);

    string voc_file, settings_file;
    bool do_rectify, do_equalize;

    try {
        voc_file = node->get_parameter("vocabulary").as_string();
        settings_file = node->get_parameter("settings").as_string();
        do_rectify = node->get_parameter("do_rectify").as_bool();
        do_equalize = node->get_parameter("do_equalize").as_bool();
    } catch (...) {
        RCLCPP_ERROR(node->get_logger(), "Failed to get parameters");
        return 1;
    }

    ORB_SLAM3::System SLAM(
        voc_file, 
        settings_file, 
        ORB_SLAM3::System::IMU_STEREO, 
        true
    );

    auto image_grabber = make_shared<ImageGrabber>(&SLAM, do_rectify, do_equalize);

    // Setup subscriber for StereoIMU messages
    auto sub_stereo_imu = image_grabber->create_subscription<lac_interfaces::msg::StereoIMU>(
        "/stereo_imu", 
        1000, 
        bind(&ImageGrabber::SyncStereoImu, image_grabber, placeholders::_1)
    );

    rclcpp::spin(node);  // Process messages synchronously

    rclcpp::shutdown();
    return 0;
}