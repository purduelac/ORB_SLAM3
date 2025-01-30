#include <GL/glew.h>
#include <GL/gl.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core/core.hpp>

#include "System.h"
#include "ImuTypes.h"

using namespace std;

class ImuGrabber : public rclcpp::Node
{
public:
    ImuGrabber() : Node("imu_grabber") {}
    
    void GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg)
    {
        lock_guard<mutex> lock(mBufMutex);
        imuBuf.push(imu_msg);
    }

    queue<sensor_msgs::msg::Imu::ConstSharedPtr> imuBuf;
    mutex mBufMutex;
};

class ImageGrabber : public rclcpp::Node
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, shared_ptr<ImuGrabber> pImuGb, bool bRect, bool bClahe) 
        : Node("image_grabber"), mpSLAM(pSLAM), mpImuGb(pImuGb), do_rectify(bRect), mbClahe(bClahe)
    {
        mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    }

    void GrabImageLeft(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        lock_guard<mutex> lock(mBufMutexLeft);
        if (!imgLeftBuf.empty())
            imgLeftBuf.pop();
        imgLeftBuf.push(msg);
    }

    void GrabImageRight(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        lock_guard<mutex> lock(mBufMutexRight);
        if (!imgRightBuf.empty())
            imgRightBuf.pop();
        imgRightBuf.push(msg);
    }

    cv::Mat GetImage(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
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

    void SyncWithImu()
    {
        const double maxTimeDiff = 0.01;
        while(rclcpp::ok()) {
            cv::Mat imLeft, imRight;
            double tImLeft = 0, tImRight = 0;
            
            if (!imgLeftBuf.empty() && !imgRightBuf.empty() && !mpImuGb->imuBuf.empty()) {
                tImLeft = rclcpp::Time(imgLeftBuf.front()->header.stamp).seconds();
                tImRight = rclcpp::Time(imgRightBuf.front()->header.stamp).seconds();

                // Synchronization logic remains similar
                // ... (keep existing synchronization code)

                vector<ORB_SLAM3::IMU::Point> vImuMeas;
                {
                    lock_guard<mutex> lock(mpImuGb->mBufMutex);
                    while(!mpImuGb->imuBuf.empty() && 
                          rclcpp::Time(mpImuGb->imuBuf.front()->header.stamp).seconds() <= tImLeft)
                    {
                        auto imu_msg = mpImuGb->imuBuf.front();
                        cv::Point3f acc(imu_msg->linear_acceleration.x, 
                                      imu_msg->linear_acceleration.y, 
                                      imu_msg->linear_acceleration.z);
                        cv::Point3f gyr(imu_msg->angular_velocity.x, 
                                      imu_msg->angular_velocity.y, 
                                      imu_msg->angular_velocity.z);
                        vImuMeas.emplace_back(acc, gyr, 
                                            rclcpp::Time(imu_msg->header.stamp).seconds());
                        mpImuGb->imuBuf.pop();
                    }
                }

                // Image processing and tracking
                if(mbClahe) {
                    mClahe->apply(imLeft, imLeft);
                    mClahe->apply(imRight, imRight);
                }

                if(do_rectify) {
                    cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
                    cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
                }

                mpSLAM->TrackStereo(imLeft, imRight, tImLeft, vImuMeas);
                this_thread::sleep_for(chrono::milliseconds(1));
            }
        }
    }

    ORB_SLAM3::System* mpSLAM;
    shared_ptr<ImuGrabber> mpImuGb;
    bool do_rectify;
    bool mbClahe;
    cv::Mat M1l, M2l, M1r, M2r;
    cv::Ptr<cv::CLAHE> mClahe;

private:
    queue<sensor_msgs::msg::Image::ConstSharedPtr> imgLeftBuf, imgRightBuf;
    mutex mBufMutexLeft, mBufMutexRight;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    auto node = rclcpp::Node::make_shared("orb_slam3");
    node->declare_parameter("vocabulary", "");
    node->declare_parameter("settings", "");
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

    auto imu_grabber = make_shared<ImuGrabber>();
    auto image_grabber = make_shared<ImageGrabber>(&SLAM, imu_grabber, do_rectify, do_equalize);

    // Setup subscribers
    auto sub_imu = imu_grabber->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 
        1000, 
        bind(&ImuGrabber::GrabImu, imu_grabber, placeholders::_1)
    );

    auto sub_img_left = image_grabber->create_subscription<sensor_msgs::msg::Image>(
        "/camera/left/image_raw", 
        rclcpp::QoS(100), 
        bind(&ImageGrabber::GrabImageLeft, image_grabber, placeholders::_1)
    );

    auto sub_img_right = image_grabber->create_subscription<sensor_msgs::msg::Image>(
        "/camera/right/image_raw", 
        rclcpp::QoS(100), 
        bind(&ImageGrabber::GrabImageRight, image_grabber, placeholders::_1)
    );

    thread sync_thread(&ImageGrabber::SyncWithImu, image_grabber);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(imu_grabber);
    executor.add_node(image_grabber);
    executor.spin();

    rclcpp::shutdown();
    sync_thread.join();

    return 0;
}
