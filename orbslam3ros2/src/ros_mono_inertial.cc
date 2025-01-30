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
    ImageGrabber(ORB_SLAM3::System* pSLAM, shared_ptr<ImuGrabber> pImuGb, bool bClahe) 
        : Node("image_grabber"), mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe)
    {
        mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    }

    void GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        lock_guard<mutex> lock(mBufMutex);
        if (!imgBuf.empty())
            imgBuf.pop();
        imgBuf.push(msg);
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
        while(rclcpp::ok()) {
            cv::Mat im;
            double tIm = 0;
            
            if (!imgBuf.empty() && !mpImuGb->imuBuf.empty()) {
                tIm = rclcpp::Time(imgBuf.front()->header.stamp).seconds();
                
                if(tIm > rclcpp::Time(mpImuGb->imuBuf.back()->header.stamp).seconds())
                    continue;

                {
                    lock_guard<mutex> lock(mBufMutex);
                    im = GetImage(imgBuf.front());
                    imgBuf.pop();
                }

                vector<ORB_SLAM3::IMU::Point> vImuMeas;
                {
                    lock_guard<mutex> lock(mpImuGb->mBufMutex);
                    while(!mpImuGb->imuBuf.empty() && 
                          rclcpp::Time(mpImuGb->imuBuf.front()->header.stamp).seconds() <= tIm)
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

                if(mbClahe) {
                    mClahe->apply(im, im);
                }

                mpSLAM->TrackMonocular(im, tIm, vImuMeas);
                this_thread::sleep_for(chrono::milliseconds(1));
            }
        }
    }

    ORB_SLAM3::System* mpSLAM;
    shared_ptr<ImuGrabber> mpImuGb;
    bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe;

private:
    queue<sensor_msgs::msg::Image::ConstSharedPtr> imgBuf;
    mutex mBufMutex;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    auto node = rclcpp::Node::make_shared("orb_slam3_mono");
    node->declare_parameter("vocabulary", "");
    node->declare_parameter("settings", "");
    node->declare_parameter("do_equalize", false);

    string voc_file, settings_file;
    bool do_equalize;

    try {
        voc_file = node->get_parameter("vocabulary").as_string();
        settings_file = node->get_parameter("settings").as_string();
        do_equalize = node->get_parameter("do_equalize").as_bool();
    } catch (...) {
        RCLCPP_ERROR(node->get_logger(), "Failed to get parameters");
        return 1;
    }

    ORB_SLAM3::System SLAM(
        voc_file, 
        settings_file, 
        ORB_SLAM3::System::IMU_MONOCULAR, 
        true
    );

    auto imu_grabber = make_shared<ImuGrabber>();
    auto image_grabber = make_shared<ImageGrabber>(&SLAM, imu_grabber, do_equalize);

    // Setup subscribers
    auto sub_imu = imu_grabber->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 
        1000, 
        bind(&ImuGrabber::GrabImu, imu_grabber, placeholders::_1)
    );

    auto sub_img = image_grabber->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 
        rclcpp::SensorDataQoS().keep_last(100), 
        bind(&ImageGrabber::GrabImage, image_grabber, placeholders::_1)
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
