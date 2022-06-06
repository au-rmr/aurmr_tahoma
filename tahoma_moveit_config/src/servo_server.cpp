/*******************************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2019, Los Alamos National Security, LLC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

/*      Title     : servo_server.cpp
 *      Project   : moveit_servo
 *      Created   : 12/31/2018
 *      Author    : Andy Zelenak
 */

#include <moveit_servo/servo.h>
#include <tahoma_moveit_config/pose_tracking.h>
#include <tahoma_moveit_config/ServoToPoseAction.h>
#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <std_srvs/SetBool.h>



namespace
{
    constexpr char LOGNAME[] = "servo_server";
    constexpr char ROS_THREADS = 8;

}  // namespace


// Class for monitoring status of moveit_servo
class StatusMonitor
{
public:
    StatusMonitor(ros::NodeHandle& nh, const std::string& topic)
    {
        sub_ = nh.subscribe(topic, 1, &StatusMonitor::statusCB, this);
    }

private:
    void statusCB(const std_msgs::Int8ConstPtr& msg)
    {
        moveit_servo::StatusCode latest_status = static_cast<moveit_servo::StatusCode>(msg->data);
        if (latest_status != status_)
        {
            status_ = latest_status;
            const auto& status_str = moveit_servo::SERVO_STATUS_CODE_MAP.at(status_);
            ROS_INFO_STREAM_NAMED(LOGNAME, "Servo status: " << status_str);
        }
    }
    moveit_servo::StatusCode status_ = moveit_servo::StatusCode::INVALID;
    ros::Subscriber sub_;
};



class ServoToPoseActionServer
{
protected:

    ros::NodeHandle nh_;
    actionlib::SimpleActionServer<tahoma_moveit_config::ServoToPoseAction> as_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
    std::string action_name_;
    // create messages that are used to published feedback/result
    tahoma_moveit_config::ServoToPoseFeedback feedback_;
    tahoma_moveit_config::ServoToPoseResult result_;
    tahoma_moveit_config::PoseTracking tracker_;
    ros::ServiceServer pause_srv_;

public:

    ServoToPoseActionServer(ros::NodeHandle nh, std::string name, planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor) : nh_(nh),
            as_(nh_, name, boost::bind(&ServoToPoseActionServer::executeCB, this, _1), false),
            action_name_(name), tracker_(nh_, planning_scene_monitor), pause_srv_(nh_.advertiseService("set_pause", &ServoToPoseActionServer::pauseCB, this))
    {
        as_.start();
    }

    ~ServoToPoseActionServer(void)
    {
    }

    void executeCB(const tahoma_moveit_config::ServoToPoseGoalConstPtr &goal)
    {

        Eigen::Vector3d lin_tol{ goal->positional_tolerance.data() };
        double rot_tol = goal->angular_tolerance;
        ros::Time timeout = ros::Time::now() + goal->timeout;

        // resetTargetPose() can be used to clear the target pose and wait for a new one, e.g. when moving between multiple
        // waypoints
        tracker_.resetTargetPose();
        tracker_.targetPoseCallback(boost::make_shared<geometry_msgs::PoseStamped>(goal->pose));

        // Run the pose tracking in a new thread
        std::thread move_to_pose_thread(
                [this, &lin_tol, &rot_tol, timeout] { this->tracker_.moveToPose(lin_tol, rot_tol, timeout.toSec() /* target pose timeout */); });

        feedback_.error.clear();
        feedback_.error.push_back(std::numeric_limits<double>::signaling_NaN());
        feedback_.error.push_back(std::numeric_limits<double>::signaling_NaN());
        feedback_.error.push_back(std::numeric_limits<double>::signaling_NaN());
        feedback_.error.push_back(std::numeric_limits<double>::signaling_NaN());
        ros::Rate loop_rate(5);
        bool success = false;
        while (!tracker_.done_moving_to_pose_)
        {
            tracker_.getPIDErrors(feedback_.error[0], feedback_.error[1],feedback_.error[2],feedback_.error[3]);
            as_.publishFeedback(feedback_);
            if (as_.isPreemptRequested() || ros::Time::now() > timeout) {
              ROS_INFO_STREAM_NAMED(LOGNAME, "Timed out or preempted. Exiting monitor loop");
              // Make sure the tracker is stopped and clean up
              tracker_.stopMotion();
              break;
            }
            loop_rate.sleep();
        }

        if (tracker_.satisfiesPoseTolerance(lin_tol, rot_tol))
        {
          ROS_INFO_STREAM_NAMED(LOGNAME, "Pose tolerance satisfied.");
          success = true;
        }

        move_to_pose_thread.join();
        if (as_.isPreemptRequested()) {
            as_.setPreempted(result_);
        }
        else if (success) {
            as_.setSucceeded(result_);
        } else {
            as_.setAborted(result_);
        }


    }

    bool pauseCB(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response)
    {
        this->tracker_.servo_->setPaused(request.data);
        response.success = true;
        return true;
    }


};


int main(int argc, char** argv)
{
    ros::init(argc, argv, LOGNAME);
    ros::NodeHandle nh("~");
    ros::AsyncSpinner spinner(8);
    spinner.start();

    // Load the planning scene monitor
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor;
    planning_scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>("robot_description");
    if (!planning_scene_monitor->getPlanningScene())
    {
        ROS_ERROR_STREAM_NAMED(LOGNAME, "Error in setting up the PlanningSceneMonitor.");
        exit(EXIT_FAILURE);
    }

    planning_scene_monitor->startSceneMonitor();
    planning_scene_monitor->startWorldGeometryMonitor(
            planning_scene_monitor::PlanningSceneMonitor::DEFAULT_COLLISION_OBJECT_TOPIC,
            planning_scene_monitor::PlanningSceneMonitor::DEFAULT_PLANNING_SCENE_WORLD_TOPIC,
            false /* skip octomap monitor */);
    planning_scene_monitor->startStateMonitor();


    ServoToPoseActionServer as{nh,"servo_to_pose", planning_scene_monitor};

    // Subscribe to servo status (and log it when it changes)
    StatusMonitor status_monitor(nh, "status");

    ros::waitForShutdown();

    return EXIT_SUCCESS;
}