from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import actionlib
import rospy

if __name__ == '__main__':
    rospy.init_node('test2')
    client = actionlib.SimpleActionClient('gripper_controller/linear_actuator_cmd', GripperCommandAction)
    rospy.sleep(1)
    goal = GripperCommandGoal()
    client.send_goal(goal)
    client.wait_for_result()