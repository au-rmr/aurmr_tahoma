from aurmr_tasks.srv import StowRequest, PickRequest, StowRequestRequest, PickRequestRequest, MultiplePickRequest, MultiplePickRequestRequest
from aurmr_tasks.msg import PickStatus

import smach_ros

from smach import State, StateMachine

import rospy

from queue import Empty, Queue


class PickStowQueue:
    def __init__(self) -> None:
        self.stow_service = rospy.Service('~stow', StowRequest, self.stow_cb)
        self.pick_service = rospy.Service('~pick', PickRequest, self.pick_cb)
        self.multi_pick_service = rospy.Service('~multiple_pick', MultiplePickRequest, self.multiple_pick_cb)
        self.status_pub = rospy.Publisher("/demo_status", PickStatus, queue_size=5, latch=True)
        self.task_queue = Queue()
        self.result_queue = Queue()

    def stow_cb(self, request: StowRequestRequest):
        rospy.loginfo("STOW REQUEST: " + str(request))
        self.task_queue.put_nowait(request)
        result = None
        while result is None and not rospy.is_shutdown():
            try:
                result = self.result_queue.get(False)
            except Empty:
                rospy.sleep(.1)
        return {"success": result}

    def multiple_pick_cb(self, requests: MultiplePickRequestRequest):
        for i in range(len(requests.bin_ids)):
            r = PickRequestRequest()
            r.object_id = requests.object_ids[i]
            r.bin_id = requests.bin_ids[i]
            r.object_asin = requests.object_asins[i]
            self.task_queue.put_nowait(r)
        # We don't want to leave the service call hanging for multiple minutes,
        # so we don't report success with multi pick
        return True

    def pick_cb(self, request: PickRequestRequest):
        rospy.loginfo("PICK REQUEST: " + str(request))
        self.task_queue.put_nowait(request)
        result = None
        while result is None and not rospy.is_shutdown():
            try:
                result = self.result_queue.get(False)
            except Empty:
                rospy.sleep(.1)
        return {"success": result}


class WaitForPickStow(State):
    def __init__(self, task_queue):
        State.__init__(self, outcomes=['pick', 'stow', 'aborted'], output_keys=["request"])
        self.task_queue = task_queue

    def execute(self, userdata):
        task = None
        while task is None and not rospy.is_shutdown():
            try:
                task = self.task_queue.task_queue.get(False)
            except Empty:
                rospy.sleep(.1)
            rospy.sleep(.1)

        if task is None:
            # We must be getting shutdown
            return "aborted"

        userdata["request"] = [task.bin_id, task.object_id, task.object_asin]
        print(task, type(task))
        if isinstance(task, StowRequestRequest):
            return "stow"
        elif isinstance(task, PickRequestRequest):
            return "pick"
        else:
            return "done"

class ReportPickStowOutcome(State):
    def __init__(self, task_queue):
        State.__init__(self, outcomes=['succeeded'], input_keys=["request", "status"])
        self.task_queue = task_queue

    def execute(self, userdata):
        # TODO: This report should include the request info
        #result = userdata["status"] == True
        result = True
        self.task_queue.result_queue.put_nowait(result)
        return "succeeded"
