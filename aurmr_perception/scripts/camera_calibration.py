#!/usr/bin/env python

import rospy
import tf
import tkinter as tk
from tkinter import Scale
import threading
import math

class TFSlider:
    def __init__(self):
        # ROS Initialization
        rospy.init_node('tf_slider_publisher', anonymous=True)
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        # Original transformation
        self.original_trans = [0.0, 0.0, 0.0]
        self.original_rot = [0.0, 0.0, 0.0, 1.0]

        # Initial offsets for translation
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0

        # Initial offsets for rotation (RPY)
        self.offset_roll = 0.0
        self.offset_pitch = 0.0
        self.offset_yaw = 0.0

        # Fetch initial transformation
        self.fetch_initial_tf()

        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_tf)
        self.is_locked = False

        # Start GUI in a separate thread
        self.gui_thread = threading.Thread(target=self.init_gui)
        self.gui_thread.start()

        rospy.Timer(rospy.Duration(0.1), self.publish_tf)

    def fetch_initial_tf(self):
        while not rospy.is_shutdown():
            try:
                (self.original_trans, self.original_rot) = self.listener.lookupTransform('base_link', 'pod_base_link', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.sleep(0.1)

    def init_gui(self):
        self.root = tk.Tk()
        self.root.title('Manual POD Calibration')

        # Sliders for transformation offset
        self.init_sliders()

        self.lock_button = tk.Button(self.root, text="Lock TF", command=self.lock_tf)
        self.lock_button.pack(padx=10, pady=10)

        self.root.mainloop()

    def init_sliders(self):
        # Slider for x transformation offset
        self.x_slider = Scale(self.root, from_=-0.15, to=0.15, resolution=0.001, orient=tk.HORIZONTAL, label='X Offset', command=self.update_x)
        self.x_slider.pack(fill=tk.X, padx=10, pady=10)

        # Slider for y transformation offset
        self.y_slider = Scale(self.root, from_=-0.15, to=0.15, resolution=0.001, orient=tk.HORIZONTAL, label='Y Offset', command=self.update_y)
        self.y_slider.pack(fill=tk.X, padx=10, pady=10)

        # Slider for z transformation offset
        self.z_slider = Scale(self.root, from_=-0.15, to=0.15, resolution=0.001, orient=tk.HORIZONTAL, label='Z Offset', command=self.update_z)
        self.z_slider.pack(fill=tk.X, padx=10, pady=10)

        # Slider for roll rotation offset
        self.roll_slider = Scale(self.root, from_=-15, to=15, resolution=0.1, orient=tk.HORIZONTAL, label='Roll Offset (Degrees)', command=self.update_roll)
        self.roll_slider.pack(fill=tk.X, padx=10, pady=10)

        # Slider for pitch rotation offset
        self.pitch_slider = Scale(self.root, from_=-15, to=15, resolution=0.1, orient=tk.HORIZONTAL, label='Pitch Offset (Degrees)', command=self.update_pitch)
        self.pitch_slider.pack(fill=tk.X, padx=10, pady=10)

        # Slider for yaw rotation offset
        self.yaw_slider = Scale(self.root, from_=-15, to=15, resolution=0.1, orient=tk.HORIZONTAL, label='Yaw Offset (Degrees)', command=self.update_yaw)
        self.yaw_slider.pack(fill=tk.X, padx=10, pady=10)

    def update_x(self, value):
        self.offset_x = float(value)

    def update_y(self, value):
        self.offset_y = float(value)

    def update_z(self, value):
        self.offset_z = float(value)

    def update_roll(self, value):
        self.offset_roll = math.radians(float(value))

    def update_pitch(self, value):
        self.offset_pitch = math.radians(float(value))

    def update_yaw(self, value):
        self.offset_yaw = math.radians(float(value))

    def publish_tf(self, event):
        modified_trans = [self.original_trans[0] + self.offset_x,
                          self.original_trans[1] + self.offset_y,
                          self.original_trans[2] + self.offset_z]

        original_roll, original_pitch, original_yaw = tf.transformations.euler_from_quaternion(self.original_rot)

        modified_rot = tf.transformations.quaternion_from_euler(original_roll + self.offset_roll,
                                                                original_pitch + self.offset_pitch,
                                                                original_yaw + self.offset_yaw)

        self.br.sendTransform(modified_trans,
                              modified_rot,
                              rospy.Time.now(),
                              "pod_base_link",
                              "base_link")

    def lock_tf(self):
        if not self.is_locked:
            # If not already locked, lock it
            self.is_locked = True
            self.lock_button.config(text="Unlock TF")

            # Disable sliders
            self.x_slider.config(state=tk.DISABLED)
            self.y_slider.config(state=tk.DISABLED)
            self.z_slider.config(state=tk.DISABLED)
            self.roll_slider.config(state=tk.DISABLED)
            self.pitch_slider.config(state=tk.DISABLED)
            self.yaw_slider.config(state=tk.DISABLED)

            # Start the thread that continuously publishes the transform
            threading.Thread(target=self.lock_thread).start()

            # Shut down the regular timer
            self.timer.shutdown()

        else:
            # If already locked, unlock it
            self.is_locked = False
            self.lock_button.config(text="Lock TF")

            # Enable sliders
            self.x_slider.config(state=tk.NORMAL)
            self.y_slider.config(state=tk.NORMAL)
            self.z_slider.config(state=tk.NORMAL)
            self.roll_slider.config(state=tk.NORMAL)
            self.pitch_slider.config(state=tk.NORMAL)
            self.yaw_slider.config(state=tk.NORMAL)

            # Restart the regular timer
            self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_tf)

    def lock_thread(self):
        while self.is_locked and not rospy.is_shutdown():
            modified_trans = [self.original_trans[0] + self.offset_x,
                            self.original_trans[1] + self.offset_y,
                            self.original_trans[2] + self.offset_z]


            original_roll, original_pitch, original_yaw = tf.transformations.euler_from_quaternion(self.original_rot)

            modified_rot = [original_roll + self.offset_roll,
                                    original_pitch + self.offset_pitch,
                                    original_yaw + self.offset_yaw]
            modified_rot = tf.transformations.quaternion_from_euler(*modified_rot)

            rospy.loginfo_throttle_identical(120, f"New pod transform: {modified_trans[0]:.3f} {modified_trans[1]:.3f} {modified_trans[2]:.3f} {modified_rot[2]:.3f} {modified_rot[1]:.3f} {modified_rot[0]:.3f}")
            self.br.sendTransform(modified_trans,
                                modified_rot,
                                rospy.Time.now(),
                                "pod_base_link",
                                "base_link")
            rospy.sleep(0.1)


if __name__ == '__main__':
    slider = TFSlider()
    rospy.spin()