"""Jetson: overhead camera -> /tool_pose_target for the requested tool, /edgehand/detections for all of them.

/tool_pose_target (PoseStamped, base_link): position = grasp point on the shaft (z = TABLE_Z + TOOL_RADIUS_M),
orientation = yaw only, with +x pointing from the grasp point towards the handle.
    ros2 run edgehand vision_node --ros-args -p show:=true    (window with detections; click = print table x, y)
"""
import json
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from edgehand import config
from edgehand.geometry import quat_from_R
from edgehand.vision import TableMap, detect_tools, draw


def image_to_bgr(msg):
    """sensor_msgs/Image -> BGR numpy, without cv_bridge (v4l2_camera publishes rgb8 by default)."""
    a = np.frombuffer(msg.data, np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = a.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
    if msg.encoding in ("yuv422_yuy2", "yuyv"):
        return cv2.cvtColor(a.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)
    if msg.encoding == "mono8":
        return cv2.cvtColor(a.reshape(msg.height, msg.step)[:, :msg.width], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported encoding {msg.encoding}")


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.show = self.declare_parameter("show", False).value
        self.tool = config.DEFAULT_TOOL
        self.tmap = TableMap()
        self.pose_pub = self.create_publisher(PoseStamped, "/tool_pose_target", 10)
        self.det_pub = self.create_publisher(String, "/edgehand/detections", 10)
        self.create_subscription(String, "/edgehand/tool_request", self.on_request, 10)
        self.create_subscription(Image, config.CAMERA_TOPIC, self.on_image, 1)
        self.clicks = []
        if self.show:
            cv2.namedWindow("edgehand vision")
            cv2.setMouseCallback("edgehand vision", lambda e, x, y, *_: self.clicks.append((x, y))
                                 if e == cv2.EVENT_LBUTTONDOWN else None)

    def on_request(self, msg):
        if msg.data in config.TOOLS:
            self.tool = msg.data

    def on_image(self, msg):
        frame = image_to_bgr(msg)
        tools = detect_tools(frame, self.tmap) if self.tmap.update(frame) else []
        self.det_pub.publish(String(data=json.dumps([
            {"tool": t["tool"], "grasp_xy": [round(float(v), 4) for v in t["grasp_xy"]],
             "handle_dir": [round(float(v), 3) for v in t["handle_dir"]]} for t in tools])))
        match = [t for t in tools if t["tool"] == self.tool]
        if match:
            t = match[0]
            ps = PoseStamped()
            ps.header.stamp, ps.header.frame_id = msg.header.stamp, config.BASE_FRAME
            ps.pose.position.x, ps.pose.position.y = map(float, t["grasp_xy"])
            ps.pose.position.z = config.TABLE_Z + config.TOOL_RADIUS_M
            hx, hy = t["handle_dir"]
            R = np.array([[hx, -hy, 0], [hy, hx, 0], [0, 0, 1]])
            q = quat_from_R(R)
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
            self.pose_pub.publish(ps)
        if self.show:
            if self.clicks and self.tmap.H is not None:
                print("pixel", self.clicks[-1], "-> table", np.round(self.tmap.px_to_table(*self.clicks.pop()), 4))
            cv2.imshow("edgehand vision", draw(frame, self.tmap, tools))
            cv2.waitKey(1)


def main():
    rclpy.init()
    rclpy.spin(VisionNode())
