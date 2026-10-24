"""Workstation: Kinova wrist camera -> where the requested tool is.

The camera rides on the wrist: a USB camera on this computer (our RealSense), the Kinova Vision module over RTSP,
or a ROS image topic (config.CAMERA_SOURCE).
handover_node moves the arm to the look pose, then publishes /edgehand/locate {"id", "tool"}. This node takes a
camera frame from after the request, finds every tool (vision.find_tools), names them (Gemma, or handle colour if VISION_NAMER = "colour")
and answers on /edgehand/locate_result {"id", "tool", "found", "grasp_xy", "handle_dir", "seen"}. Matching is by id,
not timestamps, so it also works if vision runs on another machine whose clock isn't synced (air-gapped).
It also publishes /tool_pose_target (PoseStamped, base_link: grasp point on the shaft, +x towards the handle) for RViz,
and /edgehand/detections (JSON) for debugging.
    ros2 run edgehand vision_node --ros-args -p show:=true    (window with detections; click = print table x, y)
"""
import json, threading
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from edgehand import config
from edgehand.camera import StreamCamera, configured_source
from edgehand.geometry import quat_from_R
from edgehand.vision import TableMap, draw, find_tools


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
        self.source = self.declare_parameter("camera", config.CAMERA_SOURCE).value
        self.namer_kind = self.declare_parameter("namer", config.VISION_NAMER).value
        self.namer = None
        if self.namer_kind == "gemma":
            from edgehand.namer import GemmaNamer
            self.namer = GemmaNamer()
        self.tmap = TableMap()
        self.frame, self.frame_n = None, 0          # latest frame and a counter, to wait for one taken after a locate
        self.frame_cv = threading.Condition()
        self.busy = threading.Lock()
        self.pose_pub = self.create_publisher(PoseStamped, "/tool_pose_target", 10)
        self.det_pub = self.create_publisher(String, "/edgehand/detections", 10)
        self.result_pub = self.create_publisher(String, "/edgehand/locate_result", 10)
        cb = ReentrantCallbackGroup()
        self.create_subscription(String, "/edgehand/locate", self.on_locate, 10, callback_group=cb)
        if self.source == "topic":
            self.create_subscription(Image, config.CAMERA_TOPIC, lambda m: self.on_frame(image_to_bgr(m)), 1,
                                     callback_group=cb)
        else:
            self.cam = StreamCamera(on_frame=self.on_frame)
        self.clicks = []
        if self.show:
            cv2.namedWindow("edgehand vision")
            cv2.setMouseCallback("edgehand vision", lambda e, x, y, *_: self.clicks.append((x, y))
                                 if e == cv2.EVENT_LBUTTONDOWN else None)
        self.get_logger().info(f"ready: camera {configured_source() if self.source != 'topic' else config.CAMERA_TOPIC}, "
                               f"naming tools by {self.namer_kind}")

    def on_frame(self, frame):
        with self.frame_cv:
            self.frame, self.frame_n = frame, self.frame_n + 1
            self.frame_cv.notify_all()
        if self.show:                                # unnamed outlines; naming only happens on a locate
            tm = TableMap()
            tools = find_tools(frame, tm) if tm.update(frame) else []
            if self.clicks and tm.H is not None:
                print("pixel", self.clicks[-1], "-> table", np.round(tm.px_to_table(*self.clicks.pop()), 4))
            cv2.imshow("edgehand vision", draw(frame, tm, [dict(t, tool=t["colour"]) for t in tools]))
            cv2.waitKey(1)

    def on_locate(self, msg):
        req = json.loads(msg.data)
        if not self.busy.acquire(blocking=False):
            self._reply(req, False, note="busy with another locate")
            return
        threading.Thread(target=self._locate, args=(req,), daemon=True).start()

    def _locate(self, req):
        try:
            with self.frame_cv:                      # a frame from after the request (arm settled at the look pose)
                n0 = self.frame_n
                if not self.frame_cv.wait_for(lambda: self.frame_n > n0 + config.LOCATE_SKIP_FRAMES, timeout=5.0):
                    self._reply(req, False, note="no camera frames")
                    return
                frame = self.frame
            if not self.tmap.update(frame):
                self._reply(req, False, note="can't see all four table tags")
                return
            tools = find_tools(frame, self.tmap)
            names = self.namer.name(frame, tools) if self.namer else [t["colour"] for t in tools]
            for t, n in zip(tools, names):
                t["tool"] = n
            self.det_pub.publish(String(data=json.dumps([
                {"tool": t["tool"], "grasp_xy": [round(float(v), 4) for v in t["grasp_xy"]],
                 "handle_dir": [round(float(v), 3) for v in t["handle_dir"]]} for t in tools])))
            seen = [t["tool"] or "unknown" for t in tools]
            self.get_logger().info(f"locate {req}: saw {seen}")
            match = [t for t in tools if t["tool"] == req["tool"]]
            if not match:
                self._reply(req, False, seen=seen)
                return
            t = match[0]
            self._reply(req, True, t, seen)
            self._publish_pose(t)
        finally:
            self.busy.release()

    def _reply(self, req, found, t=None, seen=(), note=""):
        r = {"id": req.get("id"), "tool": req.get("tool"), "found": found, "seen": list(seen), "note": note}
        if t is not None:
            r.update(grasp_xy=[float(v) for v in t["grasp_xy"]], handle_dir=[float(v) for v in t["handle_dir"]])
        self.result_pub.publish(String(data=json.dumps(r)))

    def _publish_pose(self, t):
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = self.get_clock().now().to_msg(), config.BASE_FRAME
        ps.pose.position.x, ps.pose.position.y = map(float, t["grasp_xy"])
        ps.pose.position.z = config.TABLE_Z + config.TOOL_RADIUS_M
        hx, hy = t["handle_dir"]
        q = quat_from_R(np.array([[hx, -hy, 0], [hy, hx, 0], [0, 0, 1]]))
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
        self.pose_pub.publish(ps)


def main():
    rclpy.init()
    ex = MultiThreadedExecutor()
    ex.add_node(VisionNode())
    ex.spin()
