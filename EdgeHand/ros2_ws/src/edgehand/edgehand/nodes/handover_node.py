"""Workstation: /edgehand/tool_request -> wait for a fresh /tool_pose_target -> pick + hand over through MoveIt.

/edgehand/stop cancels the motion (the controller holds position); /edgehand/release opens the gripper at the
handover pose. Requests are refused while busy, or while stops are arriving (Pi watchdog).
"""
import threading, time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String
from edgehand import config
from edgehand.geometry import R_from_quat
from edgehand.handover import Handover
from edgehand.moveit_client import MoveItRobot


class HandoverNode(Node):
    def __init__(self, robot=None):
        super().__init__("handover_node")
        self.pose_timeout = self.declare_parameter("pose_timeout", 5.0).value
        self.robot = robot or MoveItRobot(self)
        self.state_pub = self.create_publisher(String, "/edgehand/state", 10)
        self.say_pub = self.create_publisher(String, "/edgehand/say", 10)
        self.handover = Handover(self.robot, log=self.get_logger().info, state=self.publish_state)
        self.busy = threading.Lock()
        self.last_pose = None
        self.last_stop = 0.0
        self.create_subscription(String, "/edgehand/tool_request", self.on_request, 10)
        self.create_subscription(PoseStamped, "/tool_pose_target", self.on_pose, 10)
        self.create_subscription(Empty, "/edgehand/stop", self.on_stop, 10)
        self.create_subscription(Empty, "/edgehand/release", lambda _: self.handover.release(), 10)
        self.create_timer(0.5, self.publish_tug)
        self.publish_state("idle")

    def publish_state(self, s):
        self.state_pub.publish(String(data=s))
        self.get_logger().info(f"state: {s}")

    def publish_tug(self):
        d = self.handover.detector
        if self.busy.locked() and d.baseline is not None:
            self.state_pub.publish(String(data=f"holding: delta torque {d.last_delta:.2f} / {config.TUG_TORQUE_NM} N*m"))

    def on_pose(self, msg):
        self.last_pose = msg

    def on_stop(self, _):
        self.last_stop = time.time()
        if self.busy.locked():
            self.handover.stop()

    def on_request(self, msg):
        if msg.data not in config.TOOLS:
            return
        if time.time() - self.last_stop < 1.0:
            self.say_pub.publish(String(data="Stop is active."))
            return
        if not self.busy.acquire(blocking=False):
            self.say_pub.publish(String(data="I'm busy."))
            return
        threading.Thread(target=self._run, args=(msg.data, self.get_clock().now()), daemon=True).start()

    def _run(self, tool, t_req):
        try:
            self.publish_state(f"looking for the {tool}")
            pose = self._wait_pose(t_req)
            if pose is None:
                self.say_pub.publish(String(data=f"I can't see the {tool}."))
                self.publish_state("idle")
                return
            p = pose.pose.position
            q = pose.pose.orientation
            handle_dir = R_from_quat([q.x, q.y, q.z, q.w])[:2, 0]
            self.get_logger().info(f"{tool}: grasp ({p.x:.3f}, {p.y:.3f}), handle towards {np.round(handle_dir, 2)}")
            ok, msg = self.handover.run((p.x, p.y), handle_dir)
            self.say_pub.publish(String(data=f"Here's the {tool}." if ok else f"Handover failed: {msg}"))
            self.get_logger().info(f"handover: {ok} {msg}")
        finally:
            self.busy.release()

    def _wait_pose(self, t_req):
        """A pose from a camera frame taken after the request (vision switches to the new tool on the same topic)."""
        deadline = time.time() + self.pose_timeout
        while time.time() < deadline:
            m = self.last_pose
            if m is not None and Time.from_msg(m.header.stamp) > t_req:
                return m
            time.sleep(0.05)
        return None


def main():
    rclpy.init()
    node = HandoverNode()
    if not node.robot.wait_for_servers(30.0):
        node.get_logger().error("MoveIt / gripper servers not available; is the MoveIt launch running?")
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    ex.spin()
