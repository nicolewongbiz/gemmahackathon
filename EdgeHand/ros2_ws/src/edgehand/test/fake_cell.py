"""Stand-in for move_group, the ros2_kortex controllers and the Jetson camera, for testing the EdgeHand nodes
without hardware or MoveIt:
    /move_action, /execute_trajectory, /robotiq_gripper_controller/gripper_cmd   (succeed, log the goal)
    /compute_cartesian_path, /apply_planning_scene                                (succeed)
    /joint_states with efforts; publish std_msgs/Empty on /test/pull to simulate the operator's tug
    /image_raw: synthetic overhead frame with a phillips screwdriver (test/synthetic.py)
"""
import sys, threading, time
from pathlib import Path
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty, String
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic                                                      # noqa: E402

JOINTS = [f"joint_{i}" for i in range(1, 8)]


class FakeCell(Node):
    def __init__(self, grasp_xy=(0.45, 0.05), handle_dir=(0.0, 1.0), motion_s=1.5):
        super().__init__("fake_cell")
        self.motion_s = motion_s
        self.pulling = False
        cb = ReentrantCallbackGroup()
        self.log_pub = self.create_publisher(String, "/test/log", 50)
        ActionServer(self, MoveGroup, "/move_action", self.on_move, callback_group=cb,
                     cancel_callback=lambda _: CancelResponse.ACCEPT)
        ActionServer(self, ExecuteTrajectory, "/execute_trajectory", self.on_execute, callback_group=cb,
                     cancel_callback=lambda _: CancelResponse.ACCEPT)
        ActionServer(self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd", self.on_grip, callback_group=cb)
        self.create_service(GetCartesianPath, "/compute_cartesian_path", self.on_cartesian, callback_group=cb)
        self.create_service(ApplyPlanningScene, "/apply_planning_scene", self.on_scene, callback_group=cb)
        self.js = self.create_publisher(JointState, "/joint_states", 10)
        self.img = self.create_publisher(Image, "/image_raw", 1)
        self.create_subscription(Empty, "/test/pull", lambda _: setattr(self, "pulling", True), 10)
        self.frame = synthetic.frame([("phillips", grasp_xy, handle_dir)])[:, :, ::-1].copy()   # BGR -> rgb8
        self.create_timer(0.02, self.pub_joints)
        self.create_timer(0.2, self.pub_image)

    def log(self, s):
        self.get_logger().info(s)
        self.log_pub.publish(String(data=s))

    def _run(self, gh, what):
        t0 = time.time()
        while time.time() - t0 < self.motion_s:
            if gh.is_cancel_requested:
                gh.canceled()
                self.log(f"{what} CANCELED")
                return False
            time.sleep(0.01)
        return True

    def on_move(self, gh):
        r = gh.request.request
        c = r.goal_constraints[0]
        if c.position_constraints:
            p = c.position_constraints[0].constraint_region.primitive_poses[0].position
            what = f"move_group {c.position_constraints[0].link_name} -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) vel {r.max_velocity_scaling_factor}"
        else:
            what = f"move_group joints -> {[round(j.position, 2) for j in c.joint_constraints]}"
        self.log(what)
        res = MoveGroup.Result()
        if self._run(gh, "move_group"):
            gh.succeed()
            res.error_code.val = MoveItErrorCodes.SUCCESS
        return res

    def on_execute(self, gh):
        pts = gh.request.trajectory.joint_trajectory.points
        T = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9 if pts else 0
        self.log(f"execute_trajectory {len(pts)} points, {T:.1f} s")
        res = ExecuteTrajectory.Result()
        if self._run(gh, "execute_trajectory"):
            gh.succeed()
            res.error_code.val = MoveItErrorCodes.SUCCESS
        return res

    def on_grip(self, gh):
        self.log(f"gripper -> {gh.request.command.position}")
        if gh.request.command.position == 0.0:
            self.pulling = False                              # the operator has the tool now
        gh.succeed()
        res = GripperCommand.Result()
        res.reached_goal = True
        return res

    def on_cartesian(self, req, res):
        p = req.waypoints[0].position
        self.log(f"cartesian {req.link_name} -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) avoid_collisions={req.avoid_collisions}")
        res.fraction = 1.0
        res.solution.joint_trajectory.joint_names = JOINTS
        for t in (0.0, 1.0):
            res.solution.joint_trajectory.points.append(JointTrajectoryPoint(
                positions=[0.0] * 7, velocities=[0.0] * 7, accelerations=[0.0] * 7,
                time_from_start=Duration(sec=int(t))))
        return res

    def on_scene(self, req, res):
        self.log("planning scene: " + ", ".join(o.id for o in req.scene.world.collision_objects))
        res.success = True
        return res

    def pub_joints(self):
        tau = np.array([0.0, 12.0, 0.0, 6.0, 0.3, -1.0, 0.0]) + np.random.normal(0, 0.05, 7)
        if self.pulling:
            tau[5] -= 3.0                                     # wrist joint feels the pull
        m = JointState(name=JOINTS, position=[0.0] * 7, velocity=[0.0] * 7, effort=list(map(float, tau)))
        m.header.stamp = self.get_clock().now().to_msg()
        self.js.publish(m)

    def pub_image(self):
        h, w = self.frame.shape[:2]
        m = Image(height=h, width=w, encoding="rgb8", step=w * 3, data=self.frame.tobytes())
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), "camera"
        self.img.publish(m)


def main():
    rclpy.init()
    ex = MultiThreadedExecutor()
    ex.add_node(FakeCell())
    ex.spin()


if __name__ == "__main__":
    main()
