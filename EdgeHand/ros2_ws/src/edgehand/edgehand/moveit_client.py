"""The handover Robot interface on top of MoveIt 2 (Humble) and ros2_kortex.

    move_free  -> move_group action /move_action (pose goal for EE_LINK)
    move_line  -> /compute_cartesian_path, slowed to VEL_SCALE, then /execute_trajectory
    gripper    -> /robotiq_gripper_controller/gripper_cmd
    efforts    -> /joint_states effort (measured joint torques)

Blocking calls are made from a worker thread while a MultiThreadedExecutor spins the node.
"""
import threading, time
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from edgehand import config
from edgehand.geometry import quat_from_R

HOME_JOINTS = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]   # Kinova "Home" (joint_1..joint_7, rad)


def pose_msg(p, R):
    m = Pose()
    m.position.x, m.position.y, m.position.z = map(float, p)
    m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = map(float, quat_from_R(R))
    return m


def _wait(future, timeout):
    ev = threading.Event()
    future.add_done_callback(lambda _: ev.set())
    if not ev.wait(timeout):
        raise TimeoutError("ROS call timed out")
    return future.result()


def slow_down(traj, scale):
    """Stretch a JointTrajectory in time by 1/scale (velocities * scale, accelerations * scale^2)."""
    for pt in traj.points:
        t = (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) / scale
        pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        pt.velocities = [v * scale for v in pt.velocities]
        pt.accelerations = [a * scale * scale for a in pt.accelerations]
    return traj


class MoveItRobot:
    def __init__(self, node, timeout=60.0):
        self.node, self.timeout = node, timeout
        self.move = ActionClient(node, MoveGroup, "/move_action")
        self.execute = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
        self.grip = ActionClient(node, GripperCommand, config.GRIPPER_ACTION)
        self.cartesian = node.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._efforts = {}
        self._goal = None                        # current goal handle, for cancel()
        self._lock = threading.Lock()
        node.create_subscription(JointState, "/joint_states", self._on_joints, 10)

    def _on_joints(self, msg):
        if msg.effort:
            self._efforts.update(zip(msg.name, msg.effort))

    def wait_for_servers(self, timeout=30.0):
        ok = all(c.wait_for_server(timeout_sec=timeout) for c in (self.move, self.execute, self.grip))
        return ok and self.cartesian.wait_for_service(timeout_sec=timeout)

    def _send(self, client, goal):
        handle = _wait(client.send_goal_async(goal), self.timeout)
        if not handle.accepted:
            return None
        with self._lock:
            self._goal = handle
        try:
            return _wait(handle.get_result_async(), self.timeout).result
        finally:
            with self._lock:
                self._goal = None

    def _plan_and_go(self, constraints):
        req = MotionPlanRequest()
        req.group_name = config.PLANNING_GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = config.VEL_SCALE
        req.max_acceleration_scaling_factor = config.ACC_SCALE
        req.start_state.is_diff = True
        req.goal_constraints = [constraints]
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        res = self._send(self.move, goal)
        ok = res is not None and res.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.node.get_logger().error(f"move_group failed: {None if res is None else res.error_code.val}")
        return ok

    def move_free(self, p, R):
        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name, pc.weight = config.BASE_FRAME, config.EE_LINK, 1.0
        pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.002])]
        pc.constraint_region.primitive_poses = [pose_msg(p, R)]
        oc = OrientationConstraint()
        oc.header.frame_id, oc.link_name, oc.weight = config.BASE_FRAME, config.EE_LINK, 1.0
        oc.orientation = pose_msg(p, R).orientation
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.02
        c.position_constraints, c.orientation_constraints = [pc], [oc]
        return self._plan_and_go(c)

    def move_line(self, p, R):
        req = GetCartesianPath.Request()
        req.header.frame_id = config.BASE_FRAME
        req.start_state.is_diff = True
        req.group_name, req.link_name = config.PLANNING_GROUP, config.EE_LINK
        req.waypoints = [pose_msg(p, R)]
        req.max_step, req.jump_threshold, req.avoid_collisions = 0.005, 0.0, True
        res = _wait(self.cartesian.call_async(req), self.timeout)
        if res.fraction < 0.99:
            self.node.get_logger().error(f"cartesian path only {res.fraction:.0%} feasible")
            return False
        slow_down(res.solution.joint_trajectory, config.VEL_SCALE)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        r = self._send(self.execute, goal)
        return r is not None and r.error_code.val == MoveItErrorCodes.SUCCESS

    def gripper(self, pos):
        goal = GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = float(pos), config.GRIPPER_MAX_EFFORT
        r = self._send(self.grip, goal)
        # closing on a tool stalls before reaching the setpoint: that's a successful grasp
        return r is not None and (r.reached_goal or r.stalled)

    def home(self):
        c = Constraints()
        for i, q in enumerate(HOME_JOINTS):
            c.joint_constraints.append(JointConstraint(joint_name=f"joint_{i + 1}", position=q, tolerance_above=0.01,
                                                       tolerance_below=0.01, weight=1.0))
        return self._plan_and_go(c)

    def efforts(self):
        return dict(self._efforts)

    def wait(self, seconds):
        time.sleep(seconds)

    def cancel(self):
        with self._lock:
            if self._goal is not None:
                self._goal.cancel_goal_async()
