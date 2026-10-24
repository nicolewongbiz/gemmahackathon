"""Pose math shared by vision, the handover state machine and the sim. Quaternions are (x, y, z, w) like ROS."""
import numpy as np
from edgehand import config


def quat_from_R(R):
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def R_from_quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tool_down_R(handle_dir_xy):
    """Gripper pointing straight down (tool z = -world z), with tool +x along handle_dir_xy.

    The 2F-85 fingers close along the tool's y axis, so they close across a tool lying along x."""
    x = np.array([handle_dir_xy[0], handle_dir_xy[1], 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def handover_R():
    """Same grip, carried to the operator: still pointing down, handle (+x) along OPERATOR_DIR."""
    return tool_down_R(config.OPERATOR_DIR[:2])


def flange_from_tcp(p_tcp, R):
    """MoveIt plans for EE_LINK (the flange); the fingertips are TCP_OFFSET_M along the tool z axis."""
    return np.asarray(p_tcp, float) - R[:, 2] * config.TCP_OFFSET_M


def in_workspace(p):
    return all(lo <= v <= hi for v, (lo, hi) in zip(p, (config.WORKSPACE[k] for k in "xyz")))


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))
