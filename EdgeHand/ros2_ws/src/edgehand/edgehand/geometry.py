"""Pose math shared by vision, the handover state machine and the sim. Quaternions are (x, y, z, w) like ROS."""
import numpy as np
from edgehand import config


def quat_from_R(R):
    """Rotation matrix -> (x, y, z, w). Shepperd's method: stable for 180-degree rotations too."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def R_from_quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tool_down_R(handle_dir_xy):
    """Flange orientation for a tool lying along handle_dir_xy: pointing straight down (flange z = -world z),
    fingers closing across the tool. FINGERS_ALONG says which flange axis the fingers close along, so the other
    horizontal flange axis lies along the tool and points at the handle."""
    d = np.array([handle_dir_xy[0], handle_dir_xy[1], 0.0])
    d /= np.linalg.norm(d)
    z = np.array([0.0, 0.0, -1.0])
    if config.FINGERS_ALONG == "y":
        x = d
        y = np.cross(z, x)
    else:
        y = d
        x = np.cross(y, z)
    return np.column_stack([x, y, z])


def kortex_R(tx, ty, tz):
    """Kortex theta_x, theta_y, theta_z (deg) -> rotation matrix. Extrinsic XYZ: R = Rz @ Ry @ Rx (checked against
    the Gen3 Home pose, which reads (90, 0, 90) in the Web App and in the MuJoCo model)."""
    x, y, z = np.radians([tx, ty, tz])
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def look_pose():
    """Flange pose for the wrist camera to see the tray: (position, R)."""
    return np.array(config.LOOK_FLANGE_XYZ, float), kortex_R(*config.LOOK_THETA)


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
