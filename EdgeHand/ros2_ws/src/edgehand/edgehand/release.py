"""Tug detection: the arm holds still at the handover pose and the operator pulls the tool.

ros2_kortex reports measured joint torques in /joint_states effort (N*m). A pull on the tool changes the wrist
torques; open the gripper when the change stays above a threshold for a few samples.
"""
import numpy as np
from edgehand import config


class TugDetector:
    def __init__(self, joints=config.TUG_JOINTS, threshold=config.TUG_TORQUE_NM, samples=config.TUG_SAMPLES):
        self.joints, self.threshold, self.samples = list(joints), threshold, samples
        self.baseline = None
        self.count = 0
        self.last_delta = 0.0

    def _vec(self, efforts):
        return np.array([efforts[j] for j in self.joints], float)

    def set_baseline(self, samples):
        """samples: list of {joint: effort} taken while holding still. Averaging rejects sensor noise."""
        self.baseline = np.mean([self._vec(s) for s in samples], axis=0)
        self.count = 0

    def update(self, efforts):
        """True once the tug has lasted `samples` consecutive updates."""
        if self.baseline is None:
            raise RuntimeError("set_baseline() first")
        self.last_delta = float(np.linalg.norm(self._vec(efforts) - self.baseline))
        self.count = self.count + 1 if self.last_delta > self.threshold else 0
        return self.count >= self.samples
