"""
Dexterous hands for GR1 robot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class FourierLeftHand(GripperModel):
    """
    Dexterous left hand of GR1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/fourier_left_hand.xml"), idn=idn)

    def format_action(self, action):
        # the more correct way is to add <equality> tag in the xml
        # however the tag makes finger movement laggy, so manually copy the value for finger joints
        # 0 is thumb rot, no copying. Thumb bend has 3 joints, so copy 3 times. Other fingers has 2 joints, so copy 2 times.
        assert len(action) == self.dof
        action = np.array(action)
        indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        return action[indices]

    @property
    def init_qpos(self):
        return np.array([0.0] * 11)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 6

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "L_thumb_proximal_base_link_col",
                "L_thumb_proximal_link_col",
                "L_thumb_distal_link_col",
            ],
            "right_finger": [
                "L_index_proximal_link_col",
                "L_index_intermediate_link_col",
                "L_middle_proximal_link_col",
                "L_middle_proximal_link_col",
                "L_ring_proximal_link_col",
                "L_ring_intermediate_link_col",
                "L_pinky_proximal_link_col",
                "L_pinky_intermediate_link_col",
            ],
            "left_fingerpad": [
                "L_thumb_proximal_base_link_col",
                "L_thumb_proximal_link_col",
                "L_thumb_distal_link_col",
            ],
            "right_fingerpad": [
                "L_index_proximal_link_col",
                "L_index_intermediate_link_col",
                "L_middle_proximal_link_col",
                "L_middle_proximal_link_col",
                "L_ring_proximal_link_col",
                "L_ring_intermediate_link_col",
                "L_pinky_proximal_link_col",
                "L_pinky_intermediate_link_col",
            ],
        }


class FourierRightHand(GripperModel):
    """
    Dexterous right hand of GR1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/fourier_right_hand.xml"), idn=idn)

    def format_action(self, action):
        # the more correct way is to add <equality> tag in the xml
        # however the tag makes finger movement laggy, so manually copy the value for finger joints
        # 0 is thumb rot, no copying. Thumb bend has 3 joints, so copy 3 times. Other fingers has 2 joints, so copy 2 times.
        assert len(action) == self.dof
        action = np.array(action)
        # indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        indices = np.array([0]) # only supporting power grasp
        return action[indices]

    @property
    def init_qpos(self):
        return np.array([1.83157812, 0.0029788, 0.00743951, 0.14403124, 0.04400525, 0.15292447, 0.05207598, 0.14401271, 0.04400369, 0.13652065, 0.03757485])

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        # return 6
        return 1

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "R_thumb_proximal_base_link_col",
                "R_thumb_proximal_link_col",
                "R_thumb_distal_link_col",
            ],
            "right_finger": [
                "R_index_proximal_link_col",
                "R_index_intermediate_link_col",
                "R_middle_proximal_link_col",
                "R_middle_proximal_link_col",
                "R_ring_proximal_link_col",
                "R_ring_intermediate_link_col",
                "R_pinky_proximal_link_col",
                "R_pinky_intermediate_link_col",
            ],
            "left_fingerpad": [
                "R_thumb_proximal_base_link_col",
                "R_thumb_proximal_link_col",
                "R_thumb_distal_link_col",
            ],
            "right_fingerpad": [
                "R_index_proximal_link_col",
                "R_index_intermediate_link_col",
                "R_middle_proximal_link_col",
                "R_middle_proximal_link_col",
                "R_ring_proximal_link_col",
                "R_ring_intermediate_link_col",
                "R_pinky_proximal_link_col",
                "R_pinky_intermediate_link_col",
            ],
        }
