import numpy as np
from collections import OrderedDict
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

class Demo(ManipulatorModel):

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/demo/robot.xml"), idn=idn)
        self.set_joint_attribute(attrib="damping", values=np.array((10.0,)*6))

    # -------------------------------------------------------------------------------------- #
    # -------------------------- Private Properties ---------------------------------------- #
    # -------------------------------------------------------------------------------------- #

    @property
    def _important_sites(self):
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        # return {"right-eef": "right_gripper_mount",
        #         "right-tool": "right_tool_mount",
        #         "left-eef": "left_gripper_mount",
        #         "left-tool": "left_tool_mount"}
        return {
            "right": "right_gripper_mount", # merge name
            # "right-tool": "right_tool_mount",
            # "left": "left_gripper_mount",
            # "left-tool": "left_tool_mount"
        }


    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def default_base(self):
        # return "FixedLeggedBase"
        # return "RethinkMount"
        return "NoActuationBase"

    @property
    def default_gripper(self):
        """
        Defines the default gripper type for this robot that gets added to end effector

        Returns:
            str: Default gripper name to add to this robot
        """
        return {"right": "InspireRightHand"}

    @property
    def arm_type(self):
        """
        Type of robot arm. Should be either "bimanual" or "single" (or something else if it gets added in the future)

        Returns:
            str: Type of robot
        """
        return "single"

    @property
    def base_xpos_offset(self):
        # Copied from panda.py
        return {
            # "bins": (-0.5, -0.1, 0),
            # "empty": (-0.6, 0, 0),
            # "table": lambda table_length: (-0.16 - table_length / 2, 0, 1.0),
            "bins": (-0.30, -0.1, 0.97),
            "empty": (-0.29, 0, 0.97),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.97),
        }

    @property
    def top_offset(self):
        # Copied from panda.py
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        # Copied from panda.py
        return 0.5

    # @property
    # def default_mount(self):
    #     raise NotImplementedError

    # @property
    # def default_controller_config(self):
    #     raise NotImplementedError
    @property
    def default_controller_config(self):
        # Copied from panda.py
        return {"right": "default_gr1"}

    @property
    def init_qpos(self):
        # Copied from panda.py
        right_arm_init = np.array([0.0]*6)
        # right_arm_init = np.array([0.0, -0.1, 0.0, -np.pi/2, np.pi, 0.0])
        # right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0])
        return right_arm_init

    # @property
    # def key_map(self):
    #     return self._key_map
