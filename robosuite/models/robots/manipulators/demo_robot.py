import numpy as np
from collections import OrderedDict
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

class Demo(ManipulatorModel):

    arms = ["right"]

    def __init__(self, _path="robots/demo/robot.xml", idn=0):
        super().__init__(xml_path_completion(_path), idn=idn)

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
        return {"right": "default_gr1_fixed_lower_body"}

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

class DemoTwoFingered(Demo):
    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values
        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "PandaGripper", "left": "PandaGripper"} # return {"right": "FourierRightHand", "left": "FourierLeftHand"}
    @property
    def gripper_mount_quat_offset(self):
        if self.default_gripper.get("right") == "FourierRightHand":
            return {"right": [0.0, 0.0, 0.0, 1.0], "left": [0.0, 0.0, 0.0, 1.0]}
        #     # return {"right": [0.0, 0.0, 1.0, 0.0], "left": [0.0, 0.0, 1.0, 0.0]}
        #     # return {"right": [0.7071068, 0, 0.0, 0.7071068], "left": [0.0, 0.0, 1.0, 0.0]}
        #     # 0.6335811, 0, 0.6335811, 0.4440158
        #     # return {"right": [0.6335811, 0.6335811, 0, 0.4440158], "left": [0.0, 0.0, 1.0, 0.0]}
        elif self.default_gripper.get("right") == "PandaGripper":
            return {"right": [0.4395489, 0.8790978, 0, -0.1843469], "left": [0.0, 1.0, 0.0, 0.0]}
        else:
            raise ValueError("Invalid gripper type specified!: {}".format(self.default_gripper.get("right")))

class DemoSingleHand(Demo):
    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.0, 0.0, 0.0, 1.0], "left": [0.0, 0.0, 1.0, 0.0]}

class DemoTwoHand(Demo):
    arms = ["left", "right"]
    def __init__(self, idn=0):
        super().__init__("robots/demo/robot_dual.xml", idn=idn)

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
            "left": "left_gripper_mount",
        }

    @property
    def default_gripper(self):
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.0, 0.0, 0.0, 1.0], "left": [0.0, 0.0, 0.0, 1.0]}

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def init_qpos(self):
        # Copied from panda.py
        right_arm_init = np.array([0.0]*12)
        return right_arm_init

    @property
    def default_controller_config(self):
        # Copied from panda.py
        return {"right": "default_gr1_fixed_lower_body", "left": "default_gr1_fixed_lower_body"}
