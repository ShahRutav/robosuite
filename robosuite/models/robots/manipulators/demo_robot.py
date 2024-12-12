import numpy as np
from collections import OrderedDict
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

class Demo(ManipulatorModel):

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/demo/robot.xml"), idn=idn)
        self.set_joint_attribute(attrib="damping", values=np.array((10.0,)*6))
        # self._key_map = {"joint": {}, "actuator": {}}
        # self.grippers = OrderedDict()


    # def add_gripper(self, gripper, arm_name=None):
    #     """
    #     Mounts @gripper to arm.

    #     Throws error if robot already has a gripper or gripper type is incorrect.

    #     Args:
    #         gripper (GripperModel): gripper MJCF model
    #         arm_name (str): name of arm mount -- defaults to self.eef_name if not specified

    #     Raises:
    #         ValueError: [Multiple grippers]
    #     """
    #     if arm_name is None:
    #         arm_name = self.eef_name
    #     if arm_name in self.grippers:
    #         raise ValueError("Attempts to add multiple grippers to one body")

    #     # self.merge(gripper, merge_body=arm_name)
    #     # self.merge_assets(gripper)

    #     # self.grippers[arm_name] = gripper

    #     # # Update cameras in this model
    #     # self.cameras = self.get_element_names(self.worldbody, "camera")

    # @property
    # def eef_name(self):
    #     """
    #     Returns:
    #         str or dict of str: Prefix-adjusted eef name for this robot. If bimanual robot, returns {"left", "right"}
    #             keyword-mapped eef names
    #     """
    #     return self.correct_naming(self._eef_name)

    # -------------------------------------------------------------------------------------- #
    # Public Properties: In general, these are the name-adjusted versions from the private   #
    #                    attributes pulled from their respective raw xml files               #
    # -------------------------------------------------------------------------------------- #

    # @property
    # def eef_name(self):
    #     """
    #     Returns:
    #         str or dict of str: Prefix-adjusted eef name for this robot. If bimanual robot, returns {"left", "right"}
    #             keyword-mapped eef names
    #     """
    #     return self.correct_naming(self._eef_name)

    # @property
    # def models(self):
    #     """
    #     Returns a list of all m(sub-)models owned by this robot model. By default, this includes the gripper model,
    #     if specified

    #     Returns:
    #         list: models owned by this object
    #     """
    #     models = super().models
    #     return models + list(self.grippers.values())

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
            "right": "right_gripper_mount",
            # "right-tool": "right_tool_mount",
            # "left": "left_gripper_mount",
            # "left-tool": "left_tool_mount"
        }


    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def default_base(self):
        return "NoActuationBase"
        # return "RethinkMount"

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
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 1.0),
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
        return {"right": "default_panda"}

    @property
    def init_qpos(self):
        # Copied from panda.py
        # return np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])
        return np.array([0.0]*6)

    # @property
    # def key_map(self):
    #     return self._key_map
