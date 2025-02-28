import os
import time
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R

from robosuite.devices import Device
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import rotation_matrix
from robosuite.controllers.parts.arm.osc import OperationalSpaceController

canonical_quat = True

### Subtractions ###
def quat_diff(target, source):
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat(canonical=canonical_quat)

def rmat_to_quat(rot_mat):
    quat = R.from_matrix(rot_mat).as_quat(canonical=canonical_quat)
    return quat

def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()
    return thread

#### code adapted from https://github.com/AlexanderKhazatsky/R2D2/blob/main/r2d2/controllers/oculus_controller.py ####
class Oculus(Device):
    def __init__(
        self,
        env,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_sensitivity: float = 2.0, # 5
        rot_sensitivity: float = 1.0, # 2
        rmat_reorder: list = [-2, -1, -3, 4],
        *args,
        **kwargs
    ) -> None:
        # lazy import so that we can use the rest of the code without installing oculus_reader
        try:
            import oculus_reader
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "oculus_reader not installed! Please visit https://github.com/rail-berkeley/oculus_reader for installation instrucitons."
            )
        super().__init__(env, *args, **kwargs)
        self.oculus_reader = oculus_reader.OculusReader(run=False)

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.vr_to_global_mat = {'right': np.eye(4), 'left': np.eye(4)}
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        assert pos_sensitivity >= 2.0, "Pose sensitivity must be at least 2.0 for oculus"
        self._reset_state = 0
        self._enabled = False

        self.pos_action_gain = pos_sensitivity
        self.rot_action_gain = rot_sensitivity
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.reset_orientation = {'right': True, 'left': True}
        self.target_gripper = {'right': 0, 'left': 0}
        self._reset_internal_state()


    def start_control(self) -> None:
        print("Starting Oculus Interface...")
        self.oculus_reader.run()
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        run_threaded_command(self._update_internal_state)

    def stop(self) -> None:
        print("Stopping Oculus Interface...")
        self.oculus_reader.stop()

    # set on exit
    def __del__(self):
        # It’s a good idea to wrap cleanup in a try/except
        try:
            self.stop()
        except Exception as e:
            # Optionally log the exception
            print(f"Exception during cleanup: {e}")

    def _reset_internal_state(self) -> None:
        super()._reset_internal_state()
        self._state = {
            'right': {
                "poses": None,
                "movement_enabled": False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
            },

            'left': {
                "poses": None,
                "movement_enabled": False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
            },
            'buttons': {}
        }
        self.update_sensor = {'right': True, 'left': True}
        self.reset_origin = {'right': True, 'left': True}
        self.robot_origin = {'right': None, 'left': None}
        self.vr_origin = {'right': None, 'left': None}
        self.vr_state = {'right': None, 'left': None}

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            if poses == {}:
                continue

            # Determine Control Pipeline #
            for arm in ['left', 'right']:
                button_G = 'RG' if arm=='right' else 'LG'
                button_J = 'RJ' if arm=='right' else 'LJ'

                button_reset = 'B'
                controller_id = 'r' if arm=='right' else 'l'

                if controller_id not in poses:
                    continue
                self._state[arm]["controller_on"] = time_since_read < num_wait_sec

                toggled = self._state[arm]["movement_enabled"] != buttons[button_G]
                self.update_sensor[arm] = self.update_sensor[arm] or buttons[button_G]
                self.reset_orientation[arm] = self.reset_orientation[arm] or buttons[button_J]
                self.reset_origin[arm] = self.reset_origin[arm] or toggled

                # Save Info #
                self._state[arm]["poses"] = poses[controller_id]
                self._state["buttons"] = buttons
                self._state[arm]["movement_enabled"] = buttons[button_G]
                self._state[arm]["controller_on"] = True

                new_gripper = buttons[f"{arm}Trig"][0] > 0.5
                self._state[arm]["gripper_toggle"] = ((not self._state[arm]["prev_gripper"]) and new_gripper) or self._state[arm]["gripper_toggle"]
                self._state[arm]["prev_gripper"] = new_gripper

                last_read_time = time.time()

                stop_updating = self._state["buttons"][button_J] or self._state[arm]["movement_enabled"]
                if self.reset_orientation[arm]:
                    rot_mat = np.asarray(self._state[arm]["poses"])
                    if stop_updating:
                        self.reset_orientation[arm] = False
                    # try to invert the rotation matrix, if not possible, then just use the identity matrix
                    try:
                        rot_mat = np.linalg.inv(rot_mat)
                    except:
                        print(f"exception for rot mat: {rot_mat}")
                        # rot_mat = np.eye(4)
                        rot_mat = np.linalg.pinv(rot_mat)
                        self.reset_orientation[arm] = True
                    self.vr_to_global_mat[arm] = rot_mat

                if buttons[button_reset]:
                    self._reset_state = 1
                    self._enabled = False
                    self._reset_internal_state()

    def _process_reading(self, arm):
        rot_mat = np.asarray(self._state[arm]["poses"])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat[arm] @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        vr_gripper = self._state["buttons"]["rightTrig"][0] if arm=='right' else self._state["buttons"]["leftTrig"][0]
        gripper_toggle = self._state[arm]["gripper_toggle"]
        self._state[arm]["gripper_toggle"] = False

        self.vr_state[arm] = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper, "gripper_toggle": gripper_toggle}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def get_robot_pose(self, robot_id: int, arm: str) -> dict:
        # # read directly from env the robot body pose for right_eef_point
        eef_name = self.env.robots[robot_id].robot_model._eef_name
        if isinstance(eef_name, dict):
            eef_name = eef_name[arm]
        eef_name = f'robot{robot_id}_' + eef_name
        robot_pos = self.env.robots[robot_id].sim.data.get_body_xpos(eef_name)
        robot_quat = self.env.robots[robot_id].sim.data.get_body_xmat(eef_name)
        robot_quat = T.mat2quat(robot_quat)
        if isinstance(self.env.robots[robot_id].part_controllers[arm], OperationalSpaceController):
            ref_frame = self.env.robots[robot_id].part_controllers[arm].input_ref_frame
            if ref_frame == "world":
                pass
            elif ref_frame == "base":
                base_pos, base_ori = self.env.robots[robot_id].composite_controller.get_controller_base_pose(arm)
                base_mat = T.make_pose(base_pos, base_ori)
                pose_in_world = T.pose2mat((robot_pos, robot_quat))
                pose_in_base = np.linalg.inv(base_mat) @ pose_in_world
                robot_pos = pose_in_base[:3, 3]
                robot_quat = T.mat2quat(pose_in_base[:3, :3])
            else:
                raise NotImplementedError(f"Only world and base reference frames are supported. Current reference frame is {ref_frame}")
        else:
            raise NotImplementedError(f"Only OperationalSpaceController is supported for now. Current controller is {type(self.env.robots[robot_id].part_controllers[arm])}")
        return robot_pos, robot_quat

    def get_controller_state(self) -> dict:
        buttons = self._state["buttons"]
        arm = self.active_arm
        if (self._state[arm]["poses"] is None) or (self._state[arm]["movement_enabled"] is False):
            return dict(
                dpos=np.zeros(3),
                rotation=self.rotation,
                raw_drotation=np.zeros(3),
                grasp=self.target_gripper[arm],
                reset=self._reset_state,
                base_mode=int(self.base_mode),
            )

        robot_pos, robot_quat = self.get_robot_pose(self.active_robot, arm=arm)

        if self.update_sensor[arm]:
            self._process_reading(arm)
            self.update_sensor[arm] = False
        if self.reset_origin[arm]:
            self.robot_origin[arm] = {"pos": robot_pos, "quat": robot_quat}
            self.vr_origin[arm] = {"pos": self.vr_state[arm]["pos"], "quat": self.vr_state[arm]["quat"]}
            self.reset_origin[arm] = False

        # this should be w.r.t. to the robot??
        # dpos = self.vr_state[arm]["pos"] - self.vr_origin[arm]["pos"]
        robot_pos_offset = robot_pos - self.robot_origin[arm]["pos"]
        target_pos_offset = self.vr_state[arm]["pos"] - self.vr_origin[arm]["pos"]
        dpos = target_pos_offset - robot_pos_offset

        # robot_quat_offset = T.axisangle2quat(T.quat2axisangle(robot_quat) - T.quat2axisangle(self.robot_origin[arm]["quat"]))
        robot_quat_offset = quat_diff(robot_quat, self.robot_origin[arm]["quat"])
        # target_quat_offset = T.axisangle2quat(np.array([0.0, 0.0, 45*np.pi/180.0]))
        # target_quat_offset = T.axisangle2quat(np.array([0.0, -45*np.pi/180.0, 0.0]))
        # target_quat_offset = T.axisangle2quat(np.array([-45*np.pi/180.0, 0.0, 0.0]))
        target_quat_offset = quat_diff(self.vr_state[arm]["quat"], self.vr_origin[arm]["quat"])
        dquat = quat_diff(target_quat_offset, robot_quat_offset)
        # assert np.allclose(dquat, _dquat, atol=4e-2), f"{dquat} != {_dquat}"
        # print("*"*50)
        # print("Calculating delta robot quat")
        # print(f"{T.quat2axisangle(robot_quat)*180/np.pi} - {T.quat2axisangle(self.robot_origin[arm]['quat'])*180/np.pi} = {T.quat2axisangle(robot_quat_offset)*180/np.pi}")
        # print("Calculating delta vr quat")
        # print(f"{T.quat2axisangle(self.vr_state[arm]['quat'])*180/np.pi} - {T.quat2axisangle(self.vr_origin[arm]['quat'])*180/np.pi} = {T.quat2axisangle(target_quat_offset)*180/np.pi}")
        # print("Calculating dquat")
        # print(f"{T.quat2axisangle(target_quat_offset)*180/np.pi} - {T.quat2axisangle(robot_quat_offset)*180/np.pi} = {T.quat2axisangle(dquat)*180/np.pi}")
        # print("-"*50)


        # convert RPY to an absolute orientation
        euler = T.quat2axisangle(dquat)
        roll, pitch, yaw = euler
        yaw = -yaw # z is filipped inside the device.py
        # print("roll, pitch, yaw", roll, pitch, yaw)
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3))) # Not used. Copied from spacemouse

        self.target_gripper[arm] = buttons[f"{arm}Trig"][0] > 0.5

        # remove rotation control by setting it to default
        # self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # roll, pitch, yaw = np.zeros(3)
        # dpos = np.zeros(3)
        dpos = dpos * self.pos_action_gain
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([pitch, roll, yaw])*self.rot_action_gain,
            grasp=self.target_gripper[arm],
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)
        # print("rotation", drotation*180/np.pi)
        return dpos, drotation
