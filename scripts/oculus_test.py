import os
import yaml
import time
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix
from robosuite.devices.oculus_base import TeleopAction, TeleopObservation, run_threaded_command

from copy import deepcopy
import robosuite as suite
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.controllers import load_composite_controller_config
from robosuite.devices.spacemouse import SpaceMouse
from robosuite.devices.oculus import Oculus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="Demo")
    parser.add_argument("--device", type=str, default="oculus")
    args = parser.parse_args()
    robot = args.robot
    # robot = "GR1FixedLowerBody"
    controller_config = load_composite_controller_config(controller="BASIC", robot=robot)
    controller_config['body_parts']['right']['input_ref_frame'] = "world"


    # controller_config['body_parts']['right']['input_type'] = "absolute"
    # controller_config['body_parts']['left']['input_type'] = "absolute"
    assert controller_config['body_parts']['right']['input_type'] == "delta"
    assert controller_config['body_parts']['left']['input_type'] == "delta"
    print(controller_config)
    # dump the confil in temp.yaml
    # with open("temp.yaml", "w") as f:
    #     f.write(str(controller_config))
    # read temp.yaml and compare the content with controller_config
    dumped_config = yaml.load(open("temp.yaml", "r"), Loader=yaml.FullLoader)
    # make a recursive comparison
    def compare_dict(d1, d2):
        for k in d1:
            if k not in d2:
                print(f"{k} not in d2")
                return False
            if isinstance(d1[k], dict):
                if not compare_dict(d1[k], d2[k]):
                    return False
            else:
                if d1[k] is None:
                    # return True
                    if (d2[k] is None) or (d2[k] == "None"):
                        return True
                    else:
                        print(f"{k}: {d1[k]} != {d2[k]}")
                        print(f"types: {type(d1[k])}, {type(d2[k])}")
                        return False
                if d1[k] != d2[k]:
                    print(f"{k}: {d1[k]} != {d2[k]}")
                    return False
        return True
    # print(compare_dict(controller_config, dumped_config))


    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots=robot,  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        controller_configs=[controller_config],
        horizon=10000,
        render_camera="free",
    )
    active_robot = env.robots[0]
    # controller_inputs_types = [active_robot.part_controllers[arm].input_type for arm in active_robot.arms]
    controller_inputs_types_dict = {}
    for arm in active_robot.arms:
        if isinstance(active_robot.composite_controller, WholeBody):
            controller_inputs_types_dict[arm] = active_robot.composite_controller.joint_action_device.input_type
        else:
            controller_inputs_types_dict[arm] = active_robot.part_controllers[arm].input_type

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]
    fq = 10 # hertz
    device = None
    if args.device == "oculus":
        device = Oculus(env)
    elif args.device == "spacemouse":
        device = SpaceMouse(env)
    else:
        raise ValueError
    device.start_control()
    while True:
        # obs = {'right': np.array([0, 0, 0, 1, 0, 0, 0, 0.0]), 'left': np.array([0, 0, 0, 1, 0, 0, 0, 0.0])}
        # action = device.get_action(obs)
        input_ac_dict = device.input2action()
        # print(input_ac_dict)
        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm, controller_input_type in controller_inputs_types_dict.items():
            print(controller_input_type, arm)
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()
        # print(env_action)
        time.sleep(1/fq)
    device.stop()
