import os
import time
import pickle
import argparse
from typing import Dict, List, Union
import matplotlib.pyplot as plt

import numpy as np

import robosuite as suite
from robosuite.utils import robot_composition_utils as cu
from robosuite.utils import transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.robots import is_robosuite_robot
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robot", type=str, default="GR1FixedLowerBody")
    args = parser.parse_args()
    robot = args.robot
    env_steps = 500
    controller_config = load_composite_controller_config(controller="BASIC", robot=robot)
    gt_traj_file = 'random_traj.pkl'
    gt_traj = pickle.load(open(gt_traj_file, 'rb'))
    config = {
        "env_name": args.env,
        "robots": [robot],
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        render_camera="free",
        control_freq=20,
    )
    for robot in env.robots:
        robot.print_action_info_dict()

    env.reset()
    # env.initialize_renderer()
    low, high = env.action_spec
    low = np.clip(low, -1, 1)
    high = np.clip(high, -1, 1)
    avg_pos_error = 0.0
    avg_quat_error = 0.
    pos_error_list = []
    quat_error_list = []
    # Runs a few steps of the simulation as a sanity check
    for i in range(env_steps):
        action = np.zeros_like(env.action_spec[0])
        print(action.shape)
        eps = 1e-1
        action[2] += eps
        action[0] += eps
        # action[1] += eps
        action = np.clip(action, low, high)
        obs = env._get_observations()
        robot_pos = obs['robot0_eef_pos'] if 'robot0_eef_pos' in obs else obs['robot0_right_eef_pos']
        robot_quat = obs['robot0_eef_quat'] if 'robot0_eef_quat' in obs else obs['robot0_right_eef_quat']
        pos_error = np.linalg.norm(robot_pos - gt_traj['robot_pos'][i])
        quat_error = np.linalg.norm(T.quat2axisangle(robot_quat) - T.quat2axisangle(gt_traj['robot_quat'][i]))
        pos_error_list.append(pos_error)
        quat_error_list.append(quat_error)
        avg_pos_error += pos_error
        avg_quat_error += quat_error
        print(f"Running Average Pos Error: {avg_pos_error / (i + 1)}")
        print(f"Running Average Quat Error: {(avg_quat_error / (i + 1))*180/np.pi}")

        # print(action)
        obs, reward, done, _ = env.step(action)
        if i == 200:
            break
        # if True:
        #     env.render()
        #     time.sleep(0.1)


    print(pos_error_list)
    plt.plot(pos_error_list)
    plt.savefig('pos_error.png')

    # import ipdb; ipdb.set_trace()
    env.close()
