import os
import time
import pickle
import argparse
from typing import Dict, List, Union

import numpy as np

import robosuite as suite
from robosuite.utils import robot_composition_utils as cu
from robosuite.utils import transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.robots import is_robosuite_robot
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

# If you would like to visualize the scene during testing,
# set render to True and increase env_steps to a larger value.
def create_and_test_env(
    env: str,
    robots: Union[str, List[str]],
    controller_config: Dict,
    render: bool = True,
    env_steps: int = 20,
):

    config = {
        "env_name": env,
        "robots": robots,
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=render,
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
    random_traj = {'robot_pos': [], 'robot_quat': [], 'action': []}

    # Runs a few steps of the simulation as a sanity check
    for i in range(env_steps):
        # read the current site position of the robot
        # site_pos = env.sim.data.get_site_xpos('robot0_right_center')
        # site_quat = env.sim.data.get_site_xmat('robot0_right_center')
        # site_quat = T.mat2quat(site_quat)
        # print(site_pos, site_quat)
        # action
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
        random_traj['robot_pos'].append(robot_pos)
        random_traj['robot_quat'].append(robot_quat)
        random_traj['action'].append(action)

        # print(action)
        obs, reward, done, _ = env.step(action)
        # print(obs.keys())
        # print("robot0_floating_base", env.sim.data.get_body_xpos('robot0_floating_base'))
        if 'robot0_right_eef_pos' in obs:
            print(obs['robot0_right_eef_pos'], obs['robot0_right_eef_quat'])
        else:
            print(obs['robot0_eef_pos'], obs['robot0_eef_quat'])
        # if 'robot0_base_to_right_eef_pos' in obs:
        #     print(obs['robot0_base_to_right_eef_pos'], obs['robot0_base_to_right_eef_quat'])
        # else:
        #     print(obs['robot0_base_to_eef_pos'], obs['robot0_base_to_eef_quat'])
        # print(env.robots[0].print_action_info_dict())
        # if render:
        #     env.render()
        #     time.sleep(0.1)
        # break
            # import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    env.close()
    # pickle.dump(random_traj, open('random_traj.pkl', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robot", type=str, default="GR1TwoFingered")
    args = parser.parse_args()

    # DEMO: site_xpos, site_xquat:
    # [-0.56  0.    0.4 ] [0. 0. 0. 1.]
    # [-0.56  0.    1.3 ] [0. 0. 0. 1.]
    # GR1FixedLowerBody: site_xpos, site_xquat
    # [-0.53732824 -0.12481552  1.31455478] [-0.22058149  0.02277821  0.00119464  0.9751018 ]


    # DEMO: robot0_eef_pos, and quat:
    # [-0.29077481 -0.22787878  1.15196316] [-0.49898058  0.50647355  0.50290454 -0.491518  ]
    # [-0.3002958  -0.27911352  1.10181674] [-0.50778781  0.4910338   0.49940602 -0.50162832]
    # GRI: robot0_right_eef_pos, and quat:
    # [-0.29216848 -0.24774411  1.11376383] [-0.49988755  0.5297464   0.45484694 -0.51244068]
    robot = args.robot
    if robot == 'composite':
        robot = 'CompositeRobot'
    '''
    GR1FixedLowerBody:
    [-0.31575153 -0.26280374  1.10124089] [-0.45840404  0.53991574  0.48533814 -0.51264375]

    Demo
    [-0.25518059 -0.02546814  0.875745  ] [-0.49480522  0.52002029  0.50887186 -0.4751801 ]
    [-0.29136835 -0.25409592  1.10386265] [-0.51894727  0.50107021  0.48781047 -0.49159264]
    '''
    if robot == 'CompositeRobot':
        cu.create_composite_robot(name="CompositeRobot", robot='GR1TwoFingered', base=None, grippers="Robotiq140Gripper")
    controller_config = load_composite_controller_config(controller="BASIC", robot=robot)
    print(controller_config)
    create_and_test_env(env="Lift", robots=robot, controller_config=controller_config, render=True, env_steps=500)
