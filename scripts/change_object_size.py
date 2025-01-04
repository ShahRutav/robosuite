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

env = 'Lift'
robot = 'GR1FixedLowerBody'
cu.create_composite_robot(name="CompositeRobot", robot=robot, base=None, grippers="Robotiq140Gripper")
controller_config = load_composite_controller_config(controller="BASIC", robot="CompositeRobot")
env_steps = 500
# controller_config = load_composite_controller_config(controller="BASIC", robot=robot)
config = {
    "env_name": env,
    "robots": ['CompositeRobot'],
    "controller_configs": controller_config,
}

env = suite.make(
    **config,
    has_renderer=False,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_camera_obs=True,
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

action = np.zeros_like(env.action_spec[0])
action = np.clip(action, low, high)
obs, reward, done, _ = env.step(action)

img = obs['agentview_image']
plt.imshow(img[::-1])
plt.savefig('temp.png')
print(img.shape)
