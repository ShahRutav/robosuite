import os
import h5py
import json

import robosuite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config

from robosuite.wrappers import VisualizationWrapper
from robocasa.utils.env_utils import create_env, run_random_rollouts
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS

def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    return None

def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta

task_name = "PnPCounterToCab"
# robot = "GR1FixedLowerBody"
robot = "PandaMobile"
human_path, ds_meta = get_ds_path(task=task_name, ds_type="human_raw", return_info=True)
use_abs_actions = False
print(human_path, ds_meta)

# robot = "PandaOmron"
env_meta = get_env_metadata_from_dataset(dataset_path=human_path)
if use_abs_actions:
    env_meta["env_kwargs"]["controller_configs"][
        "control_delta"
    ] = False  # absolute action space

env_kwargs = env_meta["env_kwargs"]
env_kwargs["env_name"] = env_meta["env_name"]
env_kwargs["has_renderer"] = True
env_kwargs["renderer"] = "mjviewer"
env_kwargs["has_offscreen_renderer"] = False
env_kwargs["use_camera_obs"] = False
print(env_kwargs["robots"])
env_kwargs["robots"] = [robot]
'''
PandaMobile:
Composite control -
[robosuite INFO] Action Dimensions: [right: 6 dim, right_gripper: 1 dim, base: 3 dim, torso: 1 dim] (robot.py:966)
[robosuite INFO] Action Indices: [right: 0:6, right_gripper: 6:7, base: 7:10, torso: 10:11] (robot.py:969)
Part control -
[robosuite INFO] Action Dimensions: [right: 6 dim, right_gripper: 1 dim, base: 3 dim, torso: 1 dim] (robot.py:966)
[robosuite INFO] Action Indices: [right: 0:6, right_gripper: 6:7, base: 7:10, torso: 10:11] (robot.py:969)
'''
env_kwargs["controller_configs"] = load_composite_controller_config(controller="BASIC", robot=robot) #['body_parts']['right']
# env_kwargs["controller_configs"] = load_part_controller_config(default_controller="OSC_POSE")

# print(colored(f"Initializing environment...", "yellow"))
env = robosuite.make(**env_kwargs,)
# Wrap this with visualization wrapper
env = VisualizationWrapper(env)

f = h5py.File(human_path)
ep = "demo_5"

# Load the human demonstration
demo = f["data"][ep]
ep_meta = json.loads(demo.attrs["ep_meta"])

states = f["data/{}/states".format(ep)][()]
initial_state = dict()
initial_state["states"] = states[0]
initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
reset_to(env, initial_state)
# import ipdb; ipdb.set_trace()

actions = demo["actions"][()]
print(env.action_dim)
# import ipdb; ipdb.set_trace()
for i in range(len(actions)):
    action = actions[i] #[:env.action_dim]
    obs, _, _, _ = env.step(action)
    env.render()
    success = env._check_success()
    if success:
        print("Success")
        break

print("Success" if success else "Failed")

# run rollouts with random actions and save video
# info = run_random_rollouts(
#     env, num_rollouts=1, num_steps=env_meta["env_kwargs"]["horizon"] #, video_path="random_rollout.mp4"
# )
# print(info)
import ipdb; ipdb.set_trace()
