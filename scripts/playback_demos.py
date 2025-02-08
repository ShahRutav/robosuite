"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random
from tqdm import tqdm

import h5py
import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T
from robosuite.utils.binding_utils import MjSimState
from robosuite.utils.control_utils import convert_delta_to_abs_action

import robocasa
import robocasa.macros as macros
from robocasa.models.fixtures import FixtureType
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format

import matplotlib
matplotlib.use('Agg')

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
        '''
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        '''
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        # get the robot state index from the flattened state
        robot_indices = env.robots[0]._ref_joint_pos_indexes
        other_indices = set(range(env.sim.get_state().qpos.flatten().shape[0])) - set(robot_indices)
        other_indices = sorted(list(other_indices))

        non_robot_qpos_idx = state["non_robot_qpos_idx"]
        # if len(non_robot_qpos_idx) != len(other_indices):

        # zero_action = np.zeros(env.action_dim)
        # # controller has absolute actions, so we need to set the initial action to be the current position
        # active_robot = env.robots[0]
        # arm = "right"
        # if active_robot.part_controllers[arm].input_type == "absolute":
        #     zero_action = convert_delta_to_abs_action(zero_action, active_robot, arm, env)

        if len(non_robot_qpos_idx) > len(other_indices):
            # keep the last len(other_indices) elements since some robot indices might have been removed
            non_robot_qpos_idx = non_robot_qpos_idx[-len(other_indices):]
        if len(non_robot_qpos_idx) < len(other_indices):
            # keep the last len(non_robot_qpos_idx) elements since some robot indices might have been removed
            other_indices = other_indices[-len(non_robot_qpos_idx):]
        assert len(non_robot_qpos_idx) == len(other_indices), f"Mismatch in non_robot_qpos_idx: {len(non_robot_qpos_idx)} != {len(other_indices)}"
        qpos_state = state["qpos_state"]

        time = env.sim.data.time
        qvel = env.sim.get_state().qvel.flatten().copy()
        qpos = env.sim.get_state().qpos.flatten().copy()
        # copy over everything except the robot state
        for curr_idx, state_idx in zip(other_indices, non_robot_qpos_idx):
            qpos[curr_idx] = qpos_state[state_idx]
        curr_state = MjSimState(qpos=qpos, qvel=qvel, time=time)
        env.sim.set_state_from_flattened(curr_state.flatten())
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="If provided, will overwrite the robot model specified in the demo file",
    )
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")

    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    if args.robot is not None:
        env_info["robots"] = [args.robot]
    print(env_info)

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=env_info["robots"][0],
    )
    control_type = "absolute"
    # controller_config['body_parts']['right']['input'] = "OSC_POSE"
    controller_config['body_parts']['right']['input_type'] = control_type
    controller_config['body_parts']['right']['input_ref_frame'] = "world"
    '''
    controller_config['body_parts']['right']['kp'] = 5000
    controller_config['body_parts']['right']['kd'] *= 4
    # controller_config['body_parts']['right']['ki'] *= 2
    controller_config['body_parts']['right']['kp_limits'] = [0, 10000]
    '''
    env_info["controller_configs"] = controller_config
    print(env_info)

    env = robosuite.make(
        **env_info,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        # control_freq=30,
        renderer="mjviewer",
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    # cube_pos_errors = []
    # cube_quat_errors = []
    success_rate = 0.0
    total_index = 0

    for ep in tqdm(demos):
        print("Playing back random episode... (press ESC to quit)")

        # cube_pos_err = []
        # cube_quat_err = []
        # select an episode randomly
        # ep = random.choice(demos)
        if args.render:
            env._destroy_viewer()
        env._destroy_sim()
        env._load_model()
        env._initialize_sim()

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        initial_state = {}
        initial_state["states"] = states[0]
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
        initial_state["non_robot_qpos_idx"] = f["data/{}".format(ep)].attrs.get("non_robot_qpos_idx", None)
        initial_state["qpos_state"] = f["data/{}".format(ep)].attrs.get("initial_qpos", None)

        # ep_meta = {'layout_id': 6, 'style_id': 5}
        # ep_meta = json.loads(f["data/{}".format(ep)].attrs.get("ep_meta"))
        # env.set_ep_meta(ep_meta)
        # env.reset()
        reset_to(env, initial_state)

        zero_action = np.zeros(env.action_dim)
        # controller has absolute actions, so we need to set the initial action to be the current position
        active_robot = env.robots[0]
        arm = "right"
        if active_robot.part_controllers[arm].input_type == "absolute":
            zero_action = convert_delta_to_abs_action(zero_action, active_robot, arm, env)
        env.step(zero_action)
        # import ipdb; ipdb.set_trace()

        '''
        env.reset()
        # xml = env.edit_model_xml(model_xml)
        # env.reset_from_xml_string(xml)
        # env.sim.reset()
        # env.viewer.set_camera(0)
        env.reset()
        '''

        is_success = False

        if args.use_actions:

            # # load the initial state
            # env.sim.set_state_from_flattened(states[0])
            # env.sim.forward()
            gt_obs = f["data/{}/observations".format(ep)]
            obs = env._get_observations()
            if 'obj_pos' in obs:
                print(f"Diff: {obs['obj_pos']-gt_obs['obj_pos'][0]}")

            # obj = env.model.mujoco_objects[0]
            # env.sim.data.set_joint_qpos(
            #     obj.joints[0],
            #     np.concatenate([gt_obs['cube_pos'][0], gt_obs['cube_quat'][0]]),
            # )
            '''
            initial_obj_qpos = f["data/{}/initial_obj_qpos".format(ep)]
            joint_list = [obj.joints for obj in env.model.mujoco_objects]
            for _list in joint_list:
                for joint in _list:
                    # if joint not in initial_obj_qpos:
                    #     print(f"[WARNING!!] Joint {joint} not found in the initial_obj_qpos")
                    #     continue
                    print(joint, initial_obj_qpos[joint][()])
                    # try:
                    env.sim.data.set_joint_qpos(
                        joint,
                        initial_obj_qpos[joint][()]
                    )
                    # except:
                    #     print(f"[WARNING!!] Joint {joint} not found in the model")
                    #     pass
            env.sim.forward()
            env._check_success()
            env._update_observables(force=True)
            '''
            # for _ in range(10):
            #     env.sim.step1()
            #     env.sim.step2()
            #     env.sim.forward()
            #     env._update_observables()
            #     env.render()

            # env.sim.forward()
            # print(env._get_observations()['cube_pos']-gt_obs['cube_pos'][0])
            # cube_pos_err.append(np.linalg.norm(env._get_observations()['cube_pos']-gt_obs['cube_pos'][0]))
            # cube_quat_err.append(
            #     np.linalg.norm(
            #         T.quat2axisangle(env._get_observations()['cube_quat'])-T.quat2axisangle(gt_obs['cube_quat'][0])
            #     )
            # )

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            obs = env._get_observations()
            for j, action in enumerate(actions):
                # import ipdb; ipdb.set_trace()
                eef_pos = obs['robot0_eef_pos']
                eef_quat = obs['robot0_eef_quat']
                delta_pos = action[:3] - eef_pos

                eef_mat = T.quat2mat(eef_quat)
                action_mat = T.quat2mat(T.axisangle2quat(action[3:6]))
                delta_mat = action_mat @ eef_mat.T
                delta_quat = T.mat2quat(delta_mat)
                if np.dot(delta_quat, eef_quat) < 0:
                    delta_quat = -delta_quat

                if control_type == "delta":
                    action = np.concatenate([delta_pos, T.quat2axisangle(delta_quat), action[6:]])

                _action = None
                if args.robot is not None:
                    if len(action) == 7:
                        action_dict = {
                            'right': action[:6],
                            'right_gripper': action[6:7],
                            'left': np.zeros(6),
                            'left_gripper': np.zeros(1),
                            'base_mode': -1,
                            'base': np.zeros(3),
                        }
                    elif len(action) == 14:
                        action_dict = {
                            'right': action[:6],
                            'right_gripper': action[12:13],
                            'left': np.zeros(6),
                            'left_gripper': np.zeros(1),
                            'base_mode': -1,
                            'base': np.zeros(3),
                        }
                    else:
                        raise ValueError(f"Invalid action length: {len(action)}")
                    _action = env.robots[0].create_action_vector(action_dict)
                    # assert np.allclose(_action, action), f"Action mismatch: {_action} != {action}"
                else:
                    _action = action
                obs, _, _, _ = env.step(_action)
                # cube_pos_err.append(np.linalg.norm(obs['cube_pos']-gt_obs['cube_pos'][j]))
                # cube_quat_err.append(np.linalg.norm(T.quat2axisangle(obs['cube_quat'])-T.quat2axisangle(gt_obs['cube_quat'][j])))

                # print(gt_obs['cube_pos'][j]-obs['cube_pos'])

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                if j == num_actions - 1:
                    is_success = env._check_success()
                    print("Success:", env._check_success())
            # cube_pos_errors.append(cube_pos_err)
            # cube_quat_errors.append(cube_quat_err)
        else:
            is_success = True
            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if env.renderer == "mjviewer":
                    env.viewer.update()
                env.render()
        total_index += 1
        success_rate += int(is_success)

    # plot the errors with line i corresponding to episode i with  a shade of blue
    # import matplotlib.pyplot as plt
    # for i, (pos_err, quat_err) in enumerate(zip(cube_pos_errors, cube_quat_errors)):
    #     plt.plot(pos_err, label=f"Episode {i} Pos Error")
    # plt.legend()
    # plt.savefig(f"cube_pos_errors_{env_info['robots'][0]}.png")

    f.close()
    # # print the average error for each episode
    # print("Average pos error per episode:")
    # for i, pos_err in enumerate(cube_pos_errors):
    #     print(f"Episode {i}: {np.mean(pos_err):.4f}")
    # print("Average quat error per episode:")
    # for i, quat_err in enumerate(cube_quat_errors):
    #     print(f"Episode {i}: {np.mean(quat_err):.4f}")

    # print(f"Average pos error: {np.mean([np.mean(err) for err in cube_pos_errors]):.4f}")
    # print(f"Average quat error: {np.mean([np.mean(err) for err in cube_quat_errors]):.4f}")
    print(f"Success rate: {success_rate / total_index:.3f}")
    env.close()

'''
'''
