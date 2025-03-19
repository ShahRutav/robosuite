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
import numpy as np

import h5py
import numpy as np
import xml.etree.ElementTree as ET

import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T
from robosuite.utils.binding_utils import MjSimState
from robosuite.utils.control_utils import convert_delta_to_abs_action

import robocasa
import robocasa.macros as macros
from robocasa.models.fixtures import FixtureType
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format

from icrt.util.casa_utils import reset_to, get_zero_action, load_controller_config

import matplotlib
matplotlib.use('Agg')

def get_joint_idx_from_name(env, joint_name):
    joint_names = [name for name in env.sim.model.joint_names if joint_name in name]
    indices = []
    for joint_name in joint_names:
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, (int, np.int32, np.int64)):
            indices.append(addr)
        else:
            start, end = addr
            indices.extend(list(range(start, end)))
    return indices

def fix_qpos_state(env, val_qpos_state, train_qpos_state):
    imp_joint_names = ["spout", "vegetable", "plate", "vegetable_container", "packed_food"]
    imp_joint_indices = set()
    for joint_name in imp_joint_names:
        joint_qpos_indices = get_joint_idx_from_name(env, joint_name)
        print(f"Joint name: {joint_name}, joint index: {list(joint_qpos_indices)}")
        for idx in joint_qpos_indices:
            imp_joint_indices.add(idx)
    imp_joint_indices = sorted(list(imp_joint_indices))
    train_qpos_state[imp_joint_indices] = val_qpos_state[imp_joint_indices].copy()
    val_qpos_state = train_qpos_state
    return val_qpos_state

# Function to replace robot tags in the XML
def replace_robot_tag(new_model_xml, old_model_xml):
    """
    new_model_xml: str (XML content of the new model)
    old_model_xml: str (XML content of the old model)

    1. Delete all tags starting with 'robot0' in the <actuator> and <asset> tags in the new model.
    2. Add all 'robot0' related tags from the old model into the new model in <actuator> and <asset>.
    3. Replace 'robot0_floating_base' body tag with 'robot0_base' from the old model (including its content).

    Returns:
        str: Modified XML content as a string.
    """

    # Parse XML content
    new_root = ET.fromstring(new_model_xml)
    old_root = ET.fromstring(old_model_xml)

    # Function to filter and transfer robot0 elements
    def transfer_robot0_elements(new_parent, old_parent):
        # Remove existing robot0 elements in new_parent
        if new_parent is not None:
            for elem in list(new_parent):
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.remove(elem)

        if old_parent is not None:
            if new_parent is None:
                # add a new parent element if it doesn't exist to the
                # Add robot0 elements from old_parent to new_parent as a child of <mujoco> tag
                new_parent = ET.SubElement(new_root, old_parent.tag)
            for elem in old_parent:
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.append(elem)

    # Update <actuator> elements
    new_actuator = new_root.find("actuator")
    old_actuator = old_root.find("actuator")
    transfer_robot0_elements(new_actuator, old_actuator)

    # Update <asset> elements
    new_asset = new_root.find("asset")
    old_asset = old_root.find("asset")
    transfer_robot0_elements(new_asset, old_asset)

    new_contact = new_root.find("contact")
    old_contact = old_root.find("contact")
    transfer_robot0_elements(new_contact, old_contact)

    new_robot0_base = None
    for body in new_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_floating_base":
            new_robot0_base = body
            break

    # Locate "robot0_floating_base" in old model
    old_robot0_floating_base = None
    for body in old_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_base":
            old_robot0_floating_base = body
            break

    assert new_robot0_base is not None, "robot0_floating_base must exist in new model"
    assert old_robot0_floating_base is not None, "robot0_base must exist in old model"
    # Replace the contents of "robot0_base" with "robot0_floating_base" if both exist
    if new_robot0_base is not None and old_robot0_floating_base is not None:
        # Clear the current children of new_robot0_base
        new_robot0_base.clear()

        # Copy all attributes from old_robot0_floating_base to new_robot0_base
        new_robot0_base.attrib = old_robot0_floating_base.attrib

        # Copy all child elements
        for elem in old_robot0_floating_base:
            new_robot0_base.append(elem)

    # copy the geom name from old model to new model named robot0_floor in the worldbody tag
    # we want to put it in the same location in the new model
    new_worldbody = new_root.find("worldbody")
    old_worldbody = old_root.find("worldbody")
    for geom in old_worldbody.findall(".//geom"):
        if geom.attrib.get("name") == "robot0_floor":
            index = list(old_worldbody).index(geom)
            new_worldbody.insert(index, geom)

    # Convert back to string
    return ET.tostring(new_root, encoding="unicode")

# def reset_to(env, state, replace_robot_joints=True, change_to_gr1=False):
#     """
#     Reset to a specific simulator state.

#     Args:
#         state (dict): current simulator state that contains one or more of:
#             - states (np.ndarray): initial state of the mujoco environment
#             - model (str): mujoco scene xml

#     Returns:
#         observation (dict): observation dictionary after setting the simulator state (only
#             if "states" is in @state)
#     """
#     should_ret = False
#     if "model" in state:
#         if state.get("ep_meta", None) is not None:
#             # set relevant episode information
#             ep_meta = json.loads(state["ep_meta"])
#         else:
#             ep_meta = {}
#         if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
#             env.set_attrs_from_ep_meta(ep_meta)
#         elif hasattr(env, "set_ep_meta"):  # newer versions
#             env.set_ep_meta(ep_meta)
#         # this reset is necessary.
#         # while the call to env.reset_from_xml_string does call reset,
#         # that is only a "soft" reset that doesn't actually reload the model.
#         env.reset()
#         robosuite_version_id = int(robosuite.__version__.split(".")[1])
#         # we need to first update state["model"] to replace the robot tag with the current robot
#         curr_xml = env.sim.model.get_xml()
#         # if state["model"] != curr_xml:


#         # state["model"] = xml

#         if robosuite_version_id <= 3:
#             from robosuite.utils.mjcf_utils import postprocess_model_xml

#             xml = postprocess_model_xml(state["model"])
#         else:
#             # v1.4 and above use the class-based edit_model_xml function
#             xml = env.edit_model_xml(state["model"])

#         # with open("new_model.xml", "w") as f:
#         #     f.write(xml)
#         if change_to_gr1:
#             xml = replace_robot_tag(new_model_xml=xml, old_model_xml=curr_xml)
#         # # save the current model xml
#         # with open("current_model.xml", "w") as f:
#         #     f.write(curr_xml)
#         # with open("updated_model.xml", "w") as f:
#         #     f.write(xml)

#         env.reset_from_xml_string(xml)
#         # env.sim.reset(): resets the robot back to some position which has collision with the table. Change the xml?
#         env.sim.reset()
#         env.robots[0].reset()
#         env.sim.forward()
#         # hide teleop visualization after restoring from model
#         # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
#         # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
#     if "states" in state:
#         if replace_robot_joints:
#             print("Replacing robot joints")
#             robot_indices = env.robots[0]._ref_joint_pos_indexes
#             other_indices = set(range(env.sim.get_state().qpos.flatten().shape[0])) - set(robot_indices)
#             other_indices = sorted(list(other_indices))

#             non_robot_qpos_idx = state["non_robot_qpos_idx"]
#             assert len(non_robot_qpos_idx) == len(other_indices), f"Mismatch in non_robot_qpos_idx: {len(non_robot_qpos_idx)} != {len(other_indices)}"
#             qpos_state = state["qpos_state"]
#             time = env.sim.data.time
#             qvel = env.sim.get_state().qvel.flatten().copy()
#             qpos = env.sim.get_state().qpos.flatten().copy()
#             # copy over everything except the robot state
#             for curr_idx, state_idx in zip(other_indices, non_robot_qpos_idx):
#                 qpos[curr_idx] = qpos_state[state_idx]
#             curr_state = MjSimState(qpos=qpos, qvel=qvel, time=time)
#             env.sim.set_state_from_flattened(curr_state.flatten())
#             env.sim.forward()
#             should_ret = True
#         else:
#             env.sim.set_state_from_flattened(state["states"])
#             env.sim.forward()
#             should_ret = True

#     # update state as needed
#     if hasattr(env, "update_sites"):
#         # older versions of environment had update_sites function
#         env.update_sites()
#     if hasattr(env, "update_state"):
#         # later versions renamed this to update_state
#         env.update_state()

#     # if should_ret:
#     #     # only return obs if we've done a forward call - otherwise the observations will be garbage
#     #     return get_observation()
#     return None

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
    parser.add_argument('--replace-robot-joints', action='store_true')
    parser.add_argument('--ref_frame', type=str, default="world")
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    other_hdf5_path = hdf5_path.replace("val_old", "train_old") # HACk

    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    args.change_to_gr1 = False
    if env_info["robots"] != args.robot:
        print(f"Overwriting robot model: {env_info['robots']} -> {args.robot}")
        args.replace_robot_joints = True
        args.change_to_gr1 = True
    print(f"[debug] replace_robot_joints: {args.replace_robot_joints}")
    if args.robot is not None:
        env_info["robots"] = [args.robot]
    # print(env_info)
    orig_controller_input_type = env_info["controller_configs"]["body_parts"]["right"]["input_type"]

    control_type = "absolute"
    ref_frame = args.ref_frame
    if any(['GR1' in robot for robot in env_info["robots"]]):
        args.controller = "WHOLE_BODY_MINK_IK"

    controller_config = load_controller_config(args.controller, env_info["robots"][0], control_type, ref_frame)
    args.overwrite_ref_frame = None
    if env_info["controller_configs"]["body_parts"]["right"]["input_ref_frame"] != ref_frame:
        print(f"Overwriting ref_frame: {env_info['controller_configs']['body_parts']['right']['input_ref_frame']} -> {ref_frame}")
        args.overwrite_ref_frame = f"{env_info['controller_configs']['body_parts']['right']['input_ref_frame']}_{ref_frame}"
    env_info["controller_configs"] = controller_config
    # env_info["seed"] = 349409
    print(env_info)

    print("Creating environment...")
    first_ep_key = list(f["data"].keys())[0]
    ep_meta = f["data/{}".format(first_ep_key)].attrs.get("ep_meta", None)
    env_info["ep_meta"] = json.loads(ep_meta) if ep_meta is not None else None
    env = robosuite.make(
        **env_info,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        camera_names=["agentview_left"],
        renderer="mjviewer",
    )
    # assert ref_frame == env.robots[0].composite_controller.composite_controller_specific_config.get("ik_input_ref_frame"), f"Ref frame mismatch: {ref_frame} != {env.robots[0].composite_controller.composite_controller_specific_config.get('ik_input_ref_frame')}"

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


        # get the qpos_state from the other hdf5 file
        f2 = h5py.File(other_hdf5_path, "r")
        other_ep = list(f2["data"].keys())[0] if "data" in f2 else list(f2.keys())[0]
        other_qpos_state = f2["data/{}".format(other_ep)].attrs.get("initial_qpos", None)
        initial_state["qpos_state"] = fix_qpos_state(env, initial_state["qpos_state"].copy(), other_qpos_state.copy())
        sim_state = MjSimState.from_flattened(initial_state["states"], env.sim)
        sim_state.qpos = fix_qpos_state(env, sim_state.qpos.copy(), other_qpos_state.copy())
        initial_state["states"] = sim_state.flatten()
        # non-zero
        # (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       # 18, 19, 35, 36, 37, 38, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
       # 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
       # 71]),)


        # ep_meta = {'layout_id': 6, 'style_id': 5}
        # ep_meta = json.loads(f["data/{}".format(ep)].attrs.get("ep_meta"))
        # env.set_ep_meta(ep_meta)
        if env_name == "Lift":
            env.reset()
            if not args.change_to_gr1:
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()
        else:
            reset_to(env, initial_state, replace_robot_joints=args.replace_robot_joints, change_to_gr1=args.change_to_gr1)

        # controller has absolute actions, so we need to set the initial action to be the current position
        active_robot = env.robots[0]

        zero_action = get_zero_action(env)
        env.step(zero_action)

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
            # cube_quae_err.append(
            #     np.linalg.norm(
            #         T.quat2axisangle(env._get_observations()['cube_quat'])-T.quat2axisangle(gt_obs['cube_quat'][0])
            #     )
            # )

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            obs = env._get_observations()
            for j, action in enumerate(actions):
                # eef_pos = obs['robot0_eef_pos']
                # eef_quat = obs['robot0_eef_quat']
                # delta_pos = action[:3] - eef_pos

                _action = None
                if orig_controller_input_type != control_type:
                    assert orig_controller_input_type == "delta", f"Invalid controller input type: {orig_controller_input_type}"
                    delta_arm_action = action.copy()
                    delta_arm_action[:6] = env.robots[0].part_controllers[arm].scale_action(delta_arm_action[:6])
                    action = convert_delta_to_abs_action(delta_arm_action, env.robots[0], arm, env)

                if args.overwrite_ref_frame is not None:
                    assert orig_controller_input_type == control_type, f"mismatch between controller input type: {orig_controller_input_type} != {control_type} along with ref_frame: {args.overwrite_ref_frame}"
                    if args.overwrite_ref_frame == "world_base":
                        if active_robot.composite_controller_config["type"] in ["WHOLE_BODY_IK", "WHOLE_BODY_MINK_IK"]:
                            action_mat = T.make_pose(action[:3], T.quat2mat(T.axisangle2quat(action[3:6])))
                            action_h =  env.robots[0].composite_controller.joint_action_policy.transform_pose(action_mat, 'world', 'base')
                            # base_pos, base_ori = env.robots[0].sim.data.get_body_xpos("robot0_base"), env.robots[0].sim.data.get_body_xquat("robot0_base")
                            # # convert base_ori from quaternion wxyz to xyzw
                            # base_ori = T.convert_quat(base_ori, to="xyzw")
                            # base_mat = T.make_pose(base_pos, T.quat2mat(base_ori))
                        elif hasattr(active_robot.part_controllers[arm], "delta_to_abs_action"):
                            # # take the controller base
                            base_pos, base_ori = env.robots[0].composite_controller.get_controller_base_pose('right')
                            base_mat = T.make_pose(base_pos, base_ori)
                            action_trans = action[:3]
                            action_mat = T.quat2mat(T.axisangle2quat(action[3:6]))
                            action_h = T.make_pose(action_trans, action_mat)
                            action_h = np.dot(np.linalg.inv(base_mat), action_h)
                        else:
                            raise NotImplementedError(f"Controller type {active_robot.composite_controller_config['type']} not implemented")

                        action_rot = T.quat2axisangle(T.mat2quat(action_h[:3, :3]))
                        action_trans = action_h[:3, 3]
                        action = np.concatenate((action_trans, action_rot, action[6:]), axis=-1)
                    elif args.overwrite_ref_frame == "base_world":
                        if active_robot.composite_controller_config["type"] in ["WHOLE_BODY_IK", "WHOLE_BODY_MINK_IK"]:
                            transform_matrix = env.robots[0].composite_controller.joint_action_policy.frame_transform_matrix('base', 'world')
                            action_mat1 = T.make_pose(action[:3], T.quat2mat(T.axisangle2quat(action[3:6])))
                            action_h1 = transform_matrix @ action_mat1
                            action_h2 = None
                            if action.shape[0] == 14:
                                action_mat2 = T.make_pose(action[6:9], T.quat2mat(T.axisangle2quat(action[9:12])))
                                action_h2 = transform_matrix @ action_mat2
                        else:
                            raise NotImplementedError(f"Controller type {active_robot.composite_controller_config['type']} not implemented")
                        action_rot1 = T.quat2axisangle(T.mat2quat(action_h1[:3, :3]))
                        action_trans1 = action_h1[:3, 3]
                        action1 = np.concatenate((action_trans1, action_rot1))
                        if action_h2 is not None:
                            action_rot2 = T.quat2axisangle(T.mat2quat(action_h2[:3, :3]))
                            action_trans2 = action_h2[:3, 3]
                            action2 = np.concatenate((action_trans2, action_rot2))
                            action = np.concatenate((action1, action2, action[12:]), axis=-1)
                        else:
                            action = np.concatenate((action1, action[6:]), axis=-1)
                    else:
                        raise NotImplementedError(f"Ref frame {args.overwrite_ref_frame} not implemented")
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
