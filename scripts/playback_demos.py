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

import matplotlib
matplotlib.use('Agg')

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

def reset_to(env, state, replace_robot_joints=True, change_to_gr1=False):
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
        # we need to first update state["model"] to replace the robot tag with the current robot
        curr_xml = env.sim.model.get_xml()
        # if state["model"] != curr_xml:


        # state["model"] = xml

        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        # with open("new_model.xml", "w") as f:
        #     f.write(xml)
        if change_to_gr1:
            xml = replace_robot_tag(new_model_xml=xml, old_model_xml=curr_xml)
        # # save the current model xml
        # with open("current_model.xml", "w") as f:
        #     f.write(curr_xml)
        # with open("updated_model.xml", "w") as f:
        #     f.write(xml)

        env.reset_from_xml_string(xml)
        # env.sim.reset(): resets the robot back to some position which has collision with the table. Change the xml?
        env.sim.reset()
        env.robots[0].reset()
        env.sim.forward()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        if replace_robot_joints:
            print("Replacing robot joints")
            robot_indices = env.robots[0]._ref_joint_pos_indexes
            other_indices = set(range(env.sim.get_state().qpos.flatten().shape[0])) - set(robot_indices)
            other_indices = sorted(list(other_indices))

            non_robot_qpos_idx = state["non_robot_qpos_idx"]
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
        else:
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
    parser.add_argument('--replace-robot-joints', action='store_true')
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")

    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    if env_info["robots"] != args.robot:
        print(f"Overwriting robot model: {env_info['robots']} -> {args.robot}")
        args.replace_robot_joints = True
        args.change_to_gr1 = True
    print(f"[debug] replace_robot_joints: {args.replace_robot_joints}")
    if args.robot is not None:
        env_info["robots"] = [args.robot]
    print(env_info)
    orig_controller_input_type = env_info["controller_configs"]["body_parts"]["right"]["input_type"]

    control_type = "absolute"
    if any(['GR1' in robot for robot in env_info["robots"]]):
        args.controller = "WHOLE_BODY_MINK_IK"

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=env_info["robots"][0],
    )

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
    # env_info["seed"] = 349409
    print(env_info)

    print("Creating environment...")
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
    print(env)

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
        reset_to(env, initial_state, replace_robot_joints=args.replace_robot_joints, change_to_gr1=args.change_to_gr1)

        zero_action = np.zeros(env.action_dim)
        # controller has absolute actions, so we need to set the initial action to be the current position
        active_robot = env.robots[0]
        arm = "right"
        if active_robot.part_controllers[arm].input_type == "absolute":
            zero_action = convert_delta_to_abs_action(zero_action, active_robot, arm, env)
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
