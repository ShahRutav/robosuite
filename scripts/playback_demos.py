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

import matplotlib
matplotlib.use('Agg')

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
    import ipdb; ipdb.set_trace()

    controller_config = load_composite_controller_config(
        # controller=None,
        controller="WHOLE_BODY_IK",
        robot=env_info["robots"][0],
    )
    controller_config['body_parts']['right']['input_type'] = "absolute"
    controller_config['body_parts']['right']['input_ref_frame'] = "world"
    controller_config['body_parts']['right']['kp'] = 5000
    controller_config['body_parts']['right']['kd'] *= 4
    # controller_config['body_parts']['right']['ki'] *= 2
    controller_config['body_parts']['right']['kp_limits'] = [0, 10000]
    env_info["controller_configs"] = controller_config
    print(env_info)

    env = robosuite.make(
        **env_info,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=30,
        renderer="mjviewer",
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    cube_pos_errors = []
    cube_quat_errors = []
    success_rate = 0.0
    total_index = 0

    for ep in tqdm(demos):
        print("Playing back random episode... (press ESC to quit)")

        cube_pos_err = []
        cube_quat_err = []
        # select an episode randomly
        # ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        # xml = env.edit_model_xml(model_xml)
        # env.reset_from_xml_string(xml)
        # env.sim.reset()
        # env.viewer.set_camera(0)
        if args.render:
            env._destroy_viewer()
        env._destroy_sim()
        env._load_model()
        env._initialize_sim()
        env.reset()

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        is_success = False

        if args.use_actions:

            # # load the initial state
            # env.sim.set_state_from_flattened(states[0])
            # env.sim.forward()
            gt_obs = f["data/{}/observations".format(ep)]

            obj = env.model.mujoco_objects[0]
            env.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([gt_obs['cube_pos'][0], gt_obs['cube_quat'][0]]),
            )
            env.sim.forward()
            env._check_success()
            env._update_observables(force=True)
            # for _ in range(10):
            #     env.sim.step1()
            #     env.sim.step2()
            #     env.sim.forward()
            #     env._update_observables()
            #     env.render()

            # env.sim.forward()
            print(env._get_observations()['cube_pos']-gt_obs['cube_pos'][0])
            cube_pos_err.append(np.linalg.norm(env._get_observations()['cube_pos']-gt_obs['cube_pos'][0]))
            cube_quat_err.append(
                np.linalg.norm(
                    T.quat2axisangle(env._get_observations()['cube_quat'])-T.quat2axisangle(gt_obs['cube_quat'][0])
                )
            )

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                action_dict = {
                    'right': action[:6],
                    'right_gripper': action[6:],
                }
                _action = env.robots[0].create_action_vector(action_dict)
                obs, _, _, _ = env.step(_action)
                cube_pos_err.append(np.linalg.norm(obs['cube_pos']-gt_obs['cube_pos'][j]))
                cube_quat_err.append(np.linalg.norm(T.quat2axisangle(obs['cube_quat'])-T.quat2axisangle(gt_obs['cube_quat'][j])))

                # print(gt_obs['cube_pos'][j]-obs['cube_pos'])

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                if j == num_actions - 1:
                    is_success = env._check_success()
                    print("Success:", env._check_success())
            cube_pos_errors.append(cube_pos_err)
            cube_quat_errors.append(cube_quat_err)
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
    import matplotlib.pyplot as plt
    for i, (pos_err, quat_err) in enumerate(zip(cube_pos_errors, cube_quat_errors)):
        plt.plot(pos_err, label=f"Episode {i} Pos Error")
    plt.legend()
    plt.savefig(f"cube_pos_errors_{env_info['robots'][0]}.png")

    f.close()
    # print the average error for each episode
    print("Average pos error per episode:")
    for i, pos_err in enumerate(cube_pos_errors):
        print(f"Episode {i}: {np.mean(pos_err):.4f}")
    print("Average quat error per episode:")
    for i, quat_err in enumerate(cube_quat_errors):
        print(f"Episode {i}: {np.mean(quat_err):.4f}")

    print(f"Average pos error: {np.mean([np.mean(err) for err in cube_pos_errors]):.4f}")
    print(f"Average quat error: {np.mean([np.mean(err) for err in cube_quat_errors]):.4f}")
    print(f"Success rate: {success_rate / total_index:.3f}")

'''
'''
