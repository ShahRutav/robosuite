import argparse
import json
import numpy as np
import xml.etree.ElementTree as ET

import robosuite
import robocasa
from robocasa.utils.env_utils import create_env
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa
from robosuite.utils.transform_utils import mat2quat

#########################################################
# 1. Original function (unchanged)
#########################################################
def get_camera_axes(look_at, camera_pos, up):
    """
    Calculate the camera x and y axes given a look-at point, camera position, and up vector.

    Parameters:
        look_at (array-like): A 3D point the camera is looking at [x, y, z].
        camera_pos (array-like): The 3D position of the camera [x, y, z].
        up (array-like): The up vector [x, y, z].

    Returns:
        tuple: (x_axis, y_axis) of the camera frame
    """
    look_at = np.array(look_at)
    camera_pos = np.array(camera_pos)
    up = np.array(up)

    # Forward (z-axis) is direction from camera to object
    z_axis = look_at - camera_pos
    z_axis /= np.linalg.norm(z_axis)

    # Right (x-axis)
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)

    # True up (y-axis) re-orthogonalized
    y_axis = np.cross(x_axis, z_axis)

    return x_axis, y_axis

#########################################################
# 2. New function that calls get_camera_axes
#########################################################
def set_camera_look_at(env, camera_name, object_name, distance=1.0, up=(0, 0, 1)):
    """
    Positions and orients the specified camera so it looks at the given object (body)
    from some distance away, using 'up' as the global up direction.

    Args:
        env (MujocoEnv): A robosuite environment (already created and reset).
        camera_name (str): Name of the camera in the sim.model (e.g. "frontview").
        object_name (str): Name of the body in the simulation (e.g. "cube").
        distance (float): How far the camera should be from the object.
        up (tuple or list): The up vector to define the camera's 'up'.
    """
    # 1) Get the object's position from the simulator
    object_pos = None
    if hasattr(env, "sim"):
        object_pos = env.sim.data.get_body_xpos(object_name)
    else:
        object_pos = env.env.sim.data.get_body_xpos(object_name)
    if object_pos is None:
        raise ValueError(f"Body '{object_name}' does not exist in this environment!")

    # 2) Decide where the camera should be placed.
    #    For example: behind the object along negative y-axis by 'distance'
    new_dist = (distance ** 0.5) * 2
    camera_pos = np.array([object_pos[0]-0.5, object_pos[1]+0.01, object_pos[2] + 0.07])

    # 3) Use the original function to get the camera's x-axis and y-axis
    x_axis, y_axis = get_camera_axes(
        look_at=object_pos,
        camera_pos=camera_pos,
        up=up
    )

    # 4) Derive the camera's z-axis from x_axis and y_axis
    z_axis = np.cross(x_axis, y_axis)

    # 5) Construct a 3x3 rotation matrix, columns = [x_axis, y_axis, z_axis]
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])

    # 6) Convert rotation matrix to quaternion (xyzw)
    camera_quat_xyzw = mat2quat(rot_mat)

    # Reorder to (w, x, y, z) because MuJoCo expects w-first quaternions
    camera_quat_wxyz = np.array([
        camera_quat_xyzw[3],
        camera_quat_xyzw[0],
        camera_quat_xyzw[1],
        camera_quat_xyzw[2],
    ])

    # 7) Assign the new camera position + orientation to MuJoCo
    cam_id = None
    if hasattr(env, "sim"):
        cam_id = env.sim.model.camera_name2id(camera_name)
        env.sim.model.cam_pos[cam_id] = camera_pos
        env.sim.model.cam_quat[cam_id] = camera_quat_wxyz
    else:
        cam_id = env.env.sim.model.camera_name2id(camera_name)
        env.env.sim.model.cam_pos[cam_id] = camera_pos
        env.env.sim.model.cam_quat[cam_id] = camera_quat_wxyz

    # Return the final position + orientation in MuJoCo's (w, x, y, z) format
    return camera_pos, camera_quat_wxyz

#########################################################
# 3. Main script demonstrating usage
#########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift", help="Which robosuite environment to use.")
    parser.add_argument("--robots", nargs="+", type=str, default=["Sawyer"], help="Which robot(s) to use.")
    parser.add_argument("--object_name", type=str, default="cube", help="Body name of the object to look at.")
    parser.add_argument("--camera_name", type=str, default="frontview", help="Camera name in the environment.")
    parser.add_argument("--distance", type=float, default=1.0, help="Distance from object to camera.")
    parser.add_argument("--no_update", action="store_true", help="If set, won't update the camera position.")
    args = parser.parse_args()

    # Create the environment with on-screen rendering
    if args.env == "Lift":
        env = robosuite.make(
            args.env,
            robots=args.robots,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera=args.camera_name,
            use_camera_obs=False,
            control_freq=20,
            camera_names=[args.camera_name],
        )
    else:
        image_modalities = ["robot0_agentview_left", "robot0_agentview_right"]
        obs_modality_specs = {
            "obs": {
                "rgb": image_modalities,
            }
        }
        env_meta_file = "/home/rutavms/research/gaze/icrt/robocasa/scene_configs/SinkPlayEnvTrain/training/ep_meta_000.json"
        env_meta = json.load(open(env_meta_file, "r"))
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
        env = EnvRobocasa(
            env_name=args.env,
            robots=args.robots,
            has_renderer=True,
            has_offscreen_renderer=False,
            render=True,
            use_camera_obs=False,
            camera_names=[args.camera_name],
            ep_meta=env_meta,
        )
    for _ in range(10):
        try:
            env.reset()
            break
        except Exception as e:
            print(f"Error: {e}")
            pass

    if not args.no_update:
        # Actually set the camera to look at the chosen object
        camera_pos, camera_quat = set_camera_look_at(
            env=env,
            camera_name=args.camera_name,
            object_name=args.object_name,
            distance=args.distance,
            up=(0, 0, 1),  # you can use a different up vector if desired
        )

        print(f"Camera '{args.camera_name}' now looks at '{args.object_name}' from distance {args.distance}.")

        # ---------------------------------------------------
        # 4. Print out the <camera> XML snippet
        # ---------------------------------------------------
        # We'll create a small XML tree for the "camera" element
        # so you can copy-paste into your MJCF if desired.
        cam_tree = ET.Element("camera", attrib={"name": args.camera_name, "mode": "fixed"})
        # Set position and quaternion in the element
        # (MuJoCo typically wants "pos" and "quat" in w,x,y,z)
        cam_tree.set("pos", "{} {} {}".format(*camera_pos))
        cam_tree.set("quat", "{} {} {} {}".format(*camera_quat))

        print("\nCurrent camera tag to copy into your XML:\n")
        print(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))

    # ---------------------------------------------------
    # 5. Run a loop so you can visualize
    # ---------------------------------------------------
    while True:
        action = None
        if hasattr(env, "action_dim"):
            action = np.zeros(env.action_dim)
        else:
            action = np.zeros(env.action_dimension)
        env.step(action)  # no-op action
        env.render()

