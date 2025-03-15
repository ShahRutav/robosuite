"""
Rewritten camera-tuning script for robosuite with macOS segfault fixes:
1. No forced gc.collect() in the loop
2. Small sleep in main loop
3. Uses a with-context for Listener
"""

import argparse
import time
import xml.etree.ElementTree as ET

import numpy as np
from pynput.keyboard import Key, Listener

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import CameraMover

# If numba is causing issues, you can temporarily disable JIT by uncommenting:
# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

# Script settings
DELTA_POS_KEY_PRESS = 0.05  # delta camera position per key press
DELTA_ROT_KEY_PRESS = 1.0   # delta camera angle per key press

class KeyboardHandler:
    """
    Handles keyboard input to move or rotate the camera.
    """
    def __init__(self, camera_mover):
        self.camera_mover = camera_mover

    def on_press(self, key):
        """
        Key handler for key presses.
        """
        try:
            # Rotation with arrow keys
            if key == Key.up:
                self.camera_mover.rotate_camera(point=None, axis=[1.0, 0.0, 0.0], angle=DELTA_ROT_KEY_PRESS)
            elif key == Key.down:
                self.camera_mover.rotate_camera(point=None, axis=[-1.0, 0.0, 0.0], angle=DELTA_ROT_KEY_PRESS)
            elif key == Key.left:
                self.camera_mover.rotate_camera(point=None, axis=[0.0, 1.0, 0.0], angle=DELTA_ROT_KEY_PRESS)
            elif key == Key.right:
                self.camera_mover.rotate_camera(point=None, axis=[0.0, -1.0, 0.0], angle=DELTA_ROT_KEY_PRESS)

            # Movement with w/s/a/d/r/f
            elif key.char == "w":
                self.camera_mover.move_camera(direction=[0.0, 0.0, -1.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "s":
                self.camera_mover.move_camera(direction=[0.0, 0.0, 1.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "a":
                self.camera_mover.move_camera(direction=[-1.0, 0.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "d":
                self.camera_mover.move_camera(direction=[1.0, 0.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "r":
                self.camera_mover.move_camera(direction=[0.0, 1.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == "f":
                self.camera_mover.move_camera(direction=[0.0, -1.0, 0.0], scale=DELTA_POS_KEY_PRESS)
            elif key.char == ".":
                self.camera_mover.rotate_camera(point=None, axis=[0.0, 0.0, 1.0], angle=DELTA_ROT_KEY_PRESS)
            elif key.char == "/":
                self.camera_mover.rotate_camera(point=None, axis=[0.0, 0.0, -1.0], angle=DELTA_ROT_KEY_PRESS)

        except AttributeError:
            # Key pressed is not a standard character (e.g. Shift, Ctrl, etc.)
            pass

    def on_release(self, key):
        """
        Key handler for key releases. (Unused in this script)
        """
        pass


def print_command(char, info):
    """
    Helper to print controls in a neatly formatted way.
    """
    char += " " * (10 - len(char))
    print("{}\t{}".format(char, info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default=["Sawyer"], help="Which robot(s) to use in the env")
    args = parser.parse_args()

    print("\nWelcome to the camera tuning script! Use your keyboard to adjust the camera view.")
    print("\nControls:")
    print_command("Keys", "Command")
    print_command("w-s", "zoom the camera in/out")
    print_command("a-d", "pan the camera left/right")
    print_command("r-f", "pan the camera up/down")
    print_command("arrow keys", "rotate camera direction")
    print_command(".-/", "rotate camera roll about the viewing axis")

    # Prompt user for camera or XML tag
    inp = input(
        "\nPaste a camera name or XML tag (e.g. <camera .../>)\n"
        "Or leave blank for a default example:\n"
    )

    if len(inp) == 0:
        if args.env != "Lift":
            raise ValueError("For the default example, the env must be 'Lift'.")
        print("\nUsing default frontview camera from table_arena.xml:")
        inp = '<camera mode="fixed" name="frontview" pos="1.6 0 1.45" quat="0.56 0.43 0.43 0.56"/>'

    from_tag = "<" in inp  # check if user pasted a full XML tag
    if from_tag:
        cam_tree = ET.fromstring(inp)
        CAMERA_NAME = cam_tree.get("name")
        print("\nNOTE: Using the following XML tag:\n", inp)
    else:
        # Just the name
        CAMERA_NAME = inp
        cam_tree = ET.Element("camera", attrib={"name": CAMERA_NAME})
        print("\nNOTE: Using the following camera name (initialized at default pose):", CAMERA_NAME)

    # Make the environment
    env = robosuite.make(
        args.env,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()

    # Create camera mover
    camera_mover = CameraMover(env=env, camera=CAMERA_NAME)

    # Set the viewer to the chosen camera
    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    env.viewer.set_camera(camera_id=camera_id)

    # If user provided an XML tag, set initial camera pose from the tag
    if from_tag:
        # Extract position and quaternion
        pos_str = cam_tree.get("pos", "0 0 0").split(" ")
        initial_file_camera_pos = np.array(pos_str, dtype=float)
        quat_str = cam_tree.get("quat", "1 0 0 0").split(" ")
        # Convert from MJCF (w,x,y,z) or (x,y,z,w) depending on the code
        # Here, transform_utils expects xyzw
        initial_file_camera_quat = T.convert_quat(np.array(quat_str, dtype=float), to="xyzw")

        # Set camera
        camera_mover.set_camera_pose(
            pos=initial_file_camera_pos,
            quat=initial_file_camera_quat
        )

        # Optional: set camera FOV
        cam_fov = cam_tree.get("fovy", None)
        if cam_fov is not None:
            env.sim.model.cam_fovy[camera_id] = float(cam_fov)
    else:
        initial_file_camera_pos, initial_file_camera_quat = camera_mover.get_camera_pose()

    # Build transformation from the "file" (XML) frame to world frame
    initial_file_camera_pose = T.make_pose(
        initial_file_camera_pos,
        T.quat2mat(initial_file_camera_quat),
    )
    initial_world_camera_pos, initial_world_camera_quat = camera_mover.get_camera_pose()
    initial_world_camera_pose = T.make_pose(
        initial_world_camera_pos,
        T.quat2mat(initial_world_camera_quat)
    )
    world_in_file = initial_file_camera_pose.dot(T.pose_inv(initial_world_camera_pose))

    # Instantiate keyboard handler
    handler = KeyboardHandler(camera_mover)

    # Use a blocking approach for pynput's Listener
    with Listener(on_press=handler.on_press, on_release=handler.on_release) as listener:
        spin_count = 0
        print("\nInteractive camera tuning started. Press Ctrl+C to quit.\n")

        # Main rendering loop
        while True:
            # Step the environment (no actual control, just zeros)
            action = np.zeros(env.action_dim)
            env.step(action)
            env.render()

            spin_count += 1
            if spin_count % 500 == 0:
                # Recompute camera pose relative to original "file" frame
                camera_pos, camera_quat = camera_mover.get_camera_pose()
                world_camera_pose = T.make_pose(camera_pos, T.quat2mat(camera_quat))
                file_camera_pose = world_in_file.dot(world_camera_pose)
                # Convert pose to (pos, wxyz)
                camera_pos, camera_quat = T.mat2pose(file_camera_pose)
                camera_quat = T.convert_quat(camera_quat, to="wxyz")

                print("\nCurrent camera tag to copy into your XML:\n")
                cam_tree.set("pos", "{} {} {}".format(*camera_pos))
                cam_tree.set("quat", "{} {} {} {}".format(*camera_quat))
                print(ET.tostring(cam_tree, encoding="utf8").decode("utf8"))

            # Sleep to reduce CPU usage and potential race conditions
            time.sleep(0.01)

        # This will never be reached unless we break/exit, but for completeness:
        listener.join()

