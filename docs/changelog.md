# Changelog

## Version 1.5.2

- **Bug Fixes**:  
  - Fixed domain randomization compatibility for `mujoco!=3.1.1`.  
  - Corrected dexterous hand mounting for Panda and pot-with-handles geometry for rectangular pots.  
  - Fixed Mink IK with delta input, duplicated inertial properties, free-joint issues, and Baxter mesh problems.  
  - Resolved mobile base XML and controller logic issues, binding utils on Windows, and EGL error reporting.  
  - Hot-fixed `Task.generate_id_mappings()` to remove invalid EEF targets.

- **Features / Enhancements**:  
  - Added support for multiple cameras across renderers and observation pipelines.  
  - Enabled instance segmentation to include arena bodies.  
  - Added DualSense controller support and improved SpaceMouse auto-detection.  
  - Introduced scalable arenas and MuJoCo objects via `set_scale`.  
  - Allowed base type specification and manipulator mounts for mobile manipulation envs.  
  - Added new robots and hands (xArm7 revised, Inspire/Fourier grasp qpos).  
  - Extended whole-body IK with `input_ref_frame=base` and `skip_wbc_action`.  
  - Added sensors, shell inertia for Sawyer meshes, and joint position observations.

- **Documentation Updates**:  
  - Updated v1.5 docs, fixed broken links, improved dark-mode visibility, and refined demo/device documentation.  
  - Added USD requirements and SpaceMouse dependencies to docs and extras.  
  - Improved demos page and clarified base type comments.

- **CI / Release / Tooling**:  
  - Added and refined PyPI publishing workflows.  
  - Improved build–docs synchronization and pinned Mink version.  

- **Miscellaneous**:  
  - Added render modalities and updated demo scripts to include all manipulators by default.  
  - Refactored controller configs and composite controller logic.  
  - Improved data collection wrapper resets and environment seeding.

## Version 1.5.1

- **Bug Fixes**:  
  - Fixed segmentation demo, part controller demo, demo renderer, joint and actuator issues, equality removal, and orientation mismatch in Fourier hands.  

- **Documentation Updates**:  
  - Updated `basicusage.md`, overview page, demo docs, teleop usage info, devices, and related docs.  
  - Added a CI doc update workflow and `.nojekyll` fix.  
  - Simplified composite controller keywords and updated robots section and task images.  

- **Features/Enhancements**:  
  - Added GymWrapper support for both `gym` and `gymnasium` with `dict_obs`.  
  - Updated DexMimicGen assets and added a default whole-body Mink IK config.  
  - Improved CI to check all tests and print video save paths in demo recordings. 
  - Add GR1 and spot robot to `demo_random_actions.py` script.

- **Miscellaneous**:  
  - Added troubleshooting for SpaceMouse failures and terminated `mjviewer` on resets.  
  - Adjusted OSC position fixes and updated part controller JSONs.  

## Version 1.5.0

<div class="admonition warning">
<p class="admonition-title">Breaking API changes</p>
<div>
    <ul>
        <li>
        New controller design: Introduction of composite controllers, which take in a high-level action vector and converts it into commands for each body part controller.
        </li>
    </ul>
</div>
</div>


<div>
    <ul>
        <li>Introduction of custom robot composition: arms, grippers, and bases can be swapped to create new robots configurations</li>
        <li>Integration of more diverse robot embodiments: including humanoids, quadrupeds, and more</li>
        <li>Support for mobile manipulation, whole body IK, and third-party controllers (e.g. Mink)</li>
        <li>Implementation of MuJoCo viewer drag-drop teleoperation interface</li>
        <li>Support for photo-realistic rendering via USD exporting and NVIDIA Isaac Sim rendering</li>
    </ul>
</div>
