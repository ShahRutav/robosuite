# Changelog


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