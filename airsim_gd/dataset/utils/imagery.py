import numpy as np
import airsimneurips as airsim

from airsim_gd.dataset.utils import math_utils


def get_visible_gates(seg_image, seg_id_dict):
    gate_list = []

    for obj_id in np.unique(seg_image[:, :, 0]):
        if 'gate' in seg_id_dict[obj_id].lower():
            gate_list.append(obj_id)

    return gate_list


def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight):
    # Source: https://github.com/microsoft/AirSim/blob/master/PythonClient/computer_vision/capture_ir_segmentation.py

    # Turn the camera position into a column vector.
    camPosition = np.transpose([camXYZ])

    # Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = airsim.to_eularian_angles(camQuaternion)

    # Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = math_utils.rotation_matrix_from_angles(pitchRollYaw)

    # Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = np.transpose([subjectXYZ])
    XYZW = np.add(XYZW, -camPosition)
    print("XYZW: " + str(XYZW))
    XYZW = np.matmul(np.transpose(camRotation), XYZW)
    print("XYZW derot: " + str(XYZW))

    # Recreate the perspective projection of the camera.
    XYZW = np.concatenate([XYZW, [[1]]])
    XYZW = np.matmul(camProjMatrix4x4, XYZW)
    XYZW = XYZW / XYZW[3]

    # Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2

    return np.array([
        imageWidthHeight[0] * normX,
        imageWidthHeight[1] * normY
    ]).reshape(2, )


def get_flyable_region(airsim_client, gate_name, inner_dim=(1.6, 0.2, 1.6), outer_dim=(2.1333, 0.2, 2.1333)):
    # Gets the flyable region of the gate in global coordinates
    # Inputs:
    # airsim_client: AirSim client
    # gate_name: (str) the gate name
    # inner_dim: (tuple) inner dimensions of gate (W x D x H)
    # outer_dim: (tuple) outer dimensions of gate (W x D x H)
    #
    # Outputs:
    # fly_region_global: (dict) dictionary with global coordinates of the flyable region

    # Scale 1:
    # inner dimensions (width, thickness, height) 1.6m x 0.2 m x 1.6 m
    # outer dimensions (width, thickness, height): 2.1333 m x 0.2 m x 2.1333 m
    # gate origin is at center

    # Dimensions in meters
    # Assume flyable region is at front of the gate

    fly_region_global = {}
    x = [1, 0, 0]  # right of gate, gate front visible
    y = [0, 1, 0]  # out of gate
    z = [0, 0, 1]  # downwards in AirSim Coordinates

    k = airsim_client.simGetObjectScale(object_name=gate_name)  # gate scale

    region_size = k*[outer_dim[0] - inner_dim[0], outer_dim[2] - inner_dim[2]]  # [x, z]
    gate_t = inner_dim[1]
    gate_pose = airsim_client.simGetObjectPose(object_name=gate_name)
    gate_center = [gate_pose.position.x_val, gate_pose.position.y_val, gate_pose.position.z_val]
    rotmat = math_utils.quat2mat(gate_pose.orientation)

    # Gate Csys
    gate_x = rotmat.dot(x)
    gate_y = rotmat.dot(y)
    gate_z = rotmat.dot(z)

    fly_region_global['top_left'] = gate_center - gate_x*region_size[0]/2 + gate_y*gate_t/2 - gate_z*region_size[1]/2
    fly_region_global['top_right'] = gate_center + gate_x*region_size[0]/2 + gate_y*gate_t/2 - gate_z*region_size[1]/2
    fly_region_global['bot_left'] = gate_center - gate_x*region_size[0]/2 + gate_y*gate_t/2 + gate_z*region_size[1]/2
    fly_region_global['bot_right'] = gate_center + gate_x*region_size[0]/2 + gate_y*gate_t/2 + gate_z*region_size[1]/2

    return fly_region_global


def fly_region_global2cam(fly_region_global, camera_info, img_wh):

    cam_pose = camera_info.pose
    projection_mat = camera_info.proj_mat
    fly_region_cam = {}

    fly_region_cam['h_topleft'], fly_region_cam['w_topleft'] = project_3d_point_to_screen(fly_region_global['top_left'],
                                                                                          cam_pose.position,
                                                                                          cam_pose.orientation,
                                                                                          projection_mat, img_wh)

    fly_region_cam['h_topright'], fly_region_cam['w_topright'] = project_3d_point_to_screen(fly_region_global['top_right'],
                                                                                            cam_pose.position,
                                                                                            cam_pose.orientation,
                                                                                            projection_mat, img_wh)

    fly_region_cam['h_botleft'], fly_region_cam['w_botleft'] = project_3d_point_to_screen(fly_region_global['bot_left'],
                                                                                          cam_pose.position,
                                                                                          cam_pose.orientation,
                                                                                          projection_mat, img_wh)

    fly_region_cam['h_botright'], fly_region_cam['w_botright'] = project_3d_point_to_screen(fly_region_global['bot_right'],
                                                                                            cam_pose.position,
                                                                                            cam_pose.orientation,
                                                                                            projection_mat, img_wh)

    return fly_region_cam

def get_final_flyable_region(seg_image, fly_region_cam, seg_id_dict):
    # TODO: Determine which part of the flyable region can actually be seen using the segment ID's... occlusion!
    # Define areas to fill with flyable regions
    # Determine closest gates given segment ID's -> gate number -> gate location in sim.
    # New segment ID in increasing distance.
    # New alpha channel in PNG.
    pass


def process_flyable_region(airsim_client, gate_name, seg_image, camera_info, seg_id_dict):
    # Process chain
    img_wh = seg_image.shape
    fly_region_global = get_flyable_region(airsim_client, gate_name)

    fly_region_cam = fly_region_global2cam(fly_region_global, camera_info, img_wh)
    final_seg_image = get_final_flyable_region(seg_image, fly_region_cam, seg_id_dict)

    return final_seg_image
