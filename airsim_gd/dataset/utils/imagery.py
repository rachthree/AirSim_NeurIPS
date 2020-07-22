import numpy as np
import cv2 as cv2
import airsimneurips as airsim

from airsim_gd.dataset.utils import math_utils

def get_visible_gates(seg_image, id2lvlobjs_dict):
    gate_list = []

    for obj_id in np.unique(seg_image):
        if 'gate' in ' '.join(id2lvlobjs_dict[obj_id]).lower():
            if len(id2lvlobjs_dict[obj_id]) > 1:
                raise ValueError(f"Multiple objects assigned same gate segmentation ID {obj_id}")

            gate_list.append(id2lvlobjs_dict[obj_id][0])

    return gate_list


def get_sorted_gates(airsim_client, camera_info, gate_list):
    results = []

    for gate_name in gate_list:
        gate_pose = airsim_client.simGetObjectPose(object_name=gate_name)
        distance = np.sum((np.array([camera_info.pose.position.x_val, camera_info.pose.position.y_val, camera_info.pose.position.z_val]) -
                           np.array([gate_pose.position.x_val, gate_pose.position.y_val, gate_pose.position.z_val]))**2)**0.5
        results.append([gate_name, distance])

    results.sort(key=lambda x: x[1])

    return results


def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight):
    # Source: https://github.com/microsoft/AirSim/blob/master/PythonClient/computer_vision/capture_ir_segmentation.py

    # Turn the camera position into a column vector.
    camPosition = np.array([[camXYZ.x_val, camXYZ.y_val, camXYZ.z_val]]).T

    # Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    pitchRollYaw = airsim.to_eularian_angles(camQuaternion)

    # Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = math_utils.rotation_matrix_from_angles(pitchRollYaw)

    # Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = np.transpose([subjectXYZ])
    XYZW = np.add(XYZW, -camPosition)
    #print("XYZW: " + str(XYZW))
    XYZW = np.matmul(np.transpose(camRotation), XYZW)
    #print("XYZW derot: " + str(XYZW))

    # Recreate the perspective projection of the camera.
    XYZW = np.concatenate([XYZW, [[1]]])
    XYZW = np.matmul(camProjMatrix4x4, XYZW)
    XYZW = XYZW / XYZW[3]

    # Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2

    return np.array([int(imageWidthHeight[1] * normX), int(imageWidthHeight[0] * normY)])


def get_flyable_region(airsim_client, gate_name, inner_dim=(1.6, 0.2, 1.6), outer_dim=(2.1333, 0.2, 2.1333)):
    # Gets the flyable region of the gate in global coordinates
    # Inputs:
    # airsim_client: AirSim client
    # gate_name: (str) the gate name
    # inner_dim: (tuple) inner dimensions of gate (W x D x H) in m
    # outer_dim: (tuple) outer dimensions of gate (W x D x H) in m
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
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    ks = airsim_client.simGetObjectScale(object_name=gate_name)  # gate scale

    region_size = [ks.x_val*inner_dim[0], ks.z_val*inner_dim[2]]  # [x, z]
    gate_t = ks.y_val*inner_dim[1]
    gate_pose = airsim_client.simGetObjectPose(object_name=gate_name)
    gate_center = [gate_pose.position.x_val, gate_pose.position.y_val, gate_pose.position.z_val]
    rotmat = math_utils.quat2mat(gate_pose.orientation)

    # Gate Csys
    gate_x = rotmat.dot(x)  # left of gate, gate front visible
    gate_y = rotmat.dot(y)  # front to back of gate
    gate_z = rotmat.dot(z)  # downwards in AirSim Coordinates

    fly_region_global['top_left'] = gate_center + gate_x*region_size[0]/2 + gate_y*gate_t/2 - gate_z*region_size[1]/2
    fly_region_global['top_right'] = gate_center - gate_x*region_size[0]/2 + gate_y*gate_t/2 - gate_z*region_size[1]/2
    fly_region_global['bot_left'] = gate_center + gate_x*region_size[0]/2 + gate_y*gate_t/2 + gate_z*region_size[1]/2
    fly_region_global['bot_right'] = gate_center - gate_x*region_size[0]/2 + gate_y*gate_t/2 + gate_z*region_size[1]/2

    return fly_region_global


def project_global2cam_fly_region(fly_region_global, camera_info, img_wh):

    cam_pose = camera_info.pose
    projection_mat = camera_info.proj_mat.matrix
    fly_region_cam = {}

    # H, W
    fly_region_cam['top_left'] = project_3d_point_to_screen(fly_region_global['top_left'],
                                                            cam_pose.position,
                                                            cam_pose.orientation,
                                                            projection_mat, img_wh)

    fly_region_cam['top_right'] = project_3d_point_to_screen(fly_region_global['top_right'],
                                                             cam_pose.position,
                                                             cam_pose.orientation,
                                                             projection_mat, img_wh)

    fly_region_cam['bot_left'] = project_3d_point_to_screen(fly_region_global['bot_left'],
                                                            cam_pose.position,
                                                            cam_pose.orientation,
                                                            projection_mat, img_wh)

    fly_region_cam['bot_right'] = project_3d_point_to_screen(fly_region_global['bot_right'],
                                                             cam_pose.position,
                                                             cam_pose.orientation,
                                                             projection_mat, img_wh)

    return fly_region_cam

def segment_flyable_region(region_label, seg_image, depth_image, fly_region_global, fly_region_cam, camera_info, category2id_dict):
    # TODO: Determine which part of the flyable region can actually be seen using the segment ID's... occlusion!
    # Define areas to fill with flyable regions
    # Determine closest gates given segment ID's -> gate number -> gate location in sim.
    # New segment ID in increasing distance.
    # New alpha channel in PNG.

    final_seg_image = np.copy(seg_image)

    # using fly_region_cam, get the pixels of the region and their depths
    pts = np.array([fly_region_cam['top_left'],
                    fly_region_cam['top_right'],
                    fly_region_cam['bot_right'],
                    fly_region_cam['bot_left']]).astype(np.int32)

    mask = np.zeros_like(final_seg_image)

    # in cv2, y is row, x is col
    cv2.fillConvexPoly(mask, pts, 1)
    mask = mask.astype(np.bool)

    # region_id_mask = np.zeros_like(final_seg_image)
    # region_id_mask[mask] = final_seg_image[mask]

    region_dist_mask = np.zeros_like(final_seg_image)
    region_dist_mask[mask] = depth_image[mask]

    # get plane using fly_region_global, check which points in fly_region_cam are between camera and region
    # assume that if any of the distances are less than the min distance between the camera and any of the 4 pts,
    # then there is occlusion
    thres = np.min(np.sum((np.array([camera_info.pose.position.x_val, camera_info.pose.position.y_val, camera_info.pose.position.z_val]) -
                           np.array(list(fly_region_global.values())))**2, axis=-1)**0.5
                   )

    # those that not between camera and region are changed to flyable region ID
    final_seg_image[region_dist_mask > thres] = category2id_dict[region_label]

    return final_seg_image
