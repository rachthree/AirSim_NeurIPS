import numpy as np
import cv2
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


def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageHeightWidth):
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
    if XYZW[0] < 0:  # point is behind the camera
        return None

    # Recreate the perspective projection of the camera.
    XYZW = np.concatenate([XYZW, [[1]]])
    XYZW = np.matmul(camProjMatrix4x4, XYZW)
    XYZW = XYZW / XYZW[3]

    # Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2

    return np.array([int(imageHeightWidth[0] * normY), int(imageHeightWidth[1] * normX)])


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


def project_global2cam_fly_region(fly_region_global, camera_info, img_hw):

    cam_pose = camera_info.pose
    projection_mat = camera_info.proj_mat.matrix
    fly_region_cam = {}

    # H, W
    fly_region_cam['top_left'] = project_3d_point_to_screen(fly_region_global['top_left'],
                                                            cam_pose.position,
                                                            cam_pose.orientation,
                                                            projection_mat, img_hw)

    fly_region_cam['top_right'] = project_3d_point_to_screen(fly_region_global['top_right'],
                                                             cam_pose.position,
                                                             cam_pose.orientation,
                                                             projection_mat, img_hw)

    fly_region_cam['bot_left'] = project_3d_point_to_screen(fly_region_global['bot_left'],
                                                            cam_pose.position,
                                                            cam_pose.orientation,
                                                            projection_mat, img_hw)

    fly_region_cam['bot_right'] = project_3d_point_to_screen(fly_region_global['bot_right'],
                                                             cam_pose.position,
                                                             cam_pose.orientation,
                                                             projection_mat, img_hw)

    return fly_region_cam


def segment_flyable_region(region_label, seg_image, depth_image, fly_region_global, fly_region_cam, camera_info, class2id_dict):
    final_seg_image = np.copy(seg_image)

    # Using fly_region_cam, get the pixels of the region and their depths
    # Pad image in case flyable region was calculated to be outside the bounds of the actual image
    img_hw = seg_image.shape
    corner_pts = np.array(list(fly_region_cam.values()))
    corner_row_min = min(corner_pts[:, 0])
    corner_row_max = max(corner_pts[:, 0])
    corner_col_min = min(corner_pts[:, 1])
    corner_col_max = max(corner_pts[:, 1])

    pad_row_min = min(corner_row_min, 0) - 1
    pad_row_max = max(corner_row_max, img_hw[0]) + 1
    pad_col_min = min(corner_col_min, 0) - 1
    pad_col_max = max(corner_col_max, img_hw[1]) + 1

    pad_img_h = img_hw[0] - pad_row_min + pad_row_max
    pad_img_w = img_hw[1] - pad_col_min + pad_col_max
    pad_img = np.zeros((pad_img_h, pad_img_w), dtype=np.int32)
    pad_offset_h = (pad_img_h - img_hw[0]) // 2
    pad_offset_w = (pad_img_w - img_hw[0]) // 2

    img_mask = np.copy(pad_img)
    fly_region_mask = np.copy(pad_img)
    img_mask[pad_offset_h:pad_offset_h+img_hw[0], pad_offset_w:pad_offset_w+img_hw[1]] = 1

    pts_list = []
    for corner in ['top_left', 'top_right', 'bot_right', 'bot_left']:
        pt_temp = fly_region_cam[corner] + np.array([pad_offset_h, pad_offset_w])
        pts_list.append(pt_temp[::-1].astype(np.int32))  # reversed for opencv, y is row, x is col

    cv2.fillConvexPoly(fly_region_mask, np.array(pts_list), 1)
    mask = img_mask & fly_region_mask
    mask = mask.astype(np.bool)
    mask = mask[pad_offset_h:pad_offset_h+img_hw[0], pad_offset_w:pad_offset_w+img_hw[1]]

    region_dist_mask = np.zeros_like(final_seg_image)
    region_dist_mask[mask] = depth_image[mask]

    # get plane using fly_region_global, check which points in fly_region_cam are between camera and region
    # assume that if any of the distances are less than the min distance between the camera and any of the 4 pts,
    # then there is occlusion
    thres = np.min(np.sum((np.array([camera_info.pose.position.x_val, camera_info.pose.position.y_val, camera_info.pose.position.z_val]) -
                           np.array(list(fly_region_global.values())))**2, axis=-1)**0.5
                   )

    # those that not between camera and region are changed to flyable region ID
    final_seg_image[region_dist_mask > thres] = class2id_dict[region_label]

    return final_seg_image
