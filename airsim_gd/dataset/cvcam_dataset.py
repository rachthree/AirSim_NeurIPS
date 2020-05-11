import os
import time
import json
from copy import deepcopy
from pathlib import Path
import csv
from argparse import ArgumentParser

import numpy as np
import cv2 as cv
import pandas as pd

import airsimneurips as airsim
from airsim_gd.vision.utils import setupASClient
from airsim_gd.dataset.utils import imagery


level_list = ["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard"]


def pose_to_dict(pose, t=0.0):
    return {"t": t, "x": pose.position.x_val, "y": pose.position.y_val, "z": pose.position.z_val,
            "qw": pose.orientation.w_val, "qx": pose.orientation.x_val, "qy": pose.orientation.y_val, "qz": pose.orientation.z_val}


def pose_to_array(pose):
    return [pose.position.x_val, pose.position.y_val, pose.position.z_val,
            pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val]


def pix_to_id(pix_array, rgb2id_dict):
    return rgb2id_dict[pix_array[0]][pix_array[1]][pix_array[2]]


class CVCameraDataset(object):
    def __init__(self, sid_spreadsheet_loc=Path("airsim_gd/dataset/levels_objects.xlsx"),
                 save_dir=Path.cwd().parent.joinpath("data"),
                 cam_mode="single",
                 sess_name=time.ctime(time.time()).replace(':', '.'),
                 cameras=["fpv_cam", "aft", "starboard", "port", "bottom"],
                 segmentation_id_path=Path("airsim_gd/dataset/segmentation_id_maps.json"),
                 gate_label_basename='gate',
                 fly_region_label_basename='flyregion'):
        self.client = setupASClient()
        self.sid_spreadsheet_loc = Path(sid_spreadsheet_loc)
        self.save_dir = Path(save_dir)
        self.cam_mode = cam_mode
        self.pose_history_df = None
        self.sess_name = sess_name
        self.level_name = ""
        self.cameras = cameras

        segID_maps = json.load(segmentation_id_path)
        self.id2rgb = segID_maps["id2rgb"]
        self.rgb2id = segID_maps["rgb2id"]

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gate_label_basename = gate_label_basename
        self.fly_region_label_basename = fly_region_label_basename

        self.seg_obj2id = {}
        self.seg_id2category = {}
        self.seg_category2id = {}

    def load_level(self, level_name, category_seg_id_tab):
        if level_name not in level_list:
            raise ValueError("Invalid level. Choose one from {}".format(level_list))

        self.client.simLoadLevel(level_name)
        self.level_name = level_name

        # load segment ID's - level specific
        df = pd.read_excel(self.sid_spreadsheet_loc, sheet_name=level_name)
        for obj, seg_id in zip(df['object'], df['segment_ID']):
            self.client.simSetSegmentationObjectID(obj, seg_id)
            self.seg_obj2id[obj] = seg_id

        # load segment ID's - general
        df = pd.read_excel(self.sid_spreadsheet_loc, sheet_name=category_seg_id_tab)
        for category, seg_id in zip(df['category'], df['segment_ID']):
            self.seg_id2category[seg_id] = category
            self.seg_category2id[category] = seg_id

    def record_path(self, vehicle_name="drone_1", interval=0.05):
        t_start = time.clock()
        t0 = t_start
        pose_history = []
        pose_temp = self.client.simGetVehiclePose(vehicle_name)
        pose_history.append(pose_to_dict(pose_temp, t0 - t_start))  # zero out time
        pose_array0 = pose_to_array(pose_temp)
        try:
            while True:
                t1 = time.clock()
                if t1 - t0 >= interval:
                    pose_temp = self.client.simGetVehiclePose(vehicle_name)
                    pose_array1 = pose_to_array(pose_temp)
                    if pose_array0 == pose_array1:
                        continue  # Do not record duplicates
                    else:
                        pose_history.append(pose_to_dict(pose_temp, t1 - t_start))
                        t0 = t1
                        pose_array0 = deepcopy(pose_array1)

        except KeyboardInterrupt:
            csv_savepath = self.save_dir.joinpath(self.level_name + " " + self.sess_name + ".csv")
            history_df = pd.DataFrame(pose_history)
            self.pose_history_df = history_df
            history_df.to_csv(csv_savepath, sep='\t', encoding='utf-8', index=False)
            print(f"Saved positional time history to {csv_savepath}.")

    def take_images_dataset(self):
        # get uncompressed fpv cam image
        # Scene = 0,
        # DepthPlanner = 1,
        # DepthPerspective = 2,
        # DepthVis = 3,
        # DisparityNormalized = 4,
        # Segmentation = 5,
        # SurfaceNormals = 6,
        # Infrared = 7
        img_rgb_cam = {}
        img_seg_cam = {}
        img_seg_id_cam = {}
        img_depth_cam = {}
        images_dict = {}

        for camera in self.cameras:
            # Take images from fpv_cam, back, starboard, port, bottom
            # RGB, Segmentation, Depth
            responses = self.client.simGetImages([
                # uncompressed RGBA array bytes
                airsim.ImageRequest(camera, airsim.ImageType.Scene, pixels_as_float=False, compress=False),
                # floating point uncompressed image
                airsim.ImageRequest(camera, airsim.ImageType.Segmentation, pixels_as_float=False, compress=False),
                # Depth
                airsim.ImageRequest(camera, airsim.ImageType.DepthVis, pixels_as_float=True, compress=False)
            ])

            rgb_response = responses[0]
            img_rgb_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb_1d.reshape(rgb_response.height, rgb_response.width, 3)
            img_rgb_cam[camera] = img_rgb

            seg_response = responses[1]
            seg_1d = np.frombuffer(seg_response.image_data_uint8, dtype=np.uint8)
            img_seg = seg_1d.reshape(seg_response.height, seg_response.width, 3)
            img_seg_ID = np.apply_along_axis(pix_to_id, 2, img_seg, self.rgb2id)
            img_seg_cam[camera] = img_seg
            img_seg_id_cam[camera] = img_seg_ID

            depth_response = responses[2]
            depth_1d = np.frombuffer(depth_response.image_data_uint8, dtype=np.uint8)
            img_depth = depth_1d.reshape(depth_response.height, depth_response.width, 3)
            img_depth_cam[camera] = img_depth

        images_dict['rgb'] = img_rgb_cam
        images_dict['seg'] = img_seg_cam
        images_dict['segID'] = img_seg_ID
        images_dict['depth'] = img_depth_cam

        return images_dict

    def get_flyable_region(self, images_dict):
        # Input:
        # images_dict: (dict) images dictionary created from CVCameraDataset.take_images_dataset()
        # pose: AirSim Pose of the drone

        # Output:
        # N/A, images_dict is updated in-place

        # Determine flyable region without occlusion... need projection of the 4 corners
        for camera in self.cameras:
            seg_image = images_dict['seg_ID'][camera]
            depth_image = images_dict['depth'][camera]

            gate_list = imagery.get_visible_gates(seg_image, self.seg_id2category)
            camera_info = self.client.simGetCameraInfo(camera_name=camera)
            sorted_gate_info = imagery.get_sorted_gate_labels(self.client, camera_info, gate_list,
                                                              self.gate_label_basename, self.fly_region_label_basename)

            gate_mask_list = []
            for i, (gate_name, _) in enumerate(sorted_gate_info):
                region_label = f"{self.fly_region_label_basename}_{str(i + 1)}"

                gate_mask_list.append(seg_image == self.seg_obj2id[gate_name])

                # Process chain
                img_wh = seg_image.shape
                fly_region_global = imagery.get_flyable_region(self.client, gate_name)
                fly_region_cam = imagery.project_global2cam_fly_region(fly_region_global, camera_info, img_wh)

                images_dict['seg_ID'][camera] = imagery.segment_flyable_region(region_label, seg_image, depth_image, fly_region_global, fly_region_cam,
                                                                               camera_info, self.seg_category2id)

            for i in range(sorted_gate_info):
                # Doing this outside the previous for loop to prevent multiple gates being edited at oncec
                images_dict['seg_ID'][camera][gate_mask_list[i]] = self.seg_category2id[f"{self.gate_label_basename}_{str(i+1)}"]

    def get_labeled_images(self):
        images_dict = self.take_images_dataset()
        self.get_flyable_region(images_dict)

        # save images
        # save depths as PFM

    def generate_data(self, vehicle_name="drone_1", csv_path=None, fly_region_start_id=101, gate_start_id=1, gate_lim=10):
        # load json or use attribute
        if csv_path is None:
            history_df = self.pose_history_df
        else:
            history_df = pd.read_csv(csv_path)

        # At each position, place camera/drone
        for row in history_df.itertuples():
            # At each position, place camera/drone
            pose = airsim.Pose(position_val={'x_val': row.x, 'y_val': row.y, 'z_val': row.z},
                               orientation_val={'w_val': row.qw, 'x_val': row.qx, 'y_val': row.qy, 'z_val': row.qz})
            self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name)

            # Take initial images
            self.get_labeled_images()

            # TODO: place drone 2 randomly and take images again
            # Random rotation among 4 quadrants of drone POV...
            # use airsim.utils.to_eularian_angles(pose.orientation), adjust, use airsim.utils.to_quaternion(roll, pitch, yaw)
            # euler_ang0 = airsim.utils.to_eularian_angles(pose.orientation)

            # TODO: Randomize gate sizes?
            # TODO: place drone 2 randomly and take images again


def main(sid_path, level_name, category_seg_id_tab, save_dir, cam_mode):
    CVCameraDataset(sid_path, save_dir, cam_mode)
    CVCameraDataset.load_level(level_name, category_seg_id_tab)
    CVCameraDataset.record_data()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--sid_path', type=str, default=Path("levels_objects.xlsx"))
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard"], default="Soccer_Field_Medium")
    parser.add_argument('--category_id_tab', type=str, default=Path("SegmentIDs"))
    parser.add_argument('--save_dir', type=str, default=Path("results"))
    parser.add_argument('--cam_mode', type=str, choices=["single", "cont"], default="single")
    args = parser.parse_args()
    main(args.sid_path, args.level_name, args.seg_id_tab, args.save_dir, args.cam_mode)