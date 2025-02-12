import os
import time
import json
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import csv
from argparse import ArgumentParser
from collections import defaultdict
import random
import glob

import numpy as np
import cv2
import pandas as pd

import airsimneurips as airsim
from airsim_gd.vision.utils import setupASClient
from airsim_gd.dataset.utils import imagery
from airsim_gd.dataset.utils import data


level_list = ["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard"]


def pose_to_dict(pose, t=0.0):
    return {"t": t, "x": pose.position.x_val, "y": pose.position.y_val, "z": pose.position.z_val,
            "qw": pose.orientation.w_val, "qx": pose.orientation.x_val, "qy": pose.orientation.y_val, "qz": pose.orientation.z_val}


def pose_to_array(pose):
    return [pose.position.x_val, pose.position.y_val, pose.position.z_val,
            pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val]


def rgb_to_id(pix_array, rgb2id_dict):
    return rgb2id_dict[tuple(pix_array)]


def id_to_rgb(seg_id, id2rgb_dict):
    # to be used with np.apply_along_axis with the segmentation ID array
    seg_id = seg_id if isinstance(seg_id, int) else seg_id[0]
    rgb = id2rgb_dict[seg_id]
    return np.array([rgb[0], rgb[1], rgb[2]])


def rgb2id_seg_img(rgb_seg_img, rgb2id_dict):
    return np.apply_along_axis(rgb_to_id, 2, rgb_seg_img, rgb2id_dict)


def id2rgb_seg_img(id_seg_img, id2rgb_dict):
    return np.array(np.apply_along_axis(id_to_rgb, 2, id_seg_img[:, :, np.newaxis], id2rgb_dict)).astype(np.uint8)


def json_list_to_mapping(seg_id_map_list):
    id2rgb = {x['id']: tuple(x['rgb']) for x in seg_id_map_list}
    rgb2id = {tuple(x['rgb']): x['id'] for x in seg_id_map_list}
    return id2rgb, rgb2id


class CVCameraDataset(object):
    def __init__(self, sid_spreadsheet_loc=Path("airsim_gd/dataset/levels_objects.xlsx"),
                 save_dir=Path.cwd().parent.joinpath("data"),
                 cam_mode="single",
                 sess_name=time.ctime(time.time()).replace(':', '.'),
                 cameras=["fpv_cam", "aft", "starboard", "port", "bottom"],
                 seg_id_map_path=Path("airsim_gd/dataset/segmentation_id_maps.json"),
                 gate_label_basename='gate',
                 fly_region_label_basename='flyregion',
                 max_depth=None,
                 vehicle_name='drone_1',
                 sim_seg_id_list_path=Path("airsim_gd/dataset/sim_seg_id.json")):
        self.sid_spreadsheet_loc = Path(sid_spreadsheet_loc)
        self.save_dir = Path(save_dir)
        self.cam_mode = cam_mode
        self.pose_history_df = None
        self.sess_name = sess_name
        self.level_name = ""
        self.cameras = cameras
        self.max_depth = max_depth  # meters
        self.vehicle_name = vehicle_name

        with open(Path(seg_id_map_path)) as f:
            seg_id_map_list = json.load(f)

        sim_seg_id_list_path = Path(sim_seg_id_list_path)
        if sim_seg_id_list_path.is_file():
            with open(sim_seg_id_list_path) as f:
                self.seg_id_list = json.load(f)
        else:
            self.seg_id_list = []

        self.seg_id_sess = np.array([])

        self.id2rgb, self.rgb2id = json_list_to_mapping(seg_id_map_list)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gate_label_basename = gate_label_basename
        self.fly_region_label_basename = fly_region_label_basename

        self.seg_lvlobj2id = {}
        self.seg_id2lvlobjs = defaultdict(list)
        self.seg_id2class = {}
        self.seg_class2id = {}

    def load_level(self, level_name, category_seg_id_tab):
        self.client = setupASClient()

        if level_name not in level_list:
            raise ValueError("Invalid level. Choose one from {}".format(level_list))

        self.client.simLoadLevel(level_name)
        self.level_name = level_name

        # load segment ID's - level specific
        df = pd.read_excel(self.sid_spreadsheet_loc, sheet_name=level_name)
        for obj, seg_id in zip(df['object'], df['segment_ID']):
            self.client.simSetSegmentationObjectID(obj, seg_id)
            self.seg_lvlobj2id[obj] = seg_id
            self.seg_id2lvlobjs[seg_id].append(obj)

        # load segment ID's - general
        df = pd.read_excel(self.sid_spreadsheet_loc, sheet_name=category_seg_id_tab)
        for category, seg_id in zip(df['class'], df['segment_ID']):
            self.seg_id2class[seg_id] = category
            self.seg_class2id[category] = seg_id

    def record_data(self, interval=0.05):
        t_start = time.clock()
        t0 = t_start
        pose_history = []
        pose_temp = self.client.simGetVehiclePose(self.vehicle_name)
        pose_history.append(pose_to_dict(pose_temp, t0 - t_start))  # zero out time
        pose_array0 = pose_to_array(pose_temp)
        try:
            while True:
                t1 = time.clock()
                if t1 - t0 >= interval:
                    pose_temp = self.client.simGetVehiclePose(self.vehicle_name)
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
            history_df.to_csv(csv_savepath, sep=',', encoding='utf-8', index=False)
            print(f"Saved positional time history to {csv_savepath}.")

            print('Generating data...')
            self.generate_data()

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
        img_seg_rgb_cam = {}
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
                airsim.ImageRequest(camera, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
            ])

            rgb_response = responses[0]
            img_rgb_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb_1d.reshape(rgb_response.height, rgb_response.width, 3)
            img_rgb_cam[camera] = img_rgb

            seg_response = responses[1]
            seg_1d = np.frombuffer(seg_response.image_data_uint8, dtype=np.uint8)
            img_seg = seg_1d.reshape(seg_response.height, seg_response.width, 3)
            img_seg_rgb_cam[camera] = img_seg
            img_seg_id_cam[camera] = rgb2id_seg_img(img_seg, self.rgb2id)

            depth_response = responses[2]
            depth_1d = np.array(depth_response.image_data_float)
            img_depth = depth_1d.reshape(depth_response.height, depth_response.width)
            if self.max_depth:
                img_depth[img_depth > self.max_depth] = self.max_depth

            img_depth_cam[camera] = img_depth

        images_dict['rgb'] = img_rgb_cam
        images_dict['seg_rgb'] = img_seg_rgb_cam
        images_dict['seg_id'] = img_seg_id_cam
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
            seg_image = np.copy(images_dict['seg_id'][camera])
            depth_image = images_dict['depth'][camera]

            gate_list = imagery.get_visible_gates(seg_image, self.seg_id2lvlobjs)
            camera_info = self.client.simGetCameraInfo(camera_name=camera)
            sorted_gate_info = imagery.get_sorted_gates(self.client, camera_info, gate_list)

            gate_mask_list = []
            for gate_name, _ in sorted_gate_info:
                # Doing this outside for loop to prevent multiple gates being edited at once
                gate_mask_list.append(seg_image == self.seg_lvlobj2id[gate_name])

            for i in range(len(gate_mask_list)):
                # Relabeling gates before occlusion-accounting flyable region labelling
                # only 1-n will be labels
                images_dict['seg_id'][camera][gate_mask_list[i]] = self.seg_class2id[f"{self.gate_label_basename}_{str(i + 1)}"]

            i = len(sorted_gate_info)
            for (gate_name, _) in sorted_gate_info[::-1]:
                # Reversed to account for overlapping flyable regions
                # Process chain
                region_label = f"{self.fly_region_label_basename}_{str(i)}"
                img_hw = seg_image.shape

                fly_region_global = imagery.get_flyable_region(self.client, gate_name)
                while np.isnan(np.sum(list(fly_region_global.values()))):
                    print('AirSim returned invalid values for gate scale/position... retrying... ')
                    fly_region_global = imagery.get_flyable_region(self.client, gate_name)

                fly_region_cam = imagery.project_global2cam_fly_region(fly_region_global, camera_info, img_hw)

                valid_pt = True
                for pt in fly_region_cam.values():
                    if pt is None:
                        valid_pt = False

                if valid_pt:
                    images_dict['seg_id'][camera] = imagery.segment_flyable_region(region_label,
                                                                                   images_dict['seg_id'][camera],
                                                                                   depth_image,
                                                                                   fly_region_global,
                                                                                   fly_region_cam,
                                                                                   camera_info,
                                                                                   self.seg_class2id)
                i -= 1

            self.seg_id_sess.append(np.unique(images_dict['seg_id'][camera]))

    def get_labeled_images(self, img_name):
        images_dict = self.take_images_dataset()
        self.get_flyable_region(images_dict)

        for camera in self.cameras:
            filename = img_name + '_' + camera
            # save image
            cv2.imwrite(str(Path(self.save_dir, filename + '.jpg')), images_dict['rgb'][camera]) # cv2.cvtColor(images_dict['rgb'][camera], cv2.COLOR_RGB2BGR))

            # save mask separately
            np.save(Path(self.save_dir, filename + '_mask.npy'), images_dict['seg_id'][camera])
            cv2.imwrite(str(Path(self.save_dir, filename + '_mask.jpg')), images_dict['seg_id'][camera])
            cv2.imwrite(str(Path(self.save_dir, filename + '_mask_check.jpg')), id2rgb_seg_img(images_dict['seg_id'][camera], self.id2rgb))
                        # cv2.cvtColor(id2rgb_seg_img(images_dict['seg_id'][camera], self.id2rgb), cv2.COLOR_RGB2BGR))

            # save depths
            airsim.utils.write_pfm(Path(self.save_dir, filename + '_depth.pfm'), images_dict['depth'][camera].astype(np.float32))
            np.save(Path(self.save_dir, filename + '_depth.npy'), images_dict['depth'][camera])

    def generate_data(self, csv_path=None, gate_lim=10):
        # load json or use attribute
        history_df = self.pose_history_df if csv_path is None else pd.read_csv(csv_path, sep=',')
        # At each position, place camera/drone
        for row in history_df.itertuples():
            print(f"Processing t = {row.t}...")
            # At each position, place camera/drone
            pose = airsim.Pose(position_val={'x_val': row.x, 'y_val': row.y, 'z_val': row.z},
                               orientation_val={'w_val': row.qw, 'x_val': row.qx, 'y_val': row.qy, 'z_val': row.qz})
            self.client.simSetVehiclePose(pose, ignore_collison=True, vehicle_name=self.vehicle_name)

            img_name = self.sess_name + f"_{row.t:0.3f}_0deg"
            # Take initial images
            self.get_labeled_images(img_name)

            # TODO: place drone 2 randomly and take images again
            # TODO: Random rotation among 4 quadrants of drone POV...
            # use airsim.utils.to_eularian_angles(pose.orientation), adjust, use airsim.utils.to_quaternion(roll, pitch, yaw)
            # euler_ang0 = airsim.utils.to_eularian_angles(pose.orientation)

            # TODO: Randomize gate sizes?
            # TODO: place drone 2 randomly and take images again

        # Update master list of sim seg ID's
        self.seg_id_sess = np.unique(self.seg_id_sess)
        master_set = set(self.seg_id_list)
        for seg_id in self.seg_id_sess:
            if seg_id not in master_set:
                self.seg_id_list.append(seg_id)

        filename = self.sim_seg_id_list_path.name.replace('.json', '_updated.json')
        with open(Path(self.sim_seg_id_list_path.parent, filename), 'w') as f:
            json.dump(self.seg_id_list, f)

    @staticmethod
    def _copy_data(img_list, new_dir):
        for img_path in img_list:
            img_path = Path(img_path)
            mask_path = Path(str(img_path).replace('.jpg', '_mask.npy'))
            depth_path = Path(str(img_path).replace('.jpg', '_depth.npy'))
            copyfile(img_path, Path(new_dir, img_path.name))
            copyfile(mask_path, Path(new_dir, mask_path.name))
            copyfile(depth_path, Path(new_dir, depth_path.name))

    @staticmethod
    def split_data(save_dir, val_percent=0.15, test_percent=0.15, random_seed=None,
                   cameras=["fpv_cam", "aft", "starboard", "port", "bottom"]):
        if random_seed is not None:
            random.seed(random_seed)

        if val_percent + test_percent == 1.0:
            raise ValueError('Validation and test set cannot be the full dataset')

        data_dir = Path(save_dir)
        train_dir = data_dir / 'train'
        train_dir.mkdir(exist_ok=True)
        val_dir = data_dir / 'val'
        val_dir.mkdir(exist_ok=True)
        test_dir = data_dir / 'test'
        test_dir.mkdir(exist_ok=True)

        # need same distributions across cameras
        for camera in cameras:
            camera_image_list = glob.glob(str(Path(save_dir, f'*{camera}.jpg')))
            n_images = len(camera_image_list)
            train_split = int((1 - val_percent - test_percent) * n_images)
            val_split = train_split + int(val_percent * n_images)

            random.shuffle(camera_image_list)
            train_list = camera_image_list[:train_split]
            val_list = camera_image_list[train_split:val_split]
            test_list = camera_image_list[val_split:]

            CVCameraDataset._copy_data(train_list, train_dir)
            CVCameraDataset._copy_data(val_list, val_dir)
            CVCameraDataset._copy_data(test_list, test_dir)


def main(args):
    datagen = CVCameraDataset(sid_spreadsheet_loc=args.sid_path, seg_id_map_path=args.seg_id_map_path, save_dir=args.save_dir, cam_mode=args.cam_mode,
                              vehicle_name=args.vehicle_name,
                              sess_name=args.sess_name)
    mode = args.mode
    if mode == 'record':
        datagen.load_level(args.level_name, args.category_id_tab)
        datagen.record_data()

    elif mode == 'gen_data':
        datagen.load_level(args.level_name, args.category_id_tab)
        datagen.generate_data(csv_path=args.csv_path)

    elif mode == 'split_data':
        datagen.split_data(args.save_dir, val_percent=args.val_percent, test_percent=args.test_percent, random_seed=args.random_seed)

    elif mode == 'convert':
        data.save_tf_records(Path(args.save_dir, 'train'))
        data.save_tf_records(Path(args.save_dir, 'val'))
        data.save_tf_records(Path(args.save_dir, 'test'))

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--sid_path', type=str, default=Path('levels_objects.xlsx'))
    parser.add_argument('--level_name', type=str, choices=['Soccer_Field_Easy', 'Soccer_Field_Medium', 'ZhangJiaJie_Medium', 'Building99_Hard'], default='Soccer_Field_Medium')
    parser.add_argument('--category_id_tab', type=str, default='SegmentIDs')
    parser.add_argument('--save_dir', type=str, default=Path.cwd().parent.joinpath("data"))
    parser.add_argument('--cam_mode', type=str, choices=['single', 'cont'], default='cont')
    parser.add_argument('--mode', type=str, choices=['record', 'gen_data', 'split_data', 'convert'])
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--vehicle_name', type=str, default='drone_1')
    parser.add_argument('--sess_name', type=str, default=time.ctime(time.time()).replace(':', '.'))
    parser.add_argument('--val_percent', type=float, default=0.15)
    parser.add_argument('--test_percent', type=float, default=0.15)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--seg_id_map_path', type=str, default=Path("airsim_gd/dataset/segmentation_id_maps.json"))
    parser.add_argument('--sim_seg_id_list_path', type=int, default=Path('sim_seg_id_list.json'))
    args = parser.parse_args()
    main(args)
