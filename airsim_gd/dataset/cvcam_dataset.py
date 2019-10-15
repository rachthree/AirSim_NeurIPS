import os
import time
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
    pose_dict = {"t": t, "x": pose.position.x_val, "y": pose.position.y_val, "z": pose.position.z_val,
                 "qw": pose.orientation.w_val, "qx": pose.orientation.x_val, "qy": pose.orientation.y_val, "qz": pose.orientation.z_val}
    return pose_dict

def pose_to_array(pose):
    pose_array = [pose.position.x_val, pose.position.y_val, pose.position.z_val,
                  pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val]
    return pose_array

class CVCameraDataset(object):
    def __init__(self, sid_spreadsheet_loc=Path("airsim_gd/dataset/levels_objects.xlsx"),
                 save_dir=Path.cwd().parent.joinpath("data"),
                 cam_mode="single",
                 sess_name=time.ctime(time.time()).replace(':', '.')):
        self.client = setupASClient()
        self.sid_spreadsheet_loc = Path(sid_spreadsheet_loc)
        self.save_dir = Path(save_dir)
        self.cam_mode = cam_mode
        self.pose_history_df = None
        self.sess_name = sess_name
        self.level_name = ""

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_level(self, level_name):
        if level_name not in level_list:
            raise ValueError("Invalid level. Choose one from {}".format(level_list))

        self.client.simLoadLevel(level_name)
        self.level_name = level_name

        # load segment ID's
        df = pd.read_excel(self.sid_spreadsheet_loc, sheet_name=level_name)
        for obj, seg_id in zip(df['object'], df['segment_ID']):
            self.client.simSetSegmentationObjectID(obj, seg_id)

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

    def generate_data(self, vehicle_name="drone_1", csv_path=None, fly_region_start_id=101, gate_start_id=1, gate_lim=10):
        # load json or use attribute
        if csv_path is None:
            history_df = self.pose_history_df
        else:
            history_df = pd.read_csv(csv_path)

        # At each position, place camera/drone
        for row in history_df.itertuples():
            # At each position, place camera/drone
            x = row.x
            y = row.y
            z = row.z
            qw = row.qw
            qx = row.qx
            qy = row.qy
            qz = row.qz
            pose = airsim.Pose(position_val={'x_val': x, 'y_val': y, 'z_val': z},
                               orientation_val={'w_val': qw, 'x_val': qx, 'y_val': qy, 'z_val': qz})
            self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name)
            # Take images from fpv_cam, back, starboard, port, bottom
            # RGB, Segmentation, Depth


            # place drone 2 randomly

            # Take images again

            # Random rotation among 4 quadrants of drone POV... need to do transformations!
            # Determine flyable region without occlusion... need projection!
            # Determine which part of the flyable region can actually be seen using the segment ID's... occlusion!
            # Define areas to fill with flyable regions
            # Determine closest gates given segment ID's -> gate number -> gate location in sim.
            # New segment ID in increasing distance.
            # New alpha channel in PNG.
            # Save depths as PFM


def main(sid_path, level_name, save_dir, cam_mode):
    CVCameraDataset(sid_path, save_dir, cam_mode)
    CVCameraDataset.load_level(level_name)
    CVCameraDataset.record_data()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--sid_path', type=str, default=Path("levels_objects.xlsx"))
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard"], default="Soccer_Field_Medium")
    parser.add_argument('--save_dir', type=str, default=Path("results"))
    parser.add_argument('--cam_mode', type=str, choices=["single", "cont"], default="single")
    args = parser.parse_args()
    main(args.sid_path, args.level_name, args.save_dir, args.cam_mode)