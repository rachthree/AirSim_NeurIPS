import os
from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd
from argparse import ArgumentParser

from airsim_gd.vision.utils import setupASClient
from airsim_gd.dataset.utils import imagery

level_list = ["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard"]


class CVCameraDataset(object):
    def __init__(self, sid_spreadsheet_loc=Path("airsim_gd/dataset/levels_objects.xlsx"),
                 save_dir=Path.cwd().parent.joinpath("data"),
                 cam_mode="single"):
        self.client = setupASClient()
        self.sid_spreadsheet_loc = Path(sid_spreadsheet_loc)
        self.save_dir = Path(save_dir)
        self.cam_mode = cam_mode

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_level(self, level_name):
        if level_name not in level_list:
            raise ValueError("Invalid level. Choose one from {}".format(level_list))

        self.client.simLoadLevel(level_name)

        # load segment ID's
        df = pd.read_excel(self.sid_spreadsheet_loc, sheet_name=level_name)
        for obj, seg_id in zip(df['object'], df['segment_ID']):
            self.client.simSetSegmentationObjectID(obj, seg_id)

    def record_path(self):
        # Record camera path
        # Save position and quaternion
        # Eliminate duplicates
        # While loop, but ctrl-c out gracefully
        # get rid of duplicates at end
        # write out json
        # save attribute
        pass

    def generate_data(self, vehicle_name="drone_1", path_json=None, fly_region_start_id=101, gate_start_id=1, gate_lim=10):
        # load json or use attribute
        # At each position, place camera/drone
        # Take images from fpv_cam, back, starboard, port, bottom
        # RGB, Segmentation, Depth
        # Random rotation among 4 quadrants of drone POV... need to do transformations!
        # Determine flyable region without occlusion... need projection!
        # Determine which part of the flyable region can actually be seen using the segment ID's... occlusion!
        # Define areas to fill with flyable regions
        # Determine closest gates given segment ID's -> gate number -> gate location in sim.
        # New segment ID in increasing distance.
        # New alpha channel in PNG.
        # Save depths as PFM

        pass

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