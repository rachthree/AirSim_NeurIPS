# Dataset Generation
This part of the module helps generate a dataset to use with training. The general process is:
1. Open AirSim in Computer Vision Camera mode (see `settings_dataset.json` in the settings directory) with the level loaded.
2. Record a path or positions of the drone.
    * `dataset_racer.py` is a modified `baseline_racer.py` from the competition package and can be used to create a nominal time history.
    * `cvcam_dataset.py` can be used to record a path using the `--mode=record` option.
3. Use the time history to generate data (images, depth info, labels).
    * ex: `python airsim_gd\dataset\cvcam_dataset.py --sid_path="airsim_gd\dataset\levels_objects.xlsx" --csv_path="airsim_gd\data\Soccer_Field_Medium_Nominal.csv" --mode=gen_data --sess_name=soccer_field_med_nominal`
4. Examine the data.
5. Split the data into train/val/test sets.
    * ex: `python airsim_gd\dataset\cvcam_dataset.py --mode=split_data --sess_name=soccer_field_med_nominal`
6. Examine the data in each set to determine good representation.
7. Convert the data into TFRecords.
    * ex: `python airsim_gd\dataset\cvcam_dataset.py --mode=convert --sess_name=soccer_field_med_nominal`

The labels are:
* Instanced pixel segmentation labels. Additional calculation is done to label the flyable regions of each gate.
  * The IDs are based on `airsim_gd\dataset\levels_objects.xlsx`, which helps map the AirSim object IDs into dataset labels. `cvcam_dataset.py` relabels objects to those specified in the spreadsheet. Ideally this would be in CSV format but this was easiest for development.
* Depth map in meters. The `--max_depth=<float>` flag for `cvcam_dataset.py` can be used to saturate the depths, which may help for training models.
