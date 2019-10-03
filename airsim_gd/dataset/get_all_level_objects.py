import json
from airsim_gd.vision.utils import setupASClient

level_list = ['Soccer_Field_Medium', 'ZhangJiaJie_Medium', 'Building99_Hard']
level_objects_dict = {}
client = setupASClient()

for level in level_list:
    response = client.simLoadLevel(level)
    if response:
        level_objects_dict[level] = client.simListSceneObjects()


with open("levels_objects.json", "w") as f:
    json.dump(level_objects_dict, f,
              sort_keys=True,
              indent=4, separators=(',', ': '))
