# For airsim.exe v 0.2

import airsimneurips as airsim
import numpy as np
import json

client = airsim.MultirotorClient()
client.confirmConnection()

id2rgb = {}
rgb2id = {}

for id in range(256):
    client.simSetSegmentationObjectID("Mesh_soccerField_3", id)
    responses = client.simGetImages([
                # floating point uncompressed image
                airsim.ImageRequest("fpv_cam", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False),
                ])

    seg_response = responses[0]
    seg_1d = np.frombuffer(seg_response.image_data_uint8, dtype=np.uint8)
    img_seg = seg_1d.reshape(seg_response.height, seg_response.width, 3)
    midpt = (int(seg_response.height/2), int(seg_response.height/2))
    id_rgb = img_seg[int(seg_response.height/2), int(seg_response.height/2), :].astype(int).tolist()
    id2rgb[id] = id_rgb

    if id_rgb[0] not in rgb2id.keys():
        rgb2id[id_rgb[0]] = {id_rgb[1]: {id_rgb[2]: id}}

    elif id_rgb[1] not in rgb2id[id_rgb[0]].keys():
        rgb2id[id_rgb[0]][id_rgb[1]] = {id_rgb[2]: id}

    else:
        rgb2id[id_rgb[0]][id_rgb[1]][id_rgb[2]] = id

for id in range(256):
    test_rgb = id2rgb[id]
    test_id = rgb2id[test_rgb[0]][test_rgb[1]][test_rgb[2]]
    if id != test_id:
        raise ValueError(f'{id} has RBG {test_rgb} from sim but rgb2id has ID {test_id}')

print("Verification success!")

results = {"id2rgb": id2rgb, "rgb2id": rgb2id}

with open('segmentation_id_maps.json', 'w') as f:
    json.dump(results, f, indent=4, separators=[",", ": "])