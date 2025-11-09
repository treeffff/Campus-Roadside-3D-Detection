import os
import json

path = "/root/autodl-tmp/BEVHeight/TRY/results_nusc.json"
output_path = "/root/autodl-tmp/BEVHeight/TRY/temp"


with open(path, 'r') as f:
    results = json.load(f)

img_ids = list(results['results'].keys())
results = results['results']
# print(results['results'][img_ids[0]][0])

def gen_empty_dict():
    anno = {}
    anno['obj_id'] =  "1"
    anno['obj_type'] = "Car"
    anno['psr'] = {'position': {"x": 0, "y":0, "z":0}, 'rotation': {"x":0, "y":0, "z":0}, 'scale': {"x":0, "y":0, "z":0}}
    anno['obj_score'] = 0
    return anno

names = set()

for img_id in img_ids:
    token = img_id.split('/')[-1].split('.')[0]
    annos = []
    infos = results[img_id]
    for info in infos:
        if info['detection_score'] < 0.45:
            continue
        anno = gen_empty_dict()
        anno['obj_type'] = info['detection_name'].title()
        names.add(anno['obj_type'])
        anno['psr']['position']['x'] = info['translation'][0]
        anno['psr']['position']['y'] = info['translation'][1]
        anno['psr']['position']['z'] = info['translation'][2] +info['size'][2] / 2
        anno['psr']['rotation']['z'] = info['box_yaw']
        # size: w, l, h
        anno['psr']['scale']['x'] = info['size'][0]
        anno['psr']['scale']['y'] = info['size'][1]
        anno['psr']['scale']['z'] = info['size'][2]
        anno['obj_score'] = info['detection_score']

        annos.append(anno)

    output_name = os.path.join(output_path, token+'.json')
    with open(output_name, "w") as f:
        json.dump(annos, f, indent=4)
    print(f"Saved {output_name}")
print(names)