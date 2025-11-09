'''
Modified from https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py
'''
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import os
import json

__all__ = ['RoadSideEvaluator']


class RoadSideEvaluator():
    def __init__(
        self,
        class_names,
        current_classes,
        data_root,
        modality=dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False),
        output_dir=None,
    ) -> None:
        self.class_names = class_names
        self.current_classes = current_classes
        self.data_root = data_root
        self.modality = modality
        self.output_dir = output_dir

    def format_results(self,
                       results,
                       img_metas,
                       result_names=['img_bbox'],
                       jsonfile_prefix=None,
                       **kwargs):
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = dict()
        for rasult_name in result_names:
            if '2d' in rasult_name:
                continue
            print(f'\nFormating bboxes of {rasult_name}')
            tmp_file_ = osp.join(jsonfile_prefix, rasult_name)
            if self.output_dir:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, self.output_dir)
                })
            else:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, tmp_file_)
                })
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        img_metas,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        result_names=['img_bbox'],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        result_files, tmp_dir = self.format_results(results, img_metas,
                                                    result_names,
                                                    jsonfile_prefix)
        print(result_files, tmp_dir)
        print("complete format results")
        # results_path = "outputs" 
        # if 'dair' in self.data_root:
        #     pred_label_path = result2kitti(result_files["img_bbox"], results_path, self.data_root, self.gt_label_path, demo=False)
        # elif 'school' in self.data_root:
        #     pred_label_path = result2kitti(result_files["img_bbox"], results_path, self.data_root, self.gt_label_path, demo=False)
        # else:
        #     pred_label_path = result2kitti_rope3d(result_files["img_bbox"], results_path, self.data_root, self.gt_label_path, demo=False)
        # kitti_evaluation(pred_label_path, self.gt_label_path, current_classes=self.current_classes, metric_path="outputs/metrics")

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            sample_token = img_metas[sample_id]['token']
            trans = np.array(img_metas[sample_id]['ego2global_translation'])
            rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)

                attr = name
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    box_yaw=box_yaw,
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)

        convered_res_path = "prediction_result/result"
        mmcv.mkdir_or_exist(convered_res_path)
        convert_results_to_annos(res_path, convered_res_path, score_threshold=0.45)
        return res_path


def convert_results_to_annos(input_json_path, output_dir, score_threshold=0.45):
    """
    将results_nusc.json转换成单个图片的json文件。
    
    Args:
        input_json_path (str): 输入的results.json路径。
        output_dir (str): 输出目录，会自动创建。
        score_threshold (float): 检测分数阈值，低于该值的目标会被过滤。
    
    Returns:
        set: 所有出现过的obj_type名称集合。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_json_path, 'r') as f:
        results_data = json.load(f)

    img_ids = list(results_data['results'].keys())
    results = results_data['results']

    def gen_empty_dict():
        anno = {}
        anno['obj_id'] = "1"
        anno['obj_type'] = "Car"
        anno['psr'] = {
            'position': {"x": 0, "y": 0, "z": 0},
            'rotation': {"x": 0, "y": 0, "z": 0},
            'scale': {"x": 0, "y": 0, "z": 0}
        }
        anno['obj_score'] = 0
        return anno

    names = set()

    for img_id in img_ids:
        token = os.path.splitext(os.path.basename(img_id))[0]
        annos = []
        infos = results[img_id]
        for info in infos:
            if info['detection_score'] < score_threshold:
                continue
            anno = gen_empty_dict()
            anno['obj_type'] = info['detection_name'].title()
            names.add(anno['obj_type'])
            anno['psr']['position']['x'] = info['translation'][0]
            anno['psr']['position']['y'] = info['translation'][1]
            anno['psr']['position']['z'] = info['translation'][2] + info['size'][2] / 2
            anno['psr']['rotation']['z'] = info['box_yaw']
            anno['psr']['scale']['x'] = info['size'][0]
            anno['psr']['scale']['y'] = info['size'][1]
            anno['psr']['scale']['z'] = info['size'][2]
            anno['obj_score'] = info['detection_score']

            annos.append(anno)

        output_name = os.path.join(output_dir, token + '.json')
        with open(output_name, "w") as f:
            json.dump(annos, f, indent=4)
        print(f"Saved {output_name}")
    return names