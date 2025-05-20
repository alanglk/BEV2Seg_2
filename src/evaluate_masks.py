
# For computing metrics
from scripts.evaluate_bev2seg_2 import calculate_metrics
from oldatasets.Occ2.occ2labels import occ2id2name, occ2id2color

import numpy as np
import cv2

from typing import List
import pickle
import re
import os


def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")

def get_eval_mask_paths(folder_path: str, match_pattern: str = '') -> dict:
    """RETURN dict"""
    paths = {}
    aux = os.listdir(folder_path)
    for file_name in aux:
        if not file_name.endswith(".png"):
            continue
        if re.search(match_pattern, file_name):
            # frame_key: dt_occ_mask_FRAMEKEY.png
            frame_key_with_ext = file_name.split('_')[-1]
            frame_key = int( frame_key_with_ext.replace(".png", "") )
            paths[frame_key] = os.path.join(folder_path, file_name)
    return paths

def main(scene_path:str,
         output_path:str,
         reevaluate:bool = False):
    
    # Check paths
    check_paths([scene_path])
    scene_name = os.path.dirname(scene_path)
    dt_occ_path = os.path.join(scene_path, 'generated', 'dt_occ_bev_masks')
    gt_occ_path = os.path.join(scene_path, 'generated', 'gt_occ_bev_masks')
    check_paths([dt_occ_path, gt_occ_path])

    dt_paths = get_eval_mask_paths(dt_occ_path, 'dt_occ_mask')
    gt_paths = get_eval_mask_paths(gt_occ_path, 'gt_ras_mask')

    eval_fks = [] 
    for i in range( max( [max(list(dt_paths.keys())), max(list(gt_paths.keys()))] ) ):
        if i in dt_paths and i in gt_paths:
            eval_fks.append(i)

    print(f"There are {len(dt_paths.keys())} frames with detections and {len(gt_paths.keys())} frames with ground truth annotations")
    print(f"{eval_fks} fks will be evaluated")

    labels_list = list(occ2id2name.values())
    num_labels = len(labels_list)
    colors = list(occ2id2color.values())

    # Metric Output
    data = {}
    metric_names = [ 'mean_precision_per_class', 'mean_recall_per_class', 'mean_accuracy_per_class','mean_f1score_per_class', 'mean_iou_per_class' ]
    if os.path.exists(output_path):
        print(f"Loading evaluation data from: {output_path}")
        with open(output_path, "rb") as f:
            data = pickle.load(f)

        if scene_name in data:
            print(f"Scene {scene_name} is already evaluated")
            if not reevaluate:
                res = input(f"Do you want to evaluate it again? [Y/n]")
                if res.lower() != 'y':
                    print("Finish!! :D")
                    return
            else:
                print("Evaluating it again...")
    
    data[scene_name] = data[scene_name] if scene_name in data else {} 

    for fk in eval_fks:
        for m in metric_names:
            data[scene_name][fk][m] = np.zeros(num_labels)
        data[scene_name][fk]['conf_matrix'] = np.zeros((num_labels, num_labels))
    data[scene_name]['labels'] = labels_list
    data[scene_name]['colors'] = colors

    
    for fk in eval_fks:
        dt_mask = cv2.imread(dt_paths[fk])
        gt_mask = cv2.imread(gt_paths[fk])

        pre_per_class, rec_per_class, acc_per_class, f1_per_class, iou_per_class, conf_matrix = calculate_metrics(gt_mask[:, :, 0], dt_mask[:, :, 0], num_classes=num_labels)
        data[scene_name][fk]['mean_precision_per_class']    = pre_per_class
        data[scene_name][fk]['mean_recall_per_class']       = rec_per_class
        data[scene_name][fk]['mean_accuracy_per_class']     = acc_per_class
        data[scene_name][fk]['mean_f1score_per_class']      = f1_per_class
        data[scene_name][fk]['mean_iou_per_class']          = iou_per_class
        data[scene_name][fk]['conf_matrix']                 = conf_matrix
    
    # Save results
    print(f"Shaving evaluation data in: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print("Finish!! :D")

if __name__ == "__main__":
    SCENE_PATH = "./tmp/my_scene"
    OUTPUT_PATH = "./data/pipeline_masks_evaluations.pkl"
    main(scene_path=SCENE_PATH, output_path=OUTPUT_PATH)
