from vcd import core, scl, types
from scipy.optimize import linear_sum_assignment
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from tqdm import tqdm
from typing import TypedDict, Dict, List, Tuple
import os



def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")


class ObjInfo(Dict):
    # Dict[core.UID,types.ObjectData]
    # Es mejor dejar que sea dinámico
    pass
class FramePresence(TypedDict):
    str: List[str] # Semantic-type - List of objects UIDs
class AnnotationInfo(TypedDict):
    objects: Dict[str, ObjInfo] # Semantic-type - ObjInfo
    frame_presence: Dict[int, FramePresence] # frame num - FramePresence

def get_gt_dt_inf(all_objects:ObjInfo, selected_types:List[str], filter_out:bool = False) -> Tuple[AnnotationInfo, AnnotationInfo]:
    """
    Return the information stored aboud the selected types of ground_truth and detection annotations. 
    If `filter_out` is set, the not selected types will not be returned. 

    AnnotationInfo format:
    ```
    {
        'objects': {
            'vehicle.car': {
                '0': {vcd},
                '1': {},
                '2': {}
            }
        },
        'frame_presence': {
            0: {'vehicle.car': ['0', '1']},
            1: {'vehicle.car': ['1']},
            2: {}
        }
    }
    
    ```
    """
    gt_objs = {'objects':{}, 'frame_presence': None} 
    dt_objs = {'objects':{}, 'frame_presence': None}

    # Distinguis between GT and Annotations
    for uid, obj in all_objects.items():
        tp = obj['type'] # str
        assert isinstance(tp, str)
        tps = tp.split('/')
        if 'annotated' in tps:
            # Custom annotations
            tp = tps[1] # NuLabel type  
            if tp not in dt_objs['objects']:
                dt_objs['objects'][tp] = {}
            dt_objs['objects'][tp].update({uid:obj})
        else:
            # Scene Ground_truth
            tp = '.'.join(tps).lower() # Convert type to NuLabel format
            if tp not in gt_objs['objects']:
                gt_objs['objects'][tp] = {}
            gt_objs['objects'][tp].update({uid:obj})
    
    # Check if the openlabel has groundtruth and annotations
    if len(gt_objs['objects'].keys()) == 0:
        raise Exception(f"There is no ground truth on the openLABEL")
    if len(dt_objs['objects'].keys()) == 0:
        raise Exception(f"There are no detections (custom annotations) on the openLABEL")
    
    # Check if the selected types are present in the ground_truth and annotations
    for tp in selected_types:
        if tp not in gt_objs['objects']:
            raise Exception(f"GT has no type {tp}")
        if tp not in dt_objs['objects']:
            raise Exception(f"DT (custom annotations) has no type {tp}")

    # Drop out the non selected objects   
    def filter_out(obj_dict:Dict[str, ObjInfo], selected_types:List[str]) -> Dict[str, ObjInfo]:
        dropping_keys = []
        for k in obj_dict.keys():
            if k not in selected_types:
                dropping_keys.append(k)
        for k in dropping_keys:
            obj_dict.pop(k)
        return obj_dict
    if filter_out:
        gt_objs['objects'] = filter_out(gt_objs['objects'], selected_types)
        dt_objs['objects'] = filter_out(dt_objs['objects'], selected_types)


    # Compute presence of objects in frames
    def get_frame_presence(obj_dict:ObjInfo) -> FramePresence:
        """
        Return the frame_presence dict:
        ```
        'frame_presence':{
            0:{ 'vehicle.car': [0, 1] },
            1:{ 'vehicle.car': [1] },
            2:{ 'vehicle.car': [] },
            3:{}
        }
        ```
        """
        frame_presence = {}
        for tp in obj_dict.keys():
            objs = obj_dict[tp]
            for uid, obj in objs.items():
                assert 'frame_intervals' in obj
                for interval in obj['frame_intervals']:
                    start   = interval['frame_start']
                    end     = interval['frame_end']
                    for fk in range(start, end+1):
                        if not fk in frame_presence:
                            frame_presence[fk] = {}
                        if not tp in frame_presence[fk]:
                            frame_presence[fk][tp] = []
                        frame_presence[fk][tp].append(uid)
        return frame_presence
    gt_objs['frame_presence'] = get_frame_presence(gt_objs['objects'])
    dt_objs['frame_presence'] = get_frame_presence(dt_objs['objects'])

    # Assert same number of continuous frames
    def get_frames_with_no_objs(obj_dict:AnnotationInfo, max_frame:int) -> List[int]:
        frames_with_no_objs  = []
        for fk in range(max_frame+1):
            if not fk in obj_dict['frame_presence']:
                obj_dict['frame_presence'][fk] = {}
                frames_with_no_objs.append(fk)
            elif len(list(obj_dict['frame_presence'][fk].keys())) == 0:
                frames_with_no_objs.append(fk)
            else:
                num_objs = 0
                for _, uids in obj_dict['frame_presence'][fk].items():
                    num_objs += len(uids)
                if num_objs == 0:
                    frames_with_no_objs.append(fk)
        return frames_with_no_objs

    max_frame_gt = list(gt_objs['frame_presence'].keys())[-1]
    max_frame_dt = list(dt_objs['frame_presence'].keys())[-1]
    max_frame = max(max_frame_gt, max_frame_dt)
    gt_frames_with_no_objs = get_frames_with_no_objs(gt_objs, max_frame)
    dt_frames_with_no_objs = get_frames_with_no_objs(dt_objs, max_frame)

    if len(gt_frames_with_no_objs):
        print(f"[GT]    Frames with no entries: {gt_frames_with_no_objs}")
    if len(dt_frames_with_no_objs):
        print(f"[DT]   Frames with no entries: {dt_frames_with_no_objs}")
    return gt_objs, dt_objs


def compute_cost_between_bboxes(bbox1: Tuple[float], bbox2: Tuple[float]) -> float:
    """
    bboxes on the same coords system.
    bbox1: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    bbox2: [ x, y, z, rx, ry, rz, sx, sy, sz ]
    """
    x1, y1, z1, rx1, ry1, rz1, sx1, sy1, sz1 = bbox1
    x2, y2, z2, rx2, ry2, rz2, sx2, sy2, sz2 = bbox2

    dist_2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) 
    dist = np.sqrt(dist_2)
    return dist

def get_cost_matrix(gt_bboxes:List[dict], dt_bboxes:List[dict], scene:scl.Scene, frame_num:int) -> np.ndarray:
    """
    Compute cost matrix with bboxes on 'odom' coordinate system
    {
        "name": "box3D",
        "coordinate_system": "odom",
        "val": [ x, y, z, rx, ry, rz, sx, sy, sz ]
    }
    """
    n = len(gt_bboxes)
    m = len(dt_bboxes)
    cost_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            bbox1, bbox2 = gt_bboxes[i]['val'], dt_bboxes[j]['val']
            if gt_bboxes[i]['coordinate_system'] != 'odom':
                bbox1 = scene.transform_cuboid(bbox1, cs_src=gt_bboxes[i]['coordinate_system'], cs_dst='odom', frame_num=frame_num)
            if dt_bboxes[j]['coordinate_system'] != 'odom':
                bbox2 = scene.transform_cuboid(bbox2, cs_src=dt_bboxes[j]['coordinate_system'], cs_dst='odom', frame_num=frame_num)

            cost_matrix[i][j] = compute_cost_between_bboxes(bbox1, bbox2)

    return cost_matrix

def show_cost_matrix(cost_matrix: np.ndarray, assignments:List[Tuple[int]], gt_labels: List[str], dt_labels: List[str], semantic_type:str, frame_num:int):
    plt.figure(figsize=(10, 8))
    wrapped_gt_labels = ["\n".join(textwrap.wrap(label, width=15)) for label in gt_labels]
    
    ax = sns.heatmap(
        cost_matrix,
        annot=True,  # show values
        fmt=".2f",  # value format
        cmap="coolwarm",  # color pallete
        linewidths=0.5,  # lines between cells
        xticklabels=dt_labels,
        yticklabels=wrapped_gt_labels,
        cbar_kws={'label': 'Cost'},
        annot_kws={"size": 8}
    )
    ax.set_xticklabels(dt_labels, fontsize=8)
    ax.set_yticklabels(wrapped_gt_labels, fontsize=8, rotation=0, ha="right")

    # Mark assingments
    for (i, j) in assignments:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    plt.ylabel("Ground Truth (gt)")
    plt.xlabel("Detections (dt)")
    plt.title(f"Cost Matrix for type: {semantic_type} in frame {frame_num}")
    plt.show()

def assign_detections_to_ground_truth(cost_matrix:np.ndarray) -> List[int]:
    """
    Cost matrix (N, M) where N is the number of GT elements and M is the
    number of detected elements (custom annotations). Returns the list of assignments
    """
    n, m = cost_matrix.shape
    
    # If it is not a square matrix, fill missing values with infinity
    max_dim = max(n, m)
    padded_cost_matrix = np.full((max_dim, max_dim), np.max(cost_matrix)+1)  # Valor alto
    padded_cost_matrix[:n, :m] = cost_matrix  # Insertamos la matriz original

    # Solve LSAP
    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)

    # Filtrar asignaciones inválidas (aquellas que caen en la parte añadida)
    valid_assignments = [(r, c) for r, c in zip(row_ind, col_ind) if r < n and c < m]

    assert len(valid_assignments) == min(n, m)
    return valid_assignments




def main(
        openlabel_path:str,
        semantic_types: List[str],
        debug=False
        ):
    # Check wheter the file exists
    check_paths([openlabel_path])

    # Load OpenLABEL
    vcd = core.OpenLABEL()
    vcd.load_from_file(openlabel_path)
    scene = scl.Scene(vcd)

    # Get the selected ground_truth and annotated objects 
    all_objs_inf = vcd.get_objects()
    gt_objs_inf, dt_objs_inf = get_gt_dt_inf(all_objs_inf, selected_types=semantic_types, filter_out=True)

    frame_keys = vcd.data['openlabel']['frames'].keys()
    for fk in tqdm(frame_keys, desc="frames"):
        gt_uids_in_frame = gt_objs_inf['frame_presence'][fk] # Ground_truth of frame
        dt_uids_in_frame = dt_objs_inf['frame_presence'][fk] # Detections of frame
        
        for tp in semantic_types:
            gt_uids = gt_uids_in_frame[tp]
            dt_uids = dt_uids_in_frame[tp]
            
            # cambiar box3d a bbox3d
            gt_objs_data = [ vcd.get_object_data(uid=uid, data_name='bbox3D', frame_num=fk) for uid in gt_uids]
            dt_objs_data = [ vcd.get_object_data(uid=uid, data_name='box3D', frame_num=fk) for uid in dt_uids]
            
            cost_matrix = get_cost_matrix(gt_objs_data, dt_objs_data, scene=scene, frame_num=fk)
            assignments = assign_detections_to_ground_truth(cost_matrix)
            print(cost_matrix)
            print(assignments)
            
            if debug:
                gt_labels = [gt_objs_inf['objects'][tp][uid]['name'] for uid in gt_uids]
                dt_labels = [dt_objs_inf['objects'][tp][uid]['name'] for uid in dt_uids]
                show_cost_matrix(cost_matrix, assignments, gt_labels, dt_labels, tp, fk)
            

            # ADD RELATIONS TO VISUALIZE IN WEBLABEL

            # print(f"frame: {fk}, semantic-type: {tp} first GT object data:")
            # uid_0 = gt_uids[0]
            # name_0 = gt_objs_inf['objects'][tp][uid_0]['name']
            # print(f"uid_0: {uid_0} name_0: {name_0}")


if __name__ == "__main__":
    OPENLABEL_PATH = "./tmp/my_scene/nuscenes_sequence/annotated_openlabel.json"
    semantic_types = [
        "vehicle.car"
    ]

    main(openlabel_path=OPENLABEL_PATH, semantic_types=semantic_types, debug=True)