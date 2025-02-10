from vcd import core

from tqdm import tqdm
from typing import List, Tuple
import os

from oldatasets.NuImages.nulabels import nuname2label

def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")


def get_gt_ann_inf(all_objects:dict, selected_types:List[str], filter_out:bool = False) -> Tuple[dict, dict]:
    """
    Return the information stored aboud the selected types of ground_truth and annotations. 
    If `filter_out` is set, the not selected types will not be returned. 

    dict format:
    ```
    {
        'objects':{
            'vehicle.car':{
                '0':{...},
                '1':{...}
                '2':{...}
            }
        },
        'frame_presence':{
            0:{
                'vehicle.car': [0, 1]
            },
            1:{
                'vehicle.car': [1]
            },
            2:{}
        }
    }
    
    ```
    """
    gt_objs = {'objects':{}, 'frame_presence': None} 
    ann_objs = {'objects':{}, 'frame_presence': None}

    # Distinguis between GT and Annotations
    for uid, obj in all_objects.items():
        tp = obj['type'] # str
        assert isinstance(tp, str)
        tps = tp.split('/')
        if 'annotated' in tps:
            # Custom annotations
            tp = tps[1] # NuLabel type  
            if tp not in ann_objs['objects']:
                ann_objs['objects'][tp] = []
            ann_objs['objects'][tp].append({uid:obj})
        else:
            # Scene Ground_truth
            tp = '.'.join(tps).lower() # Convert type to NuLabel format
            if tp not in gt_objs['objects']:
                gt_objs['objects'][tp] = []
            gt_objs['objects'][tp].append({uid:obj})
    
    # Check if the openlabel has groundtruth and annotations
    if len(gt_objs['objects'].keys()) == 0:
        raise Exception(f"There is no ground truth on the openLABEL")
    if len(ann_objs['objects'].keys()) == 0:
        raise Exception(f"There are no annotations on the openLABEL")
    
    # Check if the selected types are present in the ground_truth and annotations
    for tp in selected_types:
        if tp not in gt_objs['objects']:
            raise Exception(f"GT has no type {tp}")
        if tp not in ann_objs['objects']:
            raise Exception(f"Annotations has no type {tp}")

    # Drop out the non selected objects   
    def filter_out(obj_dict:dict, selected_types:List[str]) -> dict:
        dropping_keys = []
        for k in obj_dict.keys():
            if k not in selected_types:
                dropping_keys.append(k)
        for k in dropping_keys:
            obj_dict.pop(k)
        return obj_dict
    if filter_out:
        gt_objs['objects']  = filter_out(gt_objs['objects'], selected_types)
        ann_objs['objects'] = filter_out(ann_objs['objects'], selected_types)


    # Compute presence of objects in frames
    def get_frame_presence(obj_dict:dict) -> dict:
        """
        Return the frame_presence dict:
        ```
        'frame_presence':{
            0:{ 'vehicle.car': [0, 1] },
            1:{ 'vehicle.car': [1] },
            2:{}
        }
        ```
        """
        frame_presence = {}
        for tp in obj_dict.keys():
            objs = obj_dict[tp]
            for obj in objs:
                uid = list(obj.keys())[0]
                obj = obj[uid]
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

    gt_objs['frame_presence']  = get_frame_presence(gt_objs['objects'])
    ann_objs['frame_presence'] = get_frame_presence(ann_objs['objects'])
    return gt_objs, ann_objs


def main(
        openlabel_path:str,
        types: List[str]
        ):
    # Check wheter the file exists
    check_paths([openlabel_path])

    # Load OpenLABEL
    vcd = core.OpenLABEL()
    vcd.load_from_file(openlabel_path)

    # Get the selected ground_truth and annotated objects 
    all_objs_inf = vcd.get_objects()
    gt_objs_inf, ann_objs_inf = get_gt_ann_inf(all_objs_inf, selected_types=types, filter_out=True)

    frame_keys = vcd.data['openlabel']['frames'].keys()
    for fk in tqdm(frame_keys, desc="frames"):
        frame = vcd.get_frame(frame_num=fk)
        frame_properties    = frame['frame_properties']

        # vcd.get_object_data(uid=, data_name=, frame_num=fk)    

if __name__ == "__main__":
    OPENLABEL_PATH = "./tmp/my_scene/nuscenes_sequence/annotated_openlabel.json"
    types = [
        "vehicle.car"
    ]

    main(openlabel_path=OPENLABEL_PATH, types=types)