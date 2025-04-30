from typing import Tuple

class NuLabel:
    # 'token',        category_token from NuImages Dataset (loaded at parsing)
    #                   Is the identifier of the annotation type
    #                 
    # 'name',         name of the category. 'animal', 'human.pedestrian.adult'...
    #                   https://www.nuscenes.org/nuimages?externalData=all&mapData=all&modalities=Any
    #                 
    # 'id',           identifier given by default. 
    #                   It goes from 0 to the number of categories in the dataset
    #                 
    # 'trainId',      label given to the train masks. 
    #                   By default is the same as id
    #                 
    # 'dynamic',      The labeled object is dynamic
    # 'color'         color for each category
    def __init__(self, token: str, name:str, id: int, trainId: int, dynamic: bool, color:tuple):
        vars(self).update(locals())

nulabels = [
    #       token   name                                    id      trainId dynamic     color RGB
    NuLabel(  None  , "background"                          , 0     , 0       , False   , (0, 0, 0) ),
    NuLabel(  None  , "animal"                              , 1     , 1       , True    , (255, 0, 0) ),
    NuLabel(  None  , "human.pedestrian.adult"              , 2     , 2       , True    , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.child"              , 3     , 3       , True    , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.construction_worker", 4     , 4       , True    , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.personal_mobility"  , 5     , 5       , True    , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.police_officer"     , 6     , 6       , True    , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.stroller"           , 7     , 7       , True    , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.wheelchair"         , 8     , 8       , True    , (220,  20,  60) ),
    NuLabel(  None  , "movable_object.barrier"              , 9     , 9       , False   , (190, 153, 153) ),
    NuLabel(  None  , "movable_object.debris"               , 10    , 10      , False   , (152, 251, 152) ),
    NuLabel(  None  , "movable_object.pushable_pullable"    , 11    , 11      , False   , (255, 0, 0) ),
    NuLabel(  None  , "movable_object.trafficcone"          , 12    , 12      , True    , (111,  74,    0) ),
    NuLabel(  None  , "static_object.bicycle_rack"          , 13    , 13      , False   , (255, 0, 0) ),
    NuLabel(  None  , "vehicle.bicycle"                     , 14    , 14      , True    , (119,  11,  32)  ),
    NuLabel(  None  , "vehicle.bus.bendy"                   , 15    , 15      , True    , (  0,  60, 100) ),
    NuLabel(  None  , "vehicle.bus.rigid"                   , 16    , 16      , True    , (  0,  60, 100) ),
    NuLabel(  None  , "vehicle.car"                         , 17    , 17      , True    , (  0,   0, 142) ),
    NuLabel(  None  , "vehicle.construction"                , 18    , 18      , True    , (255, 0, 0) ),
    NuLabel(  None  , "vehicle.emergency.ambulance"         , 19    , 19      , True    , (255, 0, 0) ),
    NuLabel(  None  , "vehicle.emergency.police"            , 20    , 20      , True    , (255, 0, 0) ),      # antes 21
    NuLabel(  None  , "vehicle.motorcycle"                  , 21    , 21      , True    , (  0,   0, 230) ),  # antes 22
    NuLabel(  None  , "vehicle.trailer"                     , 22    , 22      , True    , (  0,   0, 110) ),  # antes 23
    NuLabel(  None  , "vehicle.truck"                       , 23    , 23      , True    , (  0,   0,  70) ),  # antes 24
    NuLabel(  None  , "vehicle.ego"                         , 24    , 24      , True    , (255, 255, 255) ),  # antes 25
    NuLabel(  None  , "flat.driveable_surface"              , 25    , 25      , False   , (128,  64, 128) ),  # antes 26
]


# name to label object
# trainId to name and to color
nuname2label      = { label.name    : label for label in nulabels }
nuid2name =  { label.trainId : label.name for label in nulabels }
nuid2color =  { label.trainId : label.color for label in nulabels }
nuid2dynamic = { label.trainId : label.dynamic for label in nulabels }

# Merge labels
DEFAULT_MERGE_DICT = {
    "background": [ "background" ],
    "animal": [ "animal" ],
    "human.pedestrian": [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair"
    ],
    "movable_object.barrier" : [
        "movable_object.barrier",
        "movable_object.debris",
        "movable_object.pushable_pullable",
        "movable_object.trafficcone",
        "static_object.bicycle_rack",
    ],
    "vehicle.car": [
        "vehicle.bus.bendy", 
        "vehicle.bus.rigid", 
        "vehicle.car", 
        "vehicle.construction", 
        "vehicle.emergency.ambulance", 
        "vehicle.emergency.police", 
        "vehicle.trailer", 
        "vehicle.truck"
    ],
    "vehicle.motorcycle":[
        "vehicle.bicycle",
        "vehicle.motorcycle"
    ],
    "flat.driveable_surface":[ "flat.driveable_surface" ]
}


def get_merged_nulabels(nuid2name:dict, nuname2label:dict, nuid2color:dict, nuid2dynamic:dict, merge_dict:dict = DEFAULT_MERGE_DICT) -> Tuple[dict]:
    """
    RETURN: ( new_nuid2name,  new_nuname2label, new_nuid2color, new_nuid2dynamic, merging_lut_ids, merging_lut_names )
    """
    
    # Assertion
    for l_id, l_n in nuid2name.items():
        considered_label = False
        for merge_l, labels in merge_dict.items():
            if l_n in labels:
                considered_label = True
                break
        assert considered_label, f"label {l_n} not pressent in merge_dict"


    # Merging LUTs
    merging_lut_ids     = {}
    merging_lut_names   = {}
    for res_label, labels in merge_dict.items():
        res_label_id = nuname2label[res_label] 
        for l in labels:
            merging_lut_names[l] = res_label
            l_id = nuname2label[l]
            merging_lut_ids[l_id] = res_label_id

    # New label dicts
    new_nuid2name = { i:k for i, k in enumerate(merge_dict.keys()) }
    new_nuname2label = { v:k for k, v in new_nuid2name.items() }

    new_nuid2color      = { i:nuid2color[ nuname2label[meged_l] ] for i, meged_l in new_nuid2name.items()}
    new_nuid2dynamic    = { i:nuid2dynamic[ nuname2label[meged_l] ] for i, meged_l in new_nuid2name.items()}

    return ( new_nuid2name,  new_nuname2label, new_nuid2color, new_nuid2dynamic, merging_lut_ids, merging_lut_names )
    

     
