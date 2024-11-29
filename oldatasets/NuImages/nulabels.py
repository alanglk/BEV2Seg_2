
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
    # 'color'         color for each category
    def __init__(self, token: str, name:str, id: int, trainId: int, color:tuple):
        vars(self).update(locals())

nulabels = [
    #       token   name                                    id      trainId    color RGB
    NuLabel(  None  , "animal"                              , 1     , 1       , (255, 0, 0) ),
    NuLabel(  None  , "human.pedestrian.adult"              , 2     , 2       , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.child"              , 3     , 3       , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.construction_worker", 4     , 4       , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.personal_mobility"  , 5     , 5       , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.police_officer"     , 6     , 6       , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.stroller"           , 7     , 7       , (220,  20,  60) ),
    NuLabel(  None  , "human.pedestrian.wheelchair"         , 8     , 8       , (220,  20,  60) ),
    NuLabel(  None  , "movable_object.barrier"              , 9     , 9       , (190, 153, 153) ),
    NuLabel(  None  , "movable_object.debris"               , 10    , 10      , (152, 251, 152) ),
    NuLabel(  None  , "movable_object.pushable_pullable"    , 11    , 11      , (255, 0, 0) ),
    NuLabel(  None  , "movable_object.trafficcone"          , 12    , 12      , (111,  74,    0) ),
    NuLabel(  None  , "static_object.bicycle_rack"          , 13    , 13      , (255, 0, 0) ),
    NuLabel(  None  , "vehicle.bicycle"                     , 14    , 14      , (119,  11,  32)  ),
    NuLabel(  None  , "vehicle.bus.bendy"                   , 15    , 15      , (  0,  60, 100) ),
    NuLabel(  None  , "vehicle.bus.rigid"                   , 16    , 16      , (  0,  60, 100) ),
    NuLabel(  None  , "vehicle.car"                         , 17    , 17      , (  0,   0, 142) ),
    NuLabel(  None  , "vehicle.construction"                , 18    , 18      , (255, 0, 0) ),
    NuLabel(  None  , "vehicle.emergency.ambulance"         , 19    , 19      , (255, 0, 0) ),
    NuLabel(  None  , "vehicle.emergency.police"            , 21    , 21      , (255, 0, 0) ),
    NuLabel(  None  , "vehicle.motorcycle"                  , 22    , 22      , (  0,   0, 230) ),
    NuLabel(  None  , "vehicle.trailer"                     , 23    , 23      , (  0,   0, 110) ),
    NuLabel(  None  , "vehicle.truck"                       , 24    , 24      , (  0,   0,  70) ),
    NuLabel(  None  , "vehicle.ego"                         , 25    , 25      , (255, 255, 255) ),
    NuLabel(  None  , "flat.driveable_surface"              , 26    , 26      , (128,  64, 128) ),
]


# name to label object
nuname2label      = { label.name    : label for label in nulabels }

# trainId to name and to color
nuid2name =  { label.trainId : label.name for label in nulabels }
nuid2color =  { label.trainId : label.color for label in nulabels }

