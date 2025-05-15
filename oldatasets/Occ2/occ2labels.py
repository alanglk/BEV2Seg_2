class Occ2Label:
    # 'name',         name of the category. 'background', 'occuped'...
    #                 
    # 'id',           identifier given by default. 
    #                   It goes from 0 to the number of categories in the dataset
    #
    # 'color'         color for each category
    def __init__(self, name:str, id: int, color:tuple):
        vars(self).update(locals())

occ2labels = [
    Occ2Label('background'  , 0 , (0,       0,      0   )),
    Occ2Label('occuped'     , 1 , (0,       74,     179 )),
    Occ2Label('occluded'    , 2 , (101,     164,    255 )),
    Occ2Label('driveable'   , 3 , (255,     192,    101 )),
]

occ2name2id     =  { label.name : label.id for label in occ2labels }
occ2id2name     =  { label.id : label.name for label in occ2labels }
occ2id2color    =  { label.id : label.color for label in occ2labels }