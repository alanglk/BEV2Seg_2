import warnings
import sys
import os

warnings.filterwarnings('ignore')
os.chdir("/home/VICOMTECH/agarciaj/GitLab/bev2seg_2")
sys.path.append('/home/VICOMTECH/agarciaj/GitLab/bev2seg_2')

from src.bev2seg_2 import Raw2Seg_BEV, Raw_BEV2Seg

model_name = "raw2segbev_mit-b0_v0.4"
model_path = os.path.join("./models", "segformer_nu_formatted", model_name)
model = Raw2Seg_BEV(model_path, None)

print(model)



