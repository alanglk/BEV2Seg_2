from datasets.NuImages import NuImagesDataset, NuImagesBEVDataset, generate_NuImages_OpenLABEL

# path = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"
# generate_NuImages_OpenLABEL(path, "./data/NuImages/OpenLABEL")

path = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/nuImages"

dataset = NuImagesBEVDataset(dataroot=path, 
            openlabelroot='./data/NuImages/OpenLABEL', 
            save_openlabel=True)

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

import cv2
for i in range(1):
    image, target = dataset.__getitem__(i)
    image = cv2.resize(image, (854, 480))
    cv2.imshow("bev", image)
    cv2.waitKey(0)

# 
# from vcd import core, scl
# 
# #file_path = 'data/NuImages/OpenLABEL/setup1.json'
# file_path = 'data/NuImages/OpenLABEL/0128b121887b4d0d86b8b1a43ac001e9.json'
# 
# vcd = core.VCD()
# vcd.load_from_file(file_path)
# 
# scene = scl.Scene(vcd)
# 
# #cam = scene.get_camera("camera_front",0)
# cam = scene.get_camera("CAM_FRONT_RIGHT",0)
# 
# print(cam)


