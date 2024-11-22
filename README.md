# BEV2Seg_2

Estructura del repositorio:
```
beb2seg_2/
├── datasets/
│   ├── __init__.py
│   ├── NuImages/
│   │   ├── __init__.py
│   │   └── nuimages_dataset.py
│   ├── CARLA/
│   │   ├── __init__.py
│   │   └── carla_dataset.py
├── src/
│   ├── main.py
└── setup.py  # (opcional, si planeas distribuir el paquete)

```

`datasets` es un paquete que incluye varias facilidades para trabajar con datasets a la hora de entrenar los modelos tanto con imágenes _raw_ como reproyectadas en _BEV_. 


## Datasets for Image Segmentation
- [ApolloScape](https://apolloscape.auto/index.html)
- [CityScapes](https://github.com/mcordts/cityscapesScripts)
- [WildDash](https://www.wilddash.cc/)
- [KITTI](https://www.cvlibs.net/datasets/kitti/index.php)
- [NuScenes]()
- [NuImages](https://github.com/nutonomy/nuscenes-devkit)

## Interesting Image Segmentation Models
- [MSeg](https://github.com/mseg-dataset/mseg-semantic)
- [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2): YOLO Panoptic Segmentation
- [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)
- [DDRNet](https://github.com/ydhongHIT/DDRNet)
- [HANet](https://github.com/shachoi/HANet): No es un modelo si no un pluging que se _monta_ sobre otras arquitecturas como _DeepLabv3_ o _ResNet50_.
- [PIDNet](https://github.com/XuJiacong/PIDNet)
- [RegSeg](https://github.com/RolandGao/RegSeg)
Y unos cuantos más... En [este](https://github.com/zh320/realtime-semantic-segmentation-pytorch) hay unos cuantos modelos para _realtime semantic segmentation_ 