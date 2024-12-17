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

## Setup
Setup del virtual enviroment:
```bash
cd beb2seg_2
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./datasets
```

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



## Scripts
En la carpeta `scripts` hay varias herramientas para automatizar la generación de datasets... Estas herramientas también están embebidas en una imagen de Docker para poder hacer el port a Singularity y ejecutarlas en el HPC.

### BEVDataset from NuImages
Generar un _BEVDataset_ a partir de _NuImages_:
```bash
python3 srcipts/generate_BEVDataset_from_NuImages.py <nuimages_path> <output_path> --version <version> --cam_name "CAM_FRONT"
# Versión HPC (ver sección HPC)
# sbatch cluster/generate_BEVDataset_from_NuImages.slurm
```

donde version puede tomar los valores `['mini', 'train', 'val', 'test', 'all']` (por defecto `'mini'`) y `cam_name` es opcional. La estructura resultante del _BEVDataset_ sería:
```
.../BEVDataset/
    mini/
        - token1.json
        - token1_bev.png
        - token1_raw.png
        - token1_color.png
        - token1_semantic.png
        ...
    train/
    test/
```

### NuImagesFormatted from NuImages
Transformar _NuImages_ al formato:
```
.../NuImagesFormatted/
    mini/
        - token1.json
        - token1_bev.png
        - token1_raw.png
        - token1_color.png
        - token1_semantic.png
        ...
    train/
    test/
```
```bash
python3 srcipts/generate_NuImagesFormatted_from_NuImages.py <nuimages_path> <output_path> --version <version> --cam_name "CAM_FRONT"
# Versión HPC (ver sección HPC)
# sbatch cluster/generate_NuImagesFormatted_from_NuImages.slurm
```


### Train SegFormer
```bash
docker run -it \
    --gpus '"device=0"' \
    -v $(pwd)/tmp/BEVDataset:/BEVDataset \
    -v $(pwd)/tmp/models:/models \
    -v $(pwd)/config:/config \
    agarciaj/bev2seg_2:v0.2 \
    /scripts/train_segformer.py /config/segformer.toml
```

Para monitorizar el entrenamiento con tensorboard se puede utilizar el siguiente comando:

```bash
#sshfs agarciaj@zegama002:/home/agarciaj/bev2seg_2/tmp/models /home/VICOMTECH/agarciaj/GitLab/bev2seg_2/tmp/models
#sshfs agarciaj@zegama002:/gpfs/VICOMTECH/Databases/GeneralDatabases/nuImages /workspaces/bev2seg_2/tmp/NuImages
#docker run -v ./tmp/models/<model_name>/runs:/runs -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /runs
docker plugin install vieux/sshfs
cd docker && docker compose up
```

### Docker image for scripts
Generar una imagen de Docker con los scripts y librerías necesarias:
```bash
bash ./docker/build-dockerfile.sh <docker-image-version>
```

Adicionalmente se creará un archivo `bev2seg_2_<version>.tar` que podrá ser utilizado para generar una imagen de Singularity con el comando:
```bash
scp docker/bev2seg_2_<version>.tar agarciaj@zegama002:/home/agarciaj/BEV2SEG_2
singularity build bev2seg_2_<version>.sif docker-archive://bev2seg_2_<version>.tar # Desde Zegama
```

Para generar un BEVDataset a partir de NuImages con Docker se puede utilizar:
```bash
docker run -it \
    -it \
    -v $(pwd)/data/BEVDataset:/data/output \
    -v $(pwd)/data/input:/data/input:ro \
    agarciaj/beg2seg_2:v0.1 \
    /scripts/generate_BEVDataset_from_NuImages.py /data/input /data/output --version "mini" --cam_name "CAM_FRONT"
```

Si se quiere generar el BEVDataset en el HPC hay que utilizar la imagen de Singularity generada y ejecutarla ya sea en un job o en una interfaz interactiva:

´´´bash
bash docker/train_segformer.sh -d ./tmp/BEVDataset/ -m ./models/ -c ./config/test.toml -g 2,3
´´´

### High Performance Computing (HPC)
```bash
squeue # Ver la cola de Jobs de Slurm
salloc --x11 -n 1 -c 1 --gres=gpu:t4:1 -t 01:00:00 # Iniciar una sesión interactiva
sbatch test.slurm # Añadir un Job a la cola
seff <job_id> # 84510 Mostrar la eficiencia de un job terminado
```

Lanzar una imagen de Singularity en el HPC (codigo dentro del job test.slurm):

```bash
srun --pty singularity run --nv .bev2seg_2_v0.1.sif # --nv para acceso a GPU
srun --pty singularity run --nv --bind /gpfs/VICOMTECH/.../data:/data .bev2seg_2_v0.1.sif
```

```bash
srun --pty singularity run --nv --bind /gpfs/VICOMTECH/Databases/GeneralDatabases/nuImages/:/nuimages --bind /gpfs/VICOMTECH/home/agarciaj/BEV2SEG_2/BEVDataset/:/BEVDataset ./bev2seg_2_v0.1.sif $(cat <<EOF
    python3 /scripts/generate_BEVDataset_from_NuImages.py \
    /nuimages \
    /BEVDataset \
    --version "mini" \
    --cam_name "CAM_FRONT"
EOF
)
```