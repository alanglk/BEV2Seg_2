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