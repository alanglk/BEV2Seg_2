#!/bin/bash

# Función de ayuda
function print_help {
    echo "Uso: $0 -d DATAROOT_PATH -m MODELS_PATH -c CONFIG_PATH [-g CUDA_VISIBLE_DEVICES] [-ns]"
    echo
    echo "  -d DATAROOT_PATH     Ruta al directorio raíz de los datos."
    echo "  -m MODELS_PATH       Ruta al directorio de los modelos."
    echo "  -c CONFIG_PATH       Ruta completa al archivo de configuración."
    echo "  -g CUDA_VISIBLE_DEVICES   (Opcional) GPUs específicas a usar, por defecto selecciona todas las GPUs disponibles."
    echo "  -ns                  (Opcional) No usar screen, ejecutar docker run directamente."
    echo
    echo "Ejemplo de uso:"
    echo "  $0 -d ./tmp/BEVDataset/ -m ./models/ -c ./config/segformer_bev.toml"
    echo "  $0 -d ./tmp/BEVDataset/ -m ./models/ -c ./config/segformer_bev.toml -g 2,3"
    echo "  $0 -d ./tmp/BEVDataset/ -m ./models/ -c ./config/segformer_bev.toml -g 2,3 -ns"
    exit 1
}

# Comprobar si hay al menos 4 parámetros
if [ $# -lt 8 ]; then
    print_help
fi

# Parseo de argumentos
USE_SCREEN=true
while getopts "d:m:c:g:" opt; do
    case ${opt} in
        d)
            DATAROOT_PATH=$(realpath $OPTARG)
            ;;
        m)
            MODELS_PATH=$(realpath $OPTARG)
            ;;
        c)
            CONFIG_PATH=$(realpath $OPTARG)
            ;;
        g)
            CUDA_VISIBLE_DEVICES=$OPTARG
            ;;
        n)
            USE_SCREEN=false
            ;;
        s)
            # Opción para no usar screen, ya se ha capturado arriba
            ;;
        *)
            print_help
            ;;
    esac
done

# Si no se ha proporcionado CUDA_VISIBLE_DEVICES, seleccionar todas las GPUs disponibles
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
fi

if [ ! -e "$CONFIG_PATH" ]; then
    echo "Error: El archivo de configuración '$CONFIG_PATH' no existe."
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: '$CONFIG_PATH' no es un archivo válido."
    exit 1
fi

CONFIG_DIR=$(dirname $CONFIG_PATH)
CONFIG_FILE=$(basename $CONFIG_PATH)

# Construcción del comando docker
ARGS=$(cat <<EOF
    -it \
    --ipc=host \
    --gpus all \
    -v ${DATAROOT_PATH}:/dataset:ro \
    -v ${CONFIG_DIR}:/config:ro \
    -v ${MODELS_PATH}:/models \
    agarciaj/bev2seg_2:v0.3 \
    bash -c "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /scripts/train_segformer.py /config/${CONFIG_FILE}"
EOF
)

# Command for mounting the dataset inside the container
# screen -dmS "train_segformer_nu.toml" bash -c "docker run $(cat <<EOF
#     -it \
#     --user root \
#     --gpus all \
#     --privileged \
#     -v ./config:/config:ro \
#     -v ./models:/models \
#     agarciaj/bev2seg_2:v0.3 \
#     bash -c "mkdir /dataset && sshfs agarciaj@zegama002:/gpfs/VICOMTECH/Databases/GeneralDatabases/nuImages /dataset && CUDA_VISIBLE_DEVICES=2,3 /scripts/train_segformer.py /config/segformer_nu.toml"
# EOF
# ) | tee -a models/train_segformer_nu.toml_$(date +%s).log"


# Imprimir el comando y ejecutarlo
echo docker run $ARGS
echo

if [ "$USE_SCREEN" = true ]; then
    # Lanzar el comando en screen con un nombre único
    SESSION_NAME="train_segformer.py_${CONFIG_FILE}"
    LOG_FILE="${MODELS_PATH}/${SESSION_NAME}_$(date +%s).log"
    screen -dmS "$SESSION_NAME" bash -c "docker run $ARGS | tee -a $LOG_FILE"
    echo Docker training launched in screen session: $SESSION_NAME
    screen -ls
else
    # Ejecutar el comando sin screen
    docker run $ARGS
fi
