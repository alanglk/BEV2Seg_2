#!/bin/bash

# Función de ayuda
function print_help {
    echo "Uso: $0 -d DATAROOT_PATH -m MODELS_PATH -c OUTPUT_PATH [-g CUDA_VISIBLE_DEVICES] [-ns]"
    echo
    echo "  -l EVALUATION_LIST   Lista de pares (model_path, dataset_path) a evaluar."
    echo "  -o OUTPUT_PATH       Ruta al archivo de salida."
    echo "  -g CUDA_VISIBLE_DEVICES   (Opcional) GPUs específicas a usar, por defecto selecciona todas las GPUs disponibles."
    echo "  -ns                  (Opcional) No usar screen, ejecutar docker run directamente."
    echo
    echo "Ejemplo de uso:"
    echo "  $0 -l [('./models/segformer_bev/model_1', './tmp/BEVDataset'), ('./models/segformer_nu/model_1', './tmp/BEVDataset'), ('./models/segformer_nu/model_1', './tmp/NuImagesFormatted'), ('./models/segformer_nu/model_2', './tmp/NuImagesFormatted')] -o ./data/model_evaluations.pkl"
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
        l)
            EVALUATION_LIST=$(realpath $OPTARG)
            ;;
        o)
            OUTPUT_PATH=$(realpath $OPTARG)
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

# EDITA ESTO ===========================================
OUTPUT_DIR=$(dirname $OUTPUT_PATH)
OUTPUT_FILE=$(basename $OUTPUT_PATH)

# Listas de paths únicos
readarray -t MODEL_PATHS < <(python3 -c "
import ast
models = ast.literal_eval(\"\"\"$EVALUATION_LIST\"\"\")
model_set = set(m[0] for m in models)
for path in sorted(model_set): print(path)
")

readarray -t DATASET_PATHS < <(python3 -c "
import ast
models = ast.literal_eval(\"\"\"$EVALUATION_LIST\"\"\")
dataset_set = set(m[1] for m in models)
for path in sorted(dataset_set): print(path)
")

DOCKER_MOUNTS=""

# Montar modelos (misma ruta dentro y fuera)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_REALPATH=$(realpath "$MODEL_PATH")
    DOCKER_MOUNTS="$DOCKER_MOUNTS -v ${MODEL_REALPATH}:${MODEL_REALPATH}:ro"
done

# Montar datasets (misma ruta dentro y fuera)
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    DATASET_REALPATH=$(realpath "$DATASET_PATH")
    DOCKER_MOUNTS="$DOCKER_MOUNTS -v ${DATASET_REALPATH}:${DATASET_REALPATH}:ro"
done

# Montar carpeta de salida
DOCKER_MOUNTS="$DOCKER_MOUNTS -v ${OUTPUT_DIR}:/output"

# Construir comando docker
ARGS=$(cat <<EOF
    -it \
    --ipc=host \
    --gpus all \
    ${DOCKER_MOUNTS} \
    agarciaj/bev2seg_2:v0.5 \
    bash -c "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES /scripts/evaluate_bev2seg_2.py --models_to_evaluate '$EVALUATION_LIST' --output_path /output/${OUTPUT_FILE}"
EOF
)
# ======================================================



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
