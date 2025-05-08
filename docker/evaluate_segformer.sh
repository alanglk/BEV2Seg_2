#!/bin/bash

# Función de ayuda
function print_help {
    echo "Uso: $0 -o OUTPUT_PATH -l EVALUATION_LIST [-g CUDA_VISIBLE_DEVICES] [-ns]"
    echo
    echo "  -o OUTPUT_PATH       Ruta al archivo de salida."
    echo "  -m MODELS_PATH       Ruta a la carpeta con todos los modelos."
    echo "  -a NU_DATASET        Ruta al NuDataset"
    echo "  -b BEV_DATASET       Ruta al BEVDataset"
    echo "  -l EVALUATION_LIST   Lista de pares (model_path, dataset_path) a evaluar."
    echo "  -g CUDA_VISIBLE_DEVICES   (Opcional) GPUs específicas a usar, por defecto selecciona todas las GPUs disponibles."
    echo "  -ns                  (Opcional) No usar screen, ejecutar docker run directamente."
    echo
    echo "Ejemplo de uso:"
    echo "  $0 -o ./data/model_evaluations.pkl -m ./models -a ./tmp/NuImagesFormatted -b ./tmp/BEVDataset -l \"[('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.4'), ('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.4'), ('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.5'), ('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.5'), ('./models/segformer_bev/raw2bevseg_mit-b0_v0.5'),('./models/segformer_bev/raw2bevseg_mit-b0_v0.6')]\""
    exit 1
}


# Comprobar si hay al menos 4 parámetros
if [ $# -lt 4 ]; then
    print_help
fi

# Parseo de argumentos
USE_SCREEN=true
while getopts "o:m:a:b:l:g:ns" opt; do
    case ${opt} in
        o)
            OUTPUT_PATH=$OPTARG
            ;;
        m)
            MODELS_PATH=$OPTARG
            ;;
        a)
            NU_DATASET=$OPTARG
            ;;
        b)
            BEV_DATASET=$OPTARG
            ;;
        l)
             EVALUATION_LIST=$OPTARG
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

# EDITA ESTO ===========================================
OUTPUT_DIR=$(dirname $OUTPUT_PATH)
OUTPUT_FILE=$(basename $OUTPUT_PATH)

# Construir comando docker
escape_for_docker_bash_c() {
    local input="$1"
    # Paso 1: reemplazar comillas simples por comillas dobles
    local double_quotes="${input//\'/\"}"
    # Paso 2: escapado doble
    local escaped="${double_quotes//\"/\\\\\"}"
    echo "$escaped"
}
ESCAPED_EVAL=$(escape_for_docker_bash_c "$EVALUATION_LIST")

ARGS=$(cat <<EOF
    -it \
    --ipc=host \
    --gpus all \
    -v ${MODELS_PATH}:/models:ro \
    -v ${NU_DATASET}:/NuImagesFormatted:ro \
    -v ${BEV_DATASET}:/BEVDataset:ro \
    -v ${OUTPUT_DIR}:/output \
    agarciaj/bev2seg_2:v0.5 \
    bash -c "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 /scripts/evaluate_bev2seg_2.py --output_path /output/${OUTPUT_FILE} --nu_dataset_path /NuImagesFormatted --bev_dataset_path /BEVDataset --models_to_evaluate \"$EVALUATION_LIST\""
EOF
)

# ======================================================

# Imprimir el comando y ejecutarlo
echo docker run $ARGS
echo

if [ "$USE_SCREEN" = true ]; then
    # Lanzar el comando en screen con un nombre único
    SESSION_NAME="evaluate_bev2seg2.py${CONFIG_FILE}"
    LOG_FILE="${SESSION_NAME}_$(date +%s).log"
    screen -dmS "$SESSION_NAME" bash -c "docker run $ARGS | tee -a $LOG_FILE"
    echo Docker training launched in screen session: $SESSION_NAME
    screen -ls
else
    # Ejecutar el comando sin screen
    docker run $ARGS
fi
