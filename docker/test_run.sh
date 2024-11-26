

ARGS=$(cat <<EOF
    -it \
    -v $(pwd)/data/BEVDataset:/data/output \
    -v $(pwd)/data/input:/data/input:ro \
    beg2seg_2:v0.1 \
    generate_BEVDataset_from_NuImages.py /data/input /data/output --version "mini" --cam_name "CAM_FRONT"
EOF
)
ARGS=$(cat <<EOF
    -it \
    -v $(pwd)/data/BEVDataset:/data/output \
    -v $(pwd)/data/input:/data/input:ro \
    beg2seg_2:v0.1 \
    /bin/bash
EOF
)

echo $ARGS
echo
docker run $ARGS