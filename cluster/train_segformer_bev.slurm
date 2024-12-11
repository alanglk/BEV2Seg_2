#!/bin/bash

#SBATCH -J agarciaj_SegformerTraining_onBEVDataset
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agarciaj@vicomtech.org
#SBATCH --time=14-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

srun --pty singularity run --nv $(cat <<EOF
    --bind /gpfs/VICOMTECH/Databases/GeneralDatabases/BEVDataset/:/dataset \
    --bind /home/agarciaj/bev2seg_2/models:/models \
    --bind /home/agarciaj/bev2seg_2/config:/config \
    ./bev2seg_2_v0.2.sif \
    /scripts/train_segformer.py /config/segformer_bev.toml
EOF
)