#!/bin/bash

#SBATCH -J agarciaj_SegformerTraining_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agarciaj@vicomtech.org
#SBATCH --time=00-1:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

srun --pty singularity run --nv $(cat <<EOF
    --bind /gpfs/VICOMTECH/Databases/GeneralDatabases/BEVDataset/:/BEVDataset \
    --bind /home/agarciaj/bev2seg_2/tmp/models:/models \
    --bind /home/agarciaj/bev2seg_2/config:/config \
    ./bev2seg_2_v0.2.sif \
    /scripts/train_segformer.py /config/segformer_bev.toml
EOF
)
