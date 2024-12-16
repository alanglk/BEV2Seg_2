#!/bin/bash

#SBATCH -J agarciaj_nuimages2nuimagesformatted
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agarciaj@vicomtech.org
#SBATCH --time=01-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

srun --pty singularity run --nv --bind /gpfs/VICOMTECH/Databases/GeneralDatabases/nuImages/:/nuimages --bind /gpfs/VICOMTECH/Databases/GeneralDatabases/NuImagesFormatted/:/NuImagesFormatted ./bev2seg_2_v0.3.sif $(cat <<EOF
    python3 /scripts/generate_NuImagesFormatted_from_NuImages.py \
    /nuimages \
    /NuImagesFormatted \
    --version "all" \
    --cam_name "CAM_FRONT"
EOF
)
