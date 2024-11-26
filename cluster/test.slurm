#!/bin/bash

#SBATCH -J agarciaj_bev
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agarciaj@vicomtech.org
#SBATCH --time=14-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

srun singularity exec --nv --bind /gpfs/VICOMTECH/Databases/GeneralDatabases/nuimages:/nuimages ./a


