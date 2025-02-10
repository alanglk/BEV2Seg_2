#!/opt/conda/bin/python3
"""
Script para hacer un split igual de varios datasets generados.
Formatos de dataset soportados: ["BEVDataset", "NuImagesFormattedDataset"]

Input:
./Dataset1/
    mini/
        - sample_1
        - sample_2
        - sample_3
        - sample_4
    dest/

Output:
./Dataset1/
    mini/
        - sample_1
        - sample_2
    dest/
        - sample_3
        - sample_4

Formatos de datasets soportados:

Ejemplo de uso:
python3 srcipts/generate_BEVDataset_from_NuImages.py <bevdataset_path> <nuimages_path> <split_name> --size 0.5 --version <version>
"""

import os
import argparse
from oldatasets.postprocessing import DatasetSplitter 

def prepare_output_folder(out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    elif len(os.listdir(out_path)) > 0:
        raise Exception(f"The output path: {out_path} is not empty")

def checker(args):
    # Ensure the Dataset paths exists
     if not os.path.exists(args.bevdataset_path):
        raise Exception(f"BEVDataset root path not found: {args.bevdataset_path}")     
     if not os.path.exists(args.nuimages_path):
        raise Exception(f"NuImagesFormatted root path not found: {args.nuimages_path}")



def main(bevdataset_path, nuimagesformatted_path, split_name, size, version):
    # Generate out paths
    bev_split_out   = os.path.join(bevdataset_path, split_name)
    nu_split_out    = os.path.join(nuimagesformatted_path, split_name)
    prepare_output_folder(bev_split_out)
    prepare_output_folder(nu_split_out)


    # Generate input dict for splitting
    dataset_paths = [
        {"dataroot": bevdataset_path, "type": "BEVDataset", "split_dest_path": bev_split_out},
        {"dataroot": nuimagesformatted_path, "type": "NuImagesFormattedDataset", "split_dest_path": nu_split_out}
    ]
    print(f"Splitting {version} version of {[i['type'] for i in dataset_paths]} into a set of {size*100}% of the original set")
    
    dataset_splitter = DatasetSplitter(dataset_paths, version=version, set2_size=size)
    dataset_splitter.move_split_to_path()
    print("Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para procesar datos con diferentes versiones.")
    parser.add_argument('bevdataset_path', type=str, help="Ruta al BEVDataset")
    parser.add_argument('nuimages_path', type=str, help="Ruta al NuImagesFormattedDataset")
    parser.add_argument('split_name', type=str, help="Name of the resulting split")
    
    # Optional args
    parser.add_argument('--size', type=float, default=0.5, help="Size of the resulting split")
    parser.add_argument('--version', choices=['mini', 'train', 'val', 'test'], default='mini', help="Versión opcional del dataset (mini, train, val, test). Por defecto es 'mini'")
    
    args = parser.parse_args()
    checker(args)
    main(args.bevdataset_path, args.nuimages_path, args.split_name, args.size, args.version)
