#!/env/bin/python3
"""
Script para generar un BEVDataset a partir del NuImages Dataset.
La estructura resultante será:
.../BEVDataset/
    - token1.json
    - token1_raw.png
    - token1_color.png
    - token1_semantic.png
    ...

Ejemplo de uso:
python3 srcipts/generate_BEVDataset_from_NuImages.py <nuimages_path> <output_path> --version <version> --cam_name "CAM_FRONT"
"""
import os
import argparse
from datasets.NuImages import NuImagesDataset, generate_BEVDataset_from_NuImages

def checker(args):
    # Ensure the NuImages path exists
    if not os.path.exists(args.nuimages_path):
        raise Exception(f"NuImages root path not found: {args.nuimages_path}")

    # Prepare the output folder
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        # Clear the tmp folder
        files = os.listdir(args.output_path)
        for f in files:
            os.remove(os.path.join(args.output_path, f))

    # Check if the version is supported
    if args.version not in NuImagesDataset.DATASET_VERSIONS:
        raise Exception(f"Not supported version: {args.version}")

def main(nuimages_path, output_path, version, cam_name):
    # Generate the BEVDataset
    print("Generating BEVDataset from NuImages...")
    num_generated, err_indices = generate_BEVDataset_from_NuImages(dataset_path=nuimages_path, out_path=output_path, version=version, cam_name= cam_name)
    
    print(f"Indices of images not generated from NuImages Dataset: {err_indices}")
    print(f"BEVDataset successfully generated with {num_generated} instances :D")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para procesar datos con diferentes versiones.")
    parser.add_argument('nuimages_path', type=str, help="Ruta al directorio de nuImages.")
    parser.add_argument('output_path', type=str, help="Ruta de salida donde se guardará el resultado.")
    
    # Optional args
    parser.add_argument('--version', choices=['mini', 'train', 'val', 'test'], default='mini', help="Versión opcional del dataset (mini, train, val, test). Por defecto es 'mini'.")
    parser.add_argument('--cam_name', default= None, help="Opcional. Selección de la cámara para generar el BEVDataset. Si es None, se generan todas las camaras de NuImages.")
    args = parser.parse_args()
    checker(args)
    main(args.nuimages_path, args.output_path, args.version, args.cam_name)
