#!/opt/conda/bin/python3
"""
Script para parsear NuImages al formato:
.../BEVDataset/
    mini/
        - token1_raw.png
        - token1_color.png
        - token1_semantic.png
        ...
    train/
    test/

Ejemplo de uso:
python3 srcipts/generate_NuImagesFormatted_from_NuImages.py <nuimages_path> <output_path> --version <version> --cam_name "CAM_FRONT"
"""
import os
import argparse
from oldatasets.NuImages import NuImagesDataset, generate_NuImagesFormatted_from_NuImages

def prepare_output_folder(out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        # Clear the tmp folder
        files = os.listdir(out_path)
        for f in files:
            os.remove(os.path.join(out_path, f))

def checker(args):
    # Ensure the NuImages path exists
    if not os.path.exists(args.nuimages_path):
        raise Exception(f"NuImages root path not found: {args.nuimages_path}")
    
    # Check if the version is supported
    if args.version != 'all' and args.version not in NuImagesDataset.DATASET_VERSIONS:
        raise Exception(f"Not supported version: {args.version}")

def main(nuimages_path, output_path, version, cam_name):
    # Generate the NuImagesFormatted
    print("Generating NuImagesFormatted from NuImages...")

    if version == 'all':
        for v in NuImagesDataset.DATASET_VERSIONS:
            out_path = os.path.join(output_path, v)
            prepare_output_folder(out_path)
            print(f"####     Generating {v} version of NuImages in {out_path}")

            num_generated, err_indices = generate_NuImagesFormatted_from_NuImages(
                dataset_path=nuimages_path, 
                out_path=out_path, 
                version=v, 
                cam_name= cam_name)
            
            print(f"Indices of images not generated from NuImages Dataset: {err_indices}")
            print(f"NuImagesFormatted successfully generated with {num_generated} instances :D")
            print()
    else:
        out_path = os.path.join(output_path, version)
        prepare_output_folder(out_path)

        num_generated, err_indices = generate_NuImagesFormatted_from_NuImages(
            dataset_path=nuimages_path, 
            out_path=out_path, 
            version=version, 
            cam_name= cam_name)
            
        print(f"Indices of images not generated from NuImages Dataset: {err_indices}")
        print(f"NuImagesFormatted successfully generated with {num_generated} instances :D")
        print()
    
    print("Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para procesar datos con diferentes versiones.")
    parser.add_argument('nuimages_path', type=str, help="Ruta al directorio de nuImages.")
    parser.add_argument('output_path', type=str, help="Ruta de salida donde se guardará el resultado.")
    
    # Optional args
    parser.add_argument('--version', choices=['mini', 'train', 'val', 'test', 'all'], default='mini', help="Versión opcional del dataset (mini, train, val, test). Por defecto es 'mini'. Si se selecciona 'all' se transformarán todos")
    parser.add_argument('--cam_name', default= None, help="Opcional. Selección de la cámara para generar el NuImagesFormatted. Si es None, se generan todas las camaras de NuImages.")
    args = parser.parse_args()
    checker(args)
    main(args.nuimages_path, args.output_path, args.version, args.cam_name)
