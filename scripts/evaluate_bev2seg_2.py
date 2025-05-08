#!/opt/conda/bin/python3
"""
Script para evaluar BEV2Seg2.
Ejemplo de uso:

raw2bevseg (eval type 0):
python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/BEVDataset --model_path ./models/segformer_bev/raw2bevseg_mit-b0_v0.3 --output_path ./data/model_evaluations.pkl
python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/BEVDataset --model_path ./models/segformer_bev/raw2bevseg_mit-b0_v0.5 --output_path ./data/model_evaluations.pkl

raw2segbev (eval types 1 and 2):
python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/BEVDataset --model_path ./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.3 --output_path ./data/model_evaluations.pkl
python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/NuImagesFormatted --model_path ./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.3 --output_path ./data/model_evaluations.pkl

"""

from oldatasets.BEV import BEVDataset
from oldatasets.NuImages import NuImagesFormattedDataset
from oldatasets.NuImages.nulabels import nuid2name, get_merged_nulabels
from src.bev2seg_2 import Raw2Seg_BEV, Raw_BEV2Seg

from vcd import core
import pandas as pd
import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt

from typing import List
from tqdm import tqdm
import argparse
import pickle
import ast
import sys
import os
import shutil

import evaluate
metric = evaluate.load("mean_iou")

def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")
def check_args(args):
    check_paths([args.nu_dataset_path, args.bev_dataset_path, args.model_path, os.path.dirname(args.output_path)])

def get_evaluation_types(model_path:str) -> List[int]:
    """
    INPUT: model_path, dataset_path
    OUTPUT: Evaluation type
        - 0 for raw2bevseg on BEVDataset
        - 1 for raw2segbev on BEVDataset
        - 2 for raw2segbev on NuImagesFormattedDataset
    """
    model_name = os.path.basename(model_path)
    evaluation_types = []
    if model_name.find("raw2segbev") != -1:
        evaluation_types = [1, 2]
    elif model_name.find("raw2bevseg") != -1:
        evaluation_types = [0]
    if len(evaluation_types) == []:
        raise Exception(f"Wrong model_name: {model_name}")
    return evaluation_types

def check_or_reset_path(path):
    if os.path.exists(path):
        print(f"La ruta '{path}' ya existe.")
        answer = input("¿Quieres eliminarla y recrearla? [s/N]: ").strip().lower()
        if answer in ['s', 'si']:
            print(f"Eliminando y recreando '{path}'...")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            os.makedirs(path, exist_ok=True)
        else:
            print(f"Manteniendo '{path}'.")
    else:
        print(f"Creando '{path}'...")
        os.makedirs(path, exist_ok=True)

def get_merging_strategy(model_id2label:dict):
    merging_lut_ids = None
    if ( len(model_id2label) -1 ) < len(nuid2name):
        id2label,_,_,_, merging_lut_ids, _ = get_merged_nulabels()
        
        aux = {int(k):v for k,v in model_id2label.items()}
        if 255 in aux:
            aux.pop(255)
        assert id2label == aux
    return merging_lut_ids

def show_eval_report(data:dict, model_name:str, eval_type:int, label_list:List):
    assert model_name in data
    assert eval_type in data[model_name]
    num_labels = len(label_list)
    
    # Mean IoU
    mean_iou = 0.0
    for i in range(num_labels):
        mean_iou += data[model_name][eval_type]['mean_iou_per_class'][i]
    mean_iou /= num_labels
    
    # Confusion Matrix
    conf_matrix = pd.DataFrame(data[model_name][eval_type]['conf_matrix'], index=label_list, columns=label_list)
    
    a = data[model_name][eval_type]['mean_precision_per_class']
    b = data[model_name][eval_type]['mean_recall_per_class']
    c = data[model_name][eval_type]['mean_accuracy_per_class']
    d = data[model_name][eval_type]['mean_f1score_per_class']
    e = data[model_name][eval_type]['mean_iou_per_class']

    # Print data
    print(f"EVALUATION REPORT OF {model_name}\n")
    print("-----------------------------------------")
    print(f"Labels: \n{label_list}\n")
    print(f"Mean Precision per class: \n{a}\n")
    print(f"Mean Recall per class: \n{b}\n")
    print(f"Mean Accuracy per class: \n{c}\n")
    print(f"Mean F1-Score per class: \n{d}\n")
    print(f"Mean IoU per class: \n{e}\n")
    print(f"Mean IoU: \n{mean_iou}\n")
    print(f"Confussion Matrix: \n{conf_matrix}\n")
    print("-----------------------------------------")
    print()

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def calculate_metrics(annotations:np.ndarray, predictions:np.ndarray, num_classes:int):
    """
    Calcula las métricas de las predicciones y la matriz de confusión.
    
    :param annotations: tensor de anotaciones, de tamaño (H, W).
    :param predictions: tensor de predicciones del modelo, de tamaño (H, W).
    :param num_classes: número total de clases.
    :return: tensores con precisión, recall, f1-score, IoU y matriz de confusión.
    """
    # Aseguramos que los valores sean enteros
    annotations = annotations.astype(np.int64)
    predictions = predictions.astype(np.int64)
    
    # Flatten
    annotations = annotations.flatten()  # (H, W) → (H*W)
    predictions = predictions.flatten()  # (H, W) → (H*W)

    # Inicializamos las métricas y la matriz de confusión
    pre_per_class   = np.zeros(num_classes)
    rec_per_class   = np.zeros(num_classes)
    acc_per_class   = np.zeros(num_classes)
    f1_per_class    = np.zeros(num_classes)
    iou_per_class   = np.zeros(num_classes)
    conf_matrix     = np.zeros((num_classes, num_classes))


    # Construcción eficiente de la matriz de confusión
    for c_real in range(num_classes):
        for c_pred in range(num_classes):
            conf_matrix[c_real, c_pred] = np.sum((annotations == c_real) & (predictions == c_pred))

    # Calcular TP, FP, FN a partir de la matriz de confusión
    tp = np.diag(conf_matrix)  # Elementos de la diagonal
    fn = conf_matrix.sum(axis=1) - tp  # Fila: Total real menos TP
    fp = conf_matrix.sum(axis=0) - tp  # Columna: Total predicho menos TP
    tn = conf_matrix.sum() - (tp + fn + fp)  # Todo lo demás

    for c in range(num_classes):
        # Precision
        pre_per_class[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0

        # Recall
        rec_per_class[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0

        # Accuracy
        acc_per_class[c] = (tp[c] + tn[c]) / (tp[c] + tn[c] + fp[c] + fn[c]) if (tp[c] + tn[c] + fp[c] + fn[c]) > 0 else 0

        # F1-Score
        f1_per_class[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) if (2 * tp[c] + fp[c] + fn[c]) > 0 else 0

        # IoU
        iou_per_class[c] = tp[c] / (tp[c] + fp[c] + fn[c]) if (tp[c] + fp[c] + fn[c]) > 0 else 0

    return pre_per_class, rec_per_class, acc_per_class, f1_per_class, iou_per_class, conf_matrix

def compute_metrics_hf(annotations, predictions, num_classes):
    metrics = metric._compute(
        predictions=predictions,
        references=annotations,
        num_labels=num_classes,
        ignore_index=255,
        reduce_labels=False,
    )
    return metrics

##################################################################################
#                                      MAIN                                      #
##################################################################################
def main(nu_dataset_path:str,
         bev_dataset_path:str,
         model_path:str,
         output_path:str,
         eval_type:int,
         dataset_version:str = 'test',
         reevaluate:bool=False,
         testing:bool=False):
    check_paths([nu_dataset_path, bev_dataset_path, model_path])

    # Load Models and datasets for inference
    model_name = os.path.basename(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Using device: {device}")
    
    eval_desc = {
        0: "[Raw -> BEV -> Seg] model evaluated with BEVDataset",
        1: "[Raw -> Seg -> BEV] model evaluated with BEVDataset",
        2: "[Raw -> Seg] model evaluated with NuImagesFormattedDataset"
    }

    merging_lut_ids = None
    if eval_type == 0:
        print(f"Evaluating model {model_name} on bev images")
        model   = Raw_BEV2Seg(model_path, None, device=device)
    elif eval_type == 1:
        print(f"Evaluating model {model_name} on bev images")
        model   = Raw2Seg_BEV(model_path, None, device=device)
    elif eval_type == 2:
        print(f"Evaluating model {model_name} on raw images")
        model   = Raw2Seg_BEV(model_path, None, device=device)
    else:
        raise Exception(f"Unknown evaluation type: {eval_type}")
    
    merging_lut_ids = get_merging_strategy(model.id2label)
    dataset     = NuImagesFormattedDataset(dataroot=nu_dataset_path, version=dataset_version, id2label=model.id2label, id2color=model.id2color, merging_lut_ids=merging_lut_ids)

    if merging_lut_ids is not None:
        print(f"Using merging strategy for evaluation")

    camera_name = 'CAM_FRONT'
    labels_list = list(model.id2label.values())
    num_labels = len(labels_list)
    colors      = list(model.id2color.values())
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    # Metric Output
    data = {}
    metric_names = [ 'mean_precision_per_class', 'mean_recall_per_class', 'mean_accuracy_per_class','mean_f1score_per_class', 'mean_iou_per_class' ]
    if os.path.exists(output_path):
        print(f"Loading evaluation data from: {output_path}")
        with open(output_path, "rb") as f:
            data = pickle.load(f)

        if model_name in data and eval_type in data[model_name]:
            print(f"Model {model_name} is already evaluated with evaluation type '{eval_type}'")
            if not reevaluate:
                res = input(f"Do you want to evaluate it again? [Y/n]")
                if res.lower() != 'y':
                    print("Finish!! :D")
                    return
            else:
                print("Evaluating it again...")
    
    data[model_name] = data[model_name] if model_name in data else {} 
    data[model_name][eval_type] = {}
    data[model_name]['model_path'] = model_path
    data[model_name]['model_size'] = get_size(model)

    for m in metric_names:
        data[model_name][eval_type][m] = np.zeros(num_labels)
    data[model_name][eval_type]['conf_matrix'] = np.zeros((num_labels, num_labels))
    data[model_name][eval_type]['labels'] = labels_list
    data[model_name][eval_type]['colors'] = colors
    data[model_name][eval_type]['description'] = eval_desc[eval_type]
    data[model_name][eval_type]['dataset_version'] = dataset_version
    
    # For saving inferences
    testing_path = None
    if testing:
        testing_path = os.path.join(os.path.dirname(output_path), "eval_debug", model_name, f"eval_type_{eval_type}")
        check_or_reset_path(testing_path)


    # Run Inference instance by instance and compute metrics
    print(f"Evaluating {model_name} with {type(dataset)} Dataset")
    for i in tqdm(range(len(dataset))):

        # Get image and target
        image, target = dataset.__getitem__(i)
        image = np.asarray(image.convert("RGB"))
        target = np.asarray(target)

        # Load OpenLABEL
        openlabel_path = os.path.join(dataset.dataroot, dataset.data_tokens[i] + ".json")
        check_paths([openlabel_path])
        vcd = core.VCD()
        vcd.load_from_file(openlabel_path)
        model.set_openlabel(vcd)

        # Model inference 
        pred = None
        if eval_type == 0:
            # raw2bevseg evaluated with bev images
            bev_image, bev_mask = model.generate_bev_segmentation(image, camera_name)
            target = model.inverse_perspective_mapping(target, camera_name)
            pred = bev_mask
        elif eval_type == 1:
            # raw2segbev evaluated with bev images
            mask, bev_mask = model.generate_bev_segmentation(image, camera_name)
            target = model.inverse_perspective_mapping(target, camera_name)
            pred = bev_mask
        elif eval_type == 2:
            # raw2segbev evaluated with normal images
            mask, bev_mask = model.generate_bev_segmentation(image, camera_name)
            pred = mask
        else:
            raise Exception(f"eval_type {eval_type} does not exist")

        # distinct_labels = np.unique(pred)
        # distinct_names = [model.id2label[l] for l in distinct_labels]
        # print(f"distinct_labels: {distinct_labels}")
        # print(f"distinct_names: {distinct_names}")

        # Metric calculations        
        pre_per_class, rec_per_class, acc_per_class, f1_per_class, iou_per_class, conf_matrix = calculate_metrics(target, pred, num_labels)
        # metrics_hf = compute_metrics_hf(target, pred, num_labels) # Es lo mismo pero lo mio calcula también la matriz de confusion

        data[model_name][eval_type]['mean_precision_per_class']    += pre_per_class
        data[model_name][eval_type]['mean_recall_per_class']       += rec_per_class
        data[model_name][eval_type]['mean_accuracy_per_class']     += acc_per_class
        data[model_name][eval_type]['mean_f1score_per_class']      += f1_per_class
        data[model_name][eval_type]['mean_iou_per_class']          += iou_per_class
        data[model_name][eval_type]['conf_matrix']                 += conf_matrix
        
        if testing:
            cv2.imwrite(os.path.join(testing_path, f"{i}_raw.png"), image)
            #cv2.imwrite(os.path.join(testing_path, f"{i}_inference_.png"), pred)
            cv2.imwrite(os.path.join(testing_path, f"{i}_inference_color.png"), model.mask2image(pred))
            #cv2.imwrite(os.path.join(testing_path, f"{i}_target_.png"), target)
            cv2.imwrite(os.path.join(testing_path, f"{i}_target_color.png"), model.mask2image(target))

            if eval_type == 2:
                cv2.imwrite(os.path.join(testing_path, f"{i}inference_bev_color.png"), model.mask2image(model.inverse_perspective_mapping(pred, camera_name)))

    # Compute means
    data[model_name][eval_type]['conf_matrix'] = data[model_name][eval_type]['conf_matrix']
    for m in metric_names:
        data[model_name][eval_type][m] = data[model_name][eval_type][m] 
        data[model_name][eval_type][m] /= len(dataset)
    
    # Show report
    show_eval_report(data, model_name, eval_type, labels_list)
    
    # Save results
    print(f"Shaving evaluation data in: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print("Finish!! :D")
      

if __name__ == "__main__":
    # 
    # CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/BEVDataset --model_path ./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.5 --output_path ./data/model_evaluations.pkl
    # CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_bev2seg_2.py --models_to_evaluate 
    #
    #
    # [
    #     ('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.4'), 
    #     ('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.4'), 
    #     ('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.5'), 
    #     ('./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.5'), 
    #     ('./models/segformer_bev/raw2bevseg_mit-b0_v0.5'),
    #     ('./models/segformer_bev/raw2bevseg_mit-b0_v0.6')
    # ]
    
    parser = argparse.ArgumentParser(description="Script for evaluating BEV2Seg models.")
    parser.add_argument('--model_path', type=str, help="Path to the model. It can be a raw2segbev or raw2bevseg model")
    parser.add_argument('--nu_dataset_path', type=str, help="Path to the NuImagesFormattedDataset.")
    parser.add_argument('--bev_dataset_path', type=str, help="Path to the BEVDataset.")
    parser.add_argument('--output_path', type=str, help="Path of the resulting evaluation data")
    parser.add_argument('--version', type=str, default='test', help="Dataset version for evaluation. Default: 'test'")
    parser.add_argument('--models_to_evaluate', type=str, help="List of tuples (model_path, dataset_path, [dataset_version]) for automatic evaluation")
    parser.add_argument('--testing', action='store_true', help="Wheter to be a simple experiment or a full evaluation. If set, semantic mask are saved.")
    
    args = parser.parse_args()
    assert args.nu_dataset_path and args.bev_dataset_path and args.output_path

    if args.models_to_evaluate:
        # ✅ MODO AUTOMÁTICO: evaluar lista de modelos
        try:
            models_to_evaluate = ast.literal_eval(args.models_to_evaluate)
            assert isinstance(models_to_evaluate, list), "models_to_evaluate debe ser una lista de tuplas"
        except Exception as e:
            raise ValueError(f"Error al parsear models_to_evaluate: {e}")
        
        output_path = args.output_path  # usa el mismo output_path para todos
        check_paths([os.path.dirname(output_path)])

        # Check all paths first
        for data in models_to_evaluate:
            if not isinstance(data, str):
                check_paths([data[0]])
            else:
                check_paths([data])
        print(f"Models to be evaluated: {models_to_evaluate}")


        # Evaluate models
        for data in models_to_evaluate:
            model_path, dataset_version = None, None
            if 1 <= len(data) <= 2:
                # Tuple
                model_path = data[0]
                dataset_version = data[1] if len(data) == 2 else 'test'
            else:
                model_path = data
                dataset_version = 'test'

            assert model_path is not None and dataset_version is not None
            for evaluation_type in get_evaluation_types(model_path):
                main(nu_dataset_path=args.nu_dataset_path,
                     bev_dataset_path=args.bev_dataset_path,
                     model_path=model_path,
                     output_path=output_path,
                     eval_type=evaluation_type,
                     dataset_version=dataset_version,
                     reevaluate=True,
                     testing=args.testing)

    elif args.model_path:
        # ✅ MODO MANUAL: evaluar solo un modelo
        check_args(args)

        for evaluation_type in get_evaluation_types(args.model_path):
            main(nu_dataset_path=args.nu_dataset_path,
                bev_dataset_path=args.bev_dataset_path,
                model_path=args.model_path,
                output_path=args.output_path,
                eval_type=evaluation_type,
                dataset_version=args.version,
                testing=args.testing)
    else:
        parser.error("Debes especificar --output_path y --models_to_evaluate para evaluar varios modelos o bien --model_path --dataset_path para evaluar un único modelo.")
