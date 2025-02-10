#!/opt/conda/bin/python3
"""
Script para evaluar BEV2Seg2.
Ejemplo de uso:

raw2bevseg (eval type 0):
python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/BEVDataset --model_path ./models/segformer_bev/raw2bevseg_mit-b0_v0.3 --output_path ./data/model_evaluations.pkl

raw2segbev (eval types 1 and 2):
python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/BEVDataset --model_path ./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.3 --output_path ./data/model_evaluations.pkl
python3 scripts/evaluate_bev2seg_2.py --dataset_path ./tmp/NuImagesFormatted --model_path ./models/segformer_nu_formatted/raw2segbev_mit-b0_v0.3 --output_path ./data/model_evaluations.pkl

"""

from oldatasets.BEV import BEVDataset
from oldatasets.NuImages import NuImagesFormattedDataset
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
import os


def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")

def check_args(args) -> int:
    """
    INPUT: args
    OUTPUT: Evaluation type
        - 0 for raw2bevseg on BEVDataset
        - 1 for raw2segbev on BEVDataset
        - 2 for raw2segbev on NuImagesFormattedDataset
    """
    check_paths([args.dataset_path, args.model_path, os.path.dirname(args.output_path)])

    model_name = os.path.basename(args.model_path)
    dataset_name = os.path.basename(args.dataset_path)
    
    evaluation_type = -1

    if model_name.find("raw2segbev") != -1:
        if dataset_name.find("NuImagesFormatted") != -1:
            evaluation_type = 2
        elif dataset_name.find("BEVDataset") != -1:
            evaluation_type = 1
    elif model_name.find("raw2bevseg") != -1:
        if dataset_name.find("BEVDataset") != -1:
            evaluation_type = 0
    
    if evaluation_type == -1:
        raise Exception(f"Wrong dataset {dataset_name} for model {model_name}")

    return evaluation_type

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

def calculate_metrics(annotations, predictions, num_classes):
    """
    Calcula las métricas de las predicciones y la matriz de confusión.
    
    :param annotations: tensor de anotaciones, de tamaño (H, W).
    :param predictions: tensor de predicciones del modelo, de tamaño (H, W).
    :param num_classes: número total de clases.
    :return: tensores con precisión, recall, f1-score, IoU y matriz de confusión.
    """
    # Aseguramos que los valores sean enteros
    annotations = annotations.to(torch.int64)
    predictions = predictions.to(torch.int64)
    
    # Flatten
    annotations = annotations.view(-1)  # (H, W) → (H*W)
    predictions = predictions.view(-1)  # (H, W) → (H*W)

    # Inicializamos las métricas y la matriz de confusión
    pre_per_class   = torch.zeros(num_classes, device=predictions.device)
    rec_per_class   = torch.zeros(num_classes, device=predictions.device)
    acc_per_class   = torch.zeros(num_classes, device=predictions.device)
    f1_per_class    = torch.zeros(num_classes, device=predictions.device)
    iou_per_class   = torch.zeros(num_classes, device=predictions.device)
    conf_matrix     = torch.zeros((num_classes, num_classes), device=predictions.device)


    # Construcción eficiente de la matriz de confusión
    for c_real in range(num_classes):
        for c_pred in range(num_classes):
            conf_matrix[c_real, c_pred] = torch.sum((annotations == c_real) & (predictions == c_pred))

    # Calcular TP, FP, FN a partir de la matriz de confusión
    tp = torch.diag(conf_matrix)  # Elementos de la diagonal
    fn = conf_matrix.sum(dim=1) - tp  # Fila: Total real menos TP
    fp = conf_matrix.sum(dim=0) - tp  # Columna: Total predicho menos TP
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

##################################################################################
#                                      MAIN                                      #
##################################################################################
def main(dataset_path:str, 
         model_path:str,
         output_path:str,
         eval_type:int,
         dataset_version:str = 'test'):
    check_paths([dataset_path, model_path])

    # Load Models and datasets for inference
    model_name = os.path.basename(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Using device: {device}")
    
    eval_desc = {
        0: "[Raw -> BEV -> Seg] model evaluated with BEVDataset",
        1: "[Raw -> Seg -> BEV] model evaluated with BEVDataset",
        2: "[Raw -> Seg] model evaluated with NuImagesFormattedDataset"
    }

    if eval_type == 0:
        print(f"Evaluating model {model_name} on bev images Dataset")
        model   = Raw_BEV2Seg(model_path, None, device=device)
        dataset = BEVDataset(dataroot=dataset_path, version=dataset_version)
    elif eval_type == 1:
        print(f"Evaluating model {model_name} on bev images Dataset")
        model   = Raw2Seg_BEV(model_path, None, device=device)
        dataset = BEVDataset(dataroot=dataset_path, version=dataset_version)
    elif eval_type == 2:
        print(f"Evaluating model {model_name} on raw images Dataset")
        model   = Raw2Seg_BEV(model_path, None, device=device)
        dataset     = NuImagesFormattedDataset(dataroot=dataset_path, version=dataset_version)
    else:
        raise Exception(f"Unknown evaluation type: {evaluation_type}")
    
    camera_name = 'CAM_FRONT'
    labels_list = list(model.id2label.values())
    num_labels = len(labels_list)
    colors      = list(model.id2color.values())
    colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    # Dataloader

    # Metric Output
    data = {}
    metric_names = [ 'mean_precision_per_class', 'mean_recall_per_class', 'mean_accuracy_per_class','mean_f1score_per_class', 'mean_iou_per_class' ]
    if os.path.exists(output_path):
        print(f"Loading evaluation data from: {output_path}")
        with open(output_path, "rb") as f:
            data = pickle.load(f)

        if model_name in data and evaluation_type in data[model_name]:
            res = input(f"Model {model_name} is already evaluated with evaluation type '{eval_type}'. Do you want to evaluate it again? [Y/n]")
            if res.lower() != 'y':
                print("Finish!! :D")
                return
    
    data[model_name] = data[model_name] if model_name in data else {} 
    data[model_name][eval_type] = {}
    
    for m in metric_names:
        data[model_name][eval_type][m] = torch.zeros(num_labels, device=device)
    data[model_name][eval_type]['conf_matrix'] = torch.zeros((num_labels, num_labels), device=device)
    data[model_name][eval_type]['labels'] = labels_list
    data[model_name][eval_type]['colors'] = colors
    data[model_name][eval_type]['description'] = eval_desc[eval_type]
    data[model_name][eval_type]['dataset_version'] = dataset_version
    
    # Run Inference instance by instance and compute metrics
    print(f"Evaluating {model_name} with {type(dataset)} Dataset")
    for i in tqdm(range(len(dataset))):
        image, target = dataset[i]
        image_path, target_path = dataset._get_item_paths(i)
        openlabel_path          = os.path.join(dataset.dataroot, dataset.data_tokens[i] + ".json")
        check_paths([image_path, target_path, openlabel_path])

        # Load files
        vcd = core.VCD()
        vcd.load_from_file(openlabel_path)
        model.set_openlabel(vcd)
        image       = cv2.imread(image_path)
        target      = torch.tensor(cv2.imread(target_path)[:, :, 0], device=device)

        # Inference and metric calculations
        if evaluation_type == 2:
            pred, _ = model.generate_bev_segmentation(image, camera_name)
        else:
            _, pred = model.generate_bev_segmentation(image, camera_name)
        pred = torch.tensor(pred, device=device)
        pre_per_class, rec_per_class, acc_per_class, f1_per_class, iou_per_class, conf_matrix = calculate_metrics(target, pred, num_labels)

        data[model_name][eval_type]['mean_precision_per_class']    += pre_per_class
        data[model_name][eval_type]['mean_recall_per_class']       += rec_per_class
        data[model_name][eval_type]['mean_accuracy_per_class']     += acc_per_class
        data[model_name][eval_type]['mean_f1score_per_class']      += f1_per_class
        data[model_name][eval_type]['mean_iou_per_class']          += iou_per_class
        data[model_name][eval_type]['conf_matrix']                 += conf_matrix
        
    # Compute means
    data[model_name][eval_type]['conf_matrix'] = data[model_name][eval_type]['conf_matrix'].cpu().detach().numpy() 
    for m in metric_names:
        data[model_name][eval_type][m] = data[model_name][eval_type][m].cpu().detach().numpy() 
        data[model_name][eval_type][m] /= len(dataset)
    
    # Show report
    show_eval_report(data, model_name, eval_type, labels_list)
    
    # Save results
    print(f"Shaving evaluation data in: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print("Finish!! :D")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluating BEV2Seg models.")
    parser.add_argument('--model_path', type=str, help="Path to the model. It can be a raw2segbev or raw2bevseg model")
    parser.add_argument('--dataset_path', type=str, help="Path to the dataset. NuImagesFormattedDataset or BEVDataset.")
    parser.add_argument('--output_path', type=str, help="Path of the resulting evaluation data")

    parser.add_argument('--version', type=str, default='test', help="Dataset version for evaluation. Default: 'test'")

    args = parser.parse_args()
    evaluation_type = check_args(args)
    main(dataset_path=args.dataset_path, 
         model_path=args.model_path, 
         output_path=args.output_path,
         eval_type=evaluation_type, 
         dataset_version=args.version)