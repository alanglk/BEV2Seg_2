
from oldatasets.BEV import BEVDataset
from oldatasets.NuImages import NuImagesFormattedDataset

import matplotlib.pyplot as plt
from matplotlib.pylab import Axes
import numpy as np

from tqdm import tqdm
from typing import Literal, List, Tuple

import argparse
import json
import os

def check_paths(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"path doesnt exist: {path}")
def dump_results(output_path:str, results:dict):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
def read_results(output_path:str):
    with open(output_path, "r") as f:
        return json.load(f)
def get_dataset_class_balance(dataset_path:str, dataset_type:Literal["bev", "nu"], dataset_versions:List[str] = None):
    check_paths([dataset_path])

    if dataset_type == "bev":
        aux = BEVDataset(dataroot=dataset_path, version='mini')
        dataset_versions = BEVDataset.DATASET_VERSIONS if dataset_versions is None else dataset_versions
    elif dataset_type == "nu":
        aux = NuImagesFormattedDataset(dataroot=dataset_path, version='mini')
        dataset_versions = NuImagesFormattedDataset.DATASET_VERSIONS if dataset_versions is None else dataset_versions
    else:
        raise Exception("Invalid dataset type")

    # Create result data estructure
    id2count = { k:0.0 for k in aux.id2label.keys() }
    id2count['total_expected_pixels'] = 0.0


    # Iterate through selected dataset versions
    results = {}
    for version in dataset_versions:
        # Create dataset object
        if dataset_type == "bev":
            dataset = BEVDataset(dataroot=dataset_path, version=version)
        elif dataset_type == "nu":
            dataset = NuImagesFormattedDataset(dataroot=dataset_path, version=version)
        else:
            raise Exception("Invalid dataset type")
        print(f"{dataset_type} {version} split...")

        # Iterate through all samples 
        for i in tqdm(range(len(dataset))):
            _, target = dataset[i]
            target = np.asarray(target)

            labels_in_target = np.unique(target)
            for l in labels_in_target:
                assert l in id2count
                id2count[l] += np.count_nonzero(target == l)
            id2count['total_expected_pixels'] += target.shape[0] * target.shape[1]
        
        # Compute summ of all labeled pixels
        actual_total = 0.0
        for k, v in id2count.items():
            if k == 'total_expected_pixels':
                continue
            actual_total += v
        id2count['actual_total'] = actual_total

        # Save results in final format
        results['names']   = aux.id2label
        results['colors']  = aux.id2color
        results[version] = {}        
        results[version]['total_count'] = id2count['actual_total']
        results[version]['expected_total_count'] = id2count['total_expected_pixels']
        for k, v in id2count.items():
            if k == 'total_expected_pixels' or k == 'actual_total':
                continue
            results[version][k] = v
    return results
def plot_results(results:dict, 
                 data_type:Literal["bev", "nu"], 
                 data_versions:List[Literal['mini', 'train', 'val', 'test']], 
                 bar_width:float=0.8, 
                 bar_edgecolor:Tuple[float] = (0.0, 0.0, 0.0), 
                 ax: Axes = None):
    if ax is None:
        ax = plt.gca()
    
    if data_type == "bev":
        title = "Class balance in BEV images"
    elif data_type == "nu":
        title = "Class balance in normal images"
    else:
        raise Exception("Invalid dataset type")
    
    # Get data from results
    data = {'total': 0.0, 'labels': {}}
    for data_version in results[data_type].keys():
        if data_version in data_versions:
            total_for_label = results[data_type][data_version]['total_count']
            for label, count in results[data_type][data_version].items():
                if label == 'total_count' or label == 'expected_total_count':
                    continue
                if label not in data:
                    data['labels'][label] = 0.0
                data['labels'][label] += count
                data['total'] += total_for_label
    
    # Plot 
    xs = range(len(data['labels']))
    for x, label_id in zip(xs, data['labels'].keys()):
        name = results[data_type]['names'][label_id]
        color = np.array(results[data_type]['colors'][label_id])
        color_float = color / 255.0 if np.max(color) > 1.0 else color
        mean_count = data['labels'][label_id] / data['total'] # Normalize

        ax.bar(x=name, height=mean_count,
               width=bar_width, 
               align='center',
               color=color_float,
               edgecolor=bar_edgecolor, 
               label=name)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Mean count")


def main(output_path:str, bevdataset_path:str="./tmp/BEVDataset", nudataset_path:str="./tmp/NuImagesFormatted", plot_results_flag:bool = True):
    results = None
    try:
        check_paths([output_path])
        print(f"Loading results from file {output_path}")
        results = read_results(output_path)
    except:
        print(f"Computing results and dumping into file {output_path}")
        bev_results = get_dataset_class_balance(bevdataset_path, dataset_type="bev")
        nu_results  = get_dataset_class_balance(nudataset_path, dataset_type="nu")
        results = {"bevdataset_path":bevdataset_path, "nudataset_path":nudataset_path, "bev":bev_results, "nu":nu_results}
        dump_results(output_path, results)

    if not plot_results_flag:
        print("Finished :D!!")
        return 

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    plot_results(results, data_type='bev',  data_versions=["train", "val", "test"], ax=axes[0])
    plot_results(results, data_type='nu',   data_versions=["train", "val", "test"], ax=axes[1])

    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get class balance of BEVDataset and NuDataset")
    parser.add_argument('output_path', type=str, help="Output json file")
    parser.add_argument('--bevdataset_path', default="./tmp/BEVDataset", help="[Optional] path of BEVDataset. by default: './tmp/BEVDataset'")
    parser.add_argument('--nudataset_path', default="./tmp/NuImagesFormatted", help="[Optional] path of NuImagesFormatted. by default: './tmp/NuImagesFormatted'")
    parser.add_argument('--plot_results', type=bool, default=True, help="[Optional] Wheter to plot results.")

    args = parser.parse_args()
    main(output_path=args.output_path,
         plot_results_flag=args.plot_results,
         bevdataset_path=args.bevdataset_path,
         nudataset_path=args.nudataset_path)
