
from oldatasets.BEV import BEVDataset
from oldatasets.NuImages import NuImagesFormattedDataset

import matplotlib.pyplot as plt
from matplotlib.pylab import Axes
import numpy as np

from tqdm import tqdm
from tabulate import tabulate
from typing import Literal, List, Tuple, Union

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

        num_labels = len(dataset.id2label)
        label2index = {l:i for i, l in enumerate(dataset.id2label.keys())}
        count_matrix = np.zeros(num_labels, dtype=np.int256)
        total_matrix = np.zeros(2, dtype=np.int256) # ['total_expected_pixels', 'actual_total']

        # Iterate through all samples 
        for i in tqdm(range(len(dataset))):
            _, target = dataset[i]
            target = np.asarray(target)

            assert len(target.shape) == 2

            labels_in_target = np.unique(target)
            in_target_counted   = 0
            in_target_expected  = target.shape[0] * target.shape[1]
            for l in labels_in_target:
                index = label2index[l]
                non_zero = np.count_nonzero(target == l)
                count_matrix[index] += non_zero
                in_target_counted   += non_zero
            
            assert in_target_counted == in_target_expected
            total_matrix[0] += in_target_expected
        total_matrix[1] += count_matrix.sum()
        
        # Save results in final format
        results['names']   = aux.id2label
        results['colors']  = aux.id2color
        results[version] = {}        
        results[version]['total_count'] = total_matrix[1]
        results[version]['expected_total_count'] = total_matrix[0]
        for l in range(dataset.id2label.keys()):
            index = label2index[l]
            results[version][l] = count_matrix[index]
    return results

def get_class_ratio(results:dict, 
                    data_type:Literal["bev", "nu"], 
                    data_versions:List[Literal['mini', 'train', 'val', 'test']]) -> List[Tuple[str, float]]:
        
    # Get data from results
    assert data_type in results
    id2label = results[data_type]['names']
    id2index = {l:i for i, l in enumerate(id2label.keys())}
    num_labels = len(id2label.keys())
    count_matrix = np.zeros(num_labels)
    
    sum_of_expected_total = 0
    sum_of_provided_total = 0

    for data_version in data_versions:
        assert data_version in results[data_type]
        
        sum_of_expected_total += results[data_type][data_version]['expected_total_count']
        sum_of_provided_total += results[data_type][data_version]['total_count']

        for l in id2label.keys():
            v = results[data_type][data_version][l]
            index = id2index[l]
            count_matrix[index] += v
    total_count = count_matrix.sum()
    
    assert total_count == sum_of_expected_total
    print(f"expected_total: {sum_of_expected_total} | provided_total: {sum_of_provided_total} | total: {total_count}")


    # Compute ratio
    ratio_matrix = np.zeros(num_labels)
    for l in id2label.keys():
        index = id2index[l]
        ratio_matrix[index] = count_matrix[index] / total_count 
    # assert ratio_matrix.sum() == 1.0
    
    # Show ratios
    headers = ["Label", "Ratio (0-1)"]
    ratios  = [[v, ratio_matrix[id2index[k]]] for k, v in id2label.items()]
    inf = tabulate(ratios + [("total", ratio_matrix.sum())], headers=headers)
    print(inf)
    return ratios


def plot_ratios(ratios:List[Tuple[str, float]],
                data_type:Literal["bev", "nu"], 
                bar_width:float=0.4, 
                bar_number:int=0,
                bar_align:str='center',
                bar_color:Union[str, Tuple[float]] = None,
                bar_edgecolor:Tuple[float] = (0.0, 0.0, 0.0), 
                ax: Axes = None):
    if ax is None:
        ax = plt.gca()
    
    if data_type == "bev":
        title = "Class balance in BEV images"
        label_name = "BEV"
    elif data_type == "nu":
        title = "Class balance in normal images"
        label_name = "Normal"
    else:
        raise Exception("Invalid dataset type")
    
    # Plot 
    names, counts = [n for n,_ in ratios], [c for _,c in ratios]

    xs = np.array(list(range(len(names))))
    if bar_number == 0:
        xs = xs - bar_width/2
    elif bar_number == 1:
        xs = xs + bar_width/2

    ax.bar(x=xs, height=counts,
           width=bar_width, 
           align=bar_align,
           color=bar_color if bar_color is not None else colors,
           edgecolor=bar_edgecolor, 
           label=label_name)
    
    xs = np.array(list(range(len(names))))
    ax.set_xticks(xs, names)
    
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Ratio")

def plot_results(results:dict, 
                 data_type:Literal["bev", "nu"], 
                 data_versions:List[Literal['mini', 'train', 'val', 'test']], 
                 bar_width:float=0.4, 
                 bar_number:int=0,
                 bar_align:str='center',
                 bar_color:Union[str, Tuple[float]] = None,
                 bar_edgecolor:Tuple[float] = (0.0, 0.0, 0.0), 
                 ax: Axes = None):
    if ax is None:
        ax = plt.gca()
    
    if data_type == "bev":
        title = "Class balance in BEV images"
        label_name = "BEV"
    elif data_type == "nu":
        title = "Class balance in normal images"
        label_name = "Normal"
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
    names, counts, colors = [], [], []
    for i, label_id in enumerate(data['labels'].keys()):
        name = results[data_type]['names'][label_id]
        
        # if name == 'background':
        #     continue

        mean_count = data['labels'][label_id] #/ data['total'] # Normalize

        color = np.array(results[data_type]['colors'][label_id])
        color = color / 255.0 if np.max(color) > 1.0 else color

        names.append(name)
        counts.append(mean_count)
        colors.append(color)


    xs = np.array(list(range(len(names))))
    if bar_number == 0:
        xs = xs - bar_width/2
    elif bar_number == 1:
        xs = xs + bar_width/2

    ax.bar(x=xs, height=counts,
           width=bar_width, 
           align=bar_align,
           color=bar_color if bar_color is not None else colors,
           edgecolor=bar_edgecolor, 
           label=label_name)
    
    xs = np.array(list(range(len(names))))
    ax.set_xticks(xs, names)
    
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of pixels")


def main(output_path:str, bevdataset_path:str="./tmp/BEVDataset", nudataset_path:str="./tmp/NuImagesFormatted", recalc:bool=False, plot_results_flag:bool = True):
    results = None

    already_computed = False
    try:
        check_paths([output_path])
        already_computed = True
    except:
        print(f"File {output_path} doesn't exist")
        
    if already_computed:
        if recalc:
            print(f"Recomputing results and dumping into file {output_path}")
            bev_results = get_dataset_class_balance(bevdataset_path, dataset_type="bev")
            nu_results  = get_dataset_class_balance(nudataset_path, dataset_type="nu")
            results = {"bevdataset_path":bevdataset_path, "nudataset_path":nudataset_path, "bev":bev_results, "nu":nu_results}
            dump_results(output_path, results)
        else:
            print(f"Loading results from file {output_path}")
            results = read_results(output_path)
    else:
        print(f"Computing results and dumping into file {output_path}")
        bev_results = get_dataset_class_balance(bevdataset_path, dataset_type="bev")
        nu_results  = get_dataset_class_balance(nudataset_path, dataset_type="nu")
        results = {"bevdataset_path":bevdataset_path, "nudataset_path":nudataset_path, "bev":bev_results, "nu":nu_results}
        dump_results(output_path, results)
    


    if not plot_results_flag:
        print("Finished :D!!")
        return 

    print("Class distribution in NuImagesFormatted Dataset")
    nu_ratios = get_class_ratio(results, data_type="nu", data_versions=["train", "val", "test"])
    print("\nClass distribution in BEVDataset")
    bev_ratios = get_class_ratio(results, data_type="bev", data_versions=["train", "val", "test"])
    print()
    


    ax = plt.gca()
    plot_results(results, data_type='nu',  bar_number=0, data_versions=["train", "val", "test"], bar_color="#FF0064", bar_edgecolor=None, ax=ax)
    plot_results(results, data_type='bev', bar_number=1, data_versions=["train", "val", "test"], bar_color="#39C39E", bar_edgecolor=None, ax=ax)
    # plot_ratios(nu_ratios, data_type='nu',  bar_number=0, bar_color="#FF0064", bar_edgecolor=None, ax=ax)
    # plot_ratios(bev_ratios, data_type='bev',  bar_number=1, bar_color="#39C39E", bar_edgecolor=None, ax=ax)
    ax.set_title("Class balance in datasets")
    ax.legend(loc='upper right', ncols=2)
    plt.xticks(rotation=90, fontsize=5)
    plt.tight_layout()
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    # python3 notebooks/BEVDataset_class_balance.py ./data/class_balance.json --plot_results True
    main(output_path="./data/class_balance.json",
         recalc=True,
         plot_results_flag=True,
         bevdataset_path="./tmp/BEVDataset",
         nudataset_path="./tmp/NuImagesFormatted")
    exit()

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
