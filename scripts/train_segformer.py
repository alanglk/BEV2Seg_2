#!/opt/conda/bin/python3
"""
Script para finetunear SegFormer.
Ejemplo de uso:

python3 srcipts/train_segformer.py <config_file_path>
"""

from transformers import SegformerImageProcessor
from transformers import SegformerConfig
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from oldatasets.BEV import BEVFeatureExtractionDataset
from oldatasets.NuImages import NuImagesFeatureExtractionDataset
from oldatasets.NuImages import NuImagesFormattedFeatureExtractionDataset

# For merging labels
from oldatasets.NuImages.nulabels import DEFAULT_MERGE_DICT, nuid2name, nuname2label, nuid2color, nuid2dynamic, get_merged_nulabels

import argparse
import toml
import os

from PIL import Image

import warnings
warnings.filterwarnings("ignore")
metric = evaluate.load("mean_iou")

import datasets
datasets.Image

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    # scale the logits to the size of the label
    pred_ids = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    return pred_ids

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    metrics = metric._compute(
        predictions=logits,
        references=labels,
        num_labels=num_labels,
        ignore_index=255,
        reduce_labels=False,
    )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()
    # metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    # metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
    return metrics

########################## MAIN ##########################
def main(config: dict):
    print("Segformer training configuration:")
    print(config)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print()

    dataset_type    = config['data']['type']
    dataroot        = config['data']['dataroot']
    testing         = config['data']['testing']

    output_path     = config['model']['output_path']
    model_name      = config['model']['model_name']
    pretrained      = config['model']['pretrained']
    
    # Hyperparams
    batch_size      = config['training']['batch_size']
    train_epochs    = config['training']['epochs']
    initial_lr      = config['training']['learning_rate']
    weight_decay    = config['training']['weight_decay']

    # Optional args
    ls_steps                = config['training']['ls_steps']                if 'ls_steps' in config['training']                 else 100
    data_augmentation       = config['training']['data_augmentation']       if 'data_augmentation' in config['training']        else False
    data_augmentation_type  = config['training']['data_augmentation_type']  if 'data_augmentation_type' in config['training']   else "use_torch_transforms"
    num_workers             = config['training']['dataloader_num_workers']  if 'dataloader_num_workers' in config['training']   else 0
    val_acc_steps           = config['training']['val_acc_steps']           if 'val_acc_steps' in config['training']            else None
    resume_from_checkpoint  = config['training']['resume_from_checkpoint']  if 'resume_from_checkpoint' in config['training']   else False
    optim                   = config['training']['optim']                   if 'optim' in config['training']                    else "adamw_torch"
    optim_args              = config['training']['optim_args']              if 'optim_args' in config['training']               else {}
    lr_scheduler_type       = config['training']['lr_scheduler_type']       if 'lr_scheduler_type' in config['training']        else "linear"
    lr_scheduler_kwargs     = config['training']['lr_scheduler_kwargs']     if 'lr_scheduler_kwargs' in config['training']      else {}
    gradient_accum_steps    = config['training']['gradient_accum_steps']    if 'gradient_accum_steps' in config['training']     else 1
    merge_semantic_labels   = config['training']['merge_semantic_labels']   if 'merge_semantic_labels' in config['training']    else False
    

    
    # Data Augmentations
    transforms = None 
    if data_augmentation:
        # Defined depending on the Dataset Type
        if dataset_type == 'BEVDataset':
            if data_augmentation_type == "use_torch_transforms":
                transforms = v2.Compose([
                    v2.RandomResizedCrop(size=(512, 512), ratio=(0.5, 2.0)),
                    v2.RandomHorizontalFlip(p=0.5)
                ])
            elif data_augmentation_type == "use_custom_bev_transforms":
                transforms = {
                    'multiple_rotations':False,
                    'use_random': True,
                    'rx': (0.0, 0.0),
                    'ry': (-0.25, 0.25),
                    'rz': (-0.25, 0.25)
                }
            else:
                raise Exception(f"ERROR: Unknow data augmentation type: {data_augmentation_type}")
            
        elif dataset_type == 'NuImages':
            pass
        elif dataset_type == 'NuImagesFormatted':
            if data_augmentation_type == "use_torch_transforms":
                transforms = v2.Compose([
                    v2.RandomResizedCrop(size=(512, 512), ratio=(0.5, 2.0)),
                    v2.RandomHorizontalFlip(p=0.5)
                ])
            else:
                raise Exception(f"ERROR: Unknow data augmentation type: {data_augmentation_type}")
    
    # Image Processor
    image_processor = SegformerImageProcessor(reduce_labels=False)

    # Merge Labels
    global num_labels, id2label
    id2label = nuid2name
    id2color = nuid2color
    merging_lut_ids = None
    if merge_semantic_labels:
        id2label,_,id2color,_, merging_lut_ids, _ = get_merged_nulabels(id2label, nuname2label, id2color, nuid2dynamic, DEFAULT_MERGE_DICT)

    # Dataset and Dataloader
    if dataset_type == 'BEVDataset':
        if testing:
            train_dataset   = BEVFeatureExtractionDataset(dataroot=dataroot, version='mini', image_processor=image_processor, transforms=transforms, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)
            eval_dataset    = BEVFeatureExtractionDataset(dataroot=dataroot, version='mini', image_processor=image_processor, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)
        else:
            train_dataset   = BEVFeatureExtractionDataset(dataroot=dataroot, version='train', image_processor=image_processor, transforms=transforms, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)
            eval_dataset    = BEVFeatureExtractionDataset(dataroot=dataroot, version='val',   image_processor=image_processor, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)

    elif dataset_type == 'NuImages':
        if testing:
            train_dataset   = NuImagesFeatureExtractionDataset( dataroot=dataroot, version='mini', image_processor=image_processor, camera='CAM_FRONT', transforms=transforms)
            eval_dataset    = NuImagesFeatureExtractionDataset( dataroot=dataroot, version='mini', image_processor=image_processor, camera='CAM_FRONT' )
        else:
            train_dataset   = NuImagesFeatureExtractionDataset( dataroot=dataroot, version='train', image_processor=image_processor, camera='CAM_FRONT', transforms=transforms)
            eval_dataset    = NuImagesFeatureExtractionDataset( dataroot=dataroot, version='val',   image_processor=image_processor, camera='CAM_FRONT' )
    
    elif dataset_type == 'NuImagesFormatted':
        if testing:
            train_dataset   = NuImagesFormattedFeatureExtractionDataset(dataroot=dataroot, version='mini', image_processor=image_processor, transforms=transforms, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)
            eval_dataset    = NuImagesFormattedFeatureExtractionDataset(dataroot=dataroot, version='mini', image_processor=image_processor, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)
        else:
            train_dataset   = NuImagesFormattedFeatureExtractionDataset(dataroot=dataroot, version='train', image_processor=image_processor, transforms=transforms, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)
            eval_dataset    = NuImagesFormattedFeatureExtractionDataset(dataroot=dataroot, version='val',   image_processor=image_processor, id2label=id2label, id2color=id2color, merging_lut_ids=merging_lut_ids)
    id2label = train_dataset.id2label
    num_labels = len(train_dataset.id2label)
    
    # Model
    print(f"Loading pretrained model: {pretrained}")
    segformer_config = SegformerConfig.from_pretrained(pretrained)
    segformer_config.num_labels = len(train_dataset.id2label)
    segformer_config.id2label = train_dataset.id2label
    segformer_config.id2color = train_dataset.id2color
    segformer_config.label2id = train_dataset.label2id

    model = SegformerForSemanticSegmentation.from_pretrained(pretrained, config=segformer_config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"USING DEVICE: {device}")

    # Training Args
    training_args = TrainingArguments(
        output_dir=os.path.join(output_path, model_name),
        overwrite_output_dir = False,
        dataloader_num_workers=num_workers,

        # Evaluation config
        eval_strategy="steps",      # ["no", "steps", "epoch"]
        #eval_steps=... (default: logging_steps if eval_strategy="steps" is set)
        metric_for_best_model= "eval_loss", # "mean_accuracy",
        #metric_for_best_model="loss",
        eval_accumulation_steps=val_acc_steps,

        # Checkpointing config
        save_strategy="steps",      # ["no", "steps", "epoch"]
        save_steps=ls_steps,        # (default: 500) only applied if save_strategy="steps" 
        save_total_limit=5,         # save the best 4 checkpoints and the final model
        load_best_model_at_end=True,
        
        # Logging config
        logging_strategy="steps",   # ["no", "steps", "epoch"]
        logging_first_step=True,
        logging_steps=ls_steps,   # (default: 500) only applied if logging_strategy="steps" 
        report_to="tensorboard",
        
        # Optimizer
        optim=optim,
        optim_args=optim_args,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,

        # Hyperparameters
        learning_rate=initial_lr,
        weight_decay = weight_decay, 
        per_device_train_batch_size=batch_size, # The batch size per GPU/TPU core/CPU 
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accum_steps,
        num_train_epochs=train_epochs,

    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    trainer.train(resume_from_checkpoint= resume_from_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para procesar datos con diferentes versiones.")
    parser.add_argument('config_file', type=str, help="Configuration .toml file")
    args = parser.parse_args()

    config = toml.load(args.config_file)
    main(config)

    # For debugging
    # debug_config = {
    #     "data": {
    #         "type": "NuImagesFormatted",
    #         "dataroot": "./tmp/NuImagesFormatted",
    #         "testing": True
    #     },
    #     "model": {
    #         "output_path": "./tmp/models",
    #         "model_name": "segformer_bev",
    #         "pretrained": "nvidia/mit-b0"
    #     },
    #     "training": {
    #         "batch_size": 8,
    #         "epochs": 1,
    #         "learning_rate": 0.00006,
    #         "weight_decay": 0.0,
    #         "ls_steps": 1,
    #         "dataloader_num_workers": 0,
    #         "val_acc_steps": 32,
    #         "resume_from_checkpoint": False,
    #         "optim": "adamw_torch",
    #         "lr_scheduler_type": "polynomial"
    #     }
    # }
    # main(debug_config)