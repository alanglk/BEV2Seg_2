#!/opt/conda/bin/python3
"""
Script para finetunear SegFormer.
Ejemplo de uso:

python3 srcipts/train_segformer.py <config_file_path>
"""

from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

import torch
import torch.nn.functional as F
from oldatasets.BEV import BEVFeatureExtractionDataset

import argparse
import toml
import os

import warnings
warnings.filterwarnings("ignore")
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = F.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels,
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
    output_path = config['model']['output_path']
    model_name = config['model']['model_name']
    
    # Hyperparams
    ls_steps        = 4     #Logging and Saving steps
    batch_size      = config['training']['batch_size']
    train_epochs    = config['training']['epochs']
    initial_lr      = config['training']['learning_rate']
    weight_decay    = config['training']['weight_decay']

    # Dataset and Dataloader
    image_processor = SegformerImageProcessor(reduce_labels=True)
    
    
    if config['data']['type'] == 'BEVDataset':
        if config['data']['testing'] == True:
            train_dataset   = BEVFeatureExtractionDataset(dataroot=config['data']['dataroot'], version='mini', image_processor=image_processor)
            eval_dataset    = BEVFeatureExtractionDataset(dataroot=config['data']['dataroot'], version='mini', image_processor=image_processor)
        else:
            train_dataset   = BEVFeatureExtractionDataset(dataroot=config['data']['dataroot'], version='train', image_processor=image_processor)
            eval_dataset    = BEVFeatureExtractionDataset(dataroot=config['data']['dataroot'], version='val',   image_processor=image_processor)

    else:
       raise Exception(f"Dataset type: {config['data']['type']} not supported")

    
    global num_labels, id2label
    id2label = train_dataset.id2label
    num_labels = len(train_dataset.id2label)

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained(config['model']['pretrained'],
                                                          num_labels=len(train_dataset.id2label),
                                                          id2label=train_dataset.id2label,
                                                          label2id=train_dataset.label2id)
    
    # Training Args
    training_args = TrainingArguments(
        output_dir=os.path.join(output_path, model_name),
        overwrite_output_dir = False,

        # Evaluation config
        eval_strategy="epoch",      # ["no", "steps", "epoch"]
        #eval_steps=... (default: logging_steps if eval_strategy="steps" is set)
        metric_for_best_model="mean_accuracy",

        # Checkpointing config
        save_strategy="epoch",      # ["no", "steps", "epoch"]
        # save_steps=ls_steps,      # (default: 500) only applied if save_strategy="steps" 
        save_total_limit=5,         # save the best 4 checkpoints and the final model
        load_best_model_at_end=True,
        
        # Logging config
        logging_strategy="epoch",   # ["no", "steps", "epoch"]
        logging_first_step=True,
        # logging_steps=ls_steps,   # (default: 500) only applied if logging_strategy="steps" 
        report_to="tensorboard",
        
        # Hyperparameters
        learning_rate=initial_lr,
        weight_decay = weight_decay, 
        per_device_train_batch_size=batch_size, # The batch size per GPU/TPU core/CPU 
        per_device_eval_batch_size=batch_size,
        num_train_epochs=train_epochs,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para procesar datos con diferentes versiones.")
    parser.add_argument('config_file', type=str, help="Configuration .toml file")
    args = parser.parse_args()

    config = toml.load(args.config_file)
    main(config)