from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
import evaluate

from oldatasets.BEV import BEVFeatureExtractionDataset
from oldatasets.common import display_images

from TrainingLogger import TrainingLogger

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
import numpy as np
import argparse
import os

# Compute metrics
def compute_metrics(metric: evaluate.EvaluationModule, predicted, labels, num_labels):
  predicted = predicted.detach().cpu().numpy()
  labels    = labels.detach().cpu().numpy()
  metrics = metric.compute(
        predictions=predicted,
        references=labels,
        num_labels=num_labels,
        ignore_index=255,
        reduce_labels=False # we've already reduced the labels ourselves
        )
  
  return metrics

# Training Loop
def train_loop(model, device: torch.device, optimizer:torch.optim, train_dataloader: DataLoader, metric: evaluate.EvaluationModule):
  train_loss, train_iou, train_acc = 0.0, 0.0, 0.0

  model.train()
  for idx, batch in enumerate(tqdm(train_dataloader)):
    # get the inputs;
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)  

    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss, logits = outputs.loss, outputs.logits

    loss.backward()
    optimizer.step()

    # Compute training metrics
    with torch.no_grad():
      upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
      predicted = upsampled_logits.argmax(dim=1)
      metrics = compute_metrics(metric, predicted, labels, model.config.num_labels)
      train_loss  += loss.item()
      train_iou   += metrics["mean_iou"]
      train_iou   += metrics["mean_accuracy"]
  
  n = len(train_dataloader)
  train_metrics = { "loss": train_loss / n, "mean_iou": train_iou / n, "mean_acc": train_acc / n }
  return train_metrics


# Evaluation Loop
def evaluation_loop(model, device: torch.device, eval_dataloader: DataLoader, metric: evaluate.EvaluationModule):
  eval_loss, eval_iou, eval_acc = 0.0, 0.0, 0.0
  with torch.no_grad():
    for idx, batch in enumerate(tqdm(eval_dataloader)):
      pixel_values = batch["pixel_values"].to(device)
      labels = batch["labels"].to(device)

      # prediction
      outputs = model(pixel_values=pixel_values, labels=labels)
      loss, logits = outputs.loss, outputs.logits
      upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
      predicted = upsampled_logits.argmax(dim=1)

      # Compute evaluation metrics
      metrics = compute_metrics(metric, predicted, labels, model.config.num_labels)
      eval_loss  += loss.item()
      eval_iou   += metrics["mean_iou"]
      eval_iou   += metrics["mean_accuracy"]
  
  n = len(eval_dataloader)
  eval_metrics = { "loss": eval_loss / n, "mean_iou": eval_iou / n, "mean_acc": eval_acc / n }
  return eval_metrics

def save_checkpoint(model, output_path, model_name, epoch:int, overwrite=False):
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  
  checkpoint_path = os.path.join(output_path, model_name, f"_e{epoch}.pt")

  if os.path.exists(checkpoint_path) and not overwrite:
    raise Exception(f"Error saving checkpoint. Checkpoint path already exists: {checkpoint_path}")

  model.save_pretrained(checkpoint_path)
  return checkpoint_path
 

########################## MAIN ##########################
def main(model_output_path, model_name, dataset_root_path, dataset_version):
  
  # Hyperparameters
  batch_size = 10
  epochs = 100
  learning_rate = 0.00006

  # Logger
  logger = TrainingLogger(model_out_path=model_output_path, 
                          model_name=model_name, 
                          overwrite=True,
                          pretrained="",
                          train_dataset=dataset_root_path,
                          eval_dataset=dataset_version
                          )
  logger.set_hyperparams({
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs
    })
  
  # Metric
  metric = evaluate.load("mean_iou")

  # Dataset and Dataloader
  image_processor = SegformerImageProcessor(reduce_labels=True)

  train_dataset = BEVFeatureExtractionDataset(dataroot=dataset_root_path, version=dataset_version, image_processor=image_processor)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  eval_dataset = BEVFeatureExtractionDataset(dataroot=dataset_root_path, version=dataset_version, image_processor=image_processor)
  eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

  # Model
  model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                          num_labels=len(train_dataset.id2label),
                                                          id2label=train_dataset.id2label,
                                                          label2id=train_dataset.label2id)

  # Move model to GPU
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # AdamW optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  # Training
  logger.start_training()
  for epoch in range(epochs):
    print("Epoch:", epoch)

    # Training Loop
    train_metrics = train_loop(model, device, optimizer, train_dataloader, metric)

    # Evaluation Loop
    eval_metrics = evaluation_loop(model, device, eval_dataloader, metric)

    # Save checkpoint and registed epoch data
    # if eval_metrics['loss'] < history_eval_loss:
    checkpoint_path = save_checkpoint(model, model_output_path, model_name, epoch, overwrite=True)
    
    logger.log_epoch(epoch=epoch,
      checkpoint_path=checkpoint_path,
      train_metrics=train_metrics, 
      eval_metrics=eval_metrics
      )
    
  logger.finish_training()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Script para procesar datos con diferentes versiones.")
  parser.add_argument('model_output_path', type=str, help="Output path for runs of the model")
  parser.add_argument('model_name', type=str, help="Name of the model")
  parser.add_argument('dataset_root_path', type=str, help="Path of the dataset.")
  parser.add_argument('dataset_version', type=str, help="Dataset version.")
  args = parser.parse_args()

  main(args.model_output_path, args.model_name, args.dataset_root_path, args.dataset_version)
