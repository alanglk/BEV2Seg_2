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
  train_loss, train_iou, train_acc = [], [], []

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
      metrics = compute_metrics(metric, predicted, labels, model.config.numlabels)
      train_loss.append(loss.item())
      train_iou.append(metrics["mean_iou"])
      train_iou.append(metrics["mean_accuracy"])

  train_loss = sum(train_loss)  / len(train_loss)
  train_iou  = sum(train_iou)   / len(train_iou)
  train_acc  = sum(train_acc)   / len(train_acc)

  train_metrics = { "loss": train_loss, "mean_iou": train_iou, "mean_acc": train_acc }
  return train_metrics


# Evaluation Loop
def evaluation_loop(model, device: torch.device, eval_dataloader: DataLoader, metric: evaluate.EvaluationModule):
  eval_loss, eval_iou, eval_acc = [], [], []
  with torch.no_grad():
    for batch in enumerate(tqdm(eval_dataloader)):
      pixel_values = batch["pixel_values"].to(device)
      labels = batch["labels"].to(device)

      # prediction
      outputs = model(pixel_values=pixel_values, labels=labels)
      loss, logits = outputs.loss, outputs.logits
      upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
      predicted = upsampled_logits.argmax(dim=1)

      # Compute evaluation metrics
      metrics = compute_metrics(metric, predicted, labels, model.config.numlabels)
      eval_loss.append(loss.item())
      eval_iou.append(metrics["mean_iou"])
      eval_iou.append(metrics["mean_accuracy"])
  
  eval_loss = sum(eval_loss)  / len(eval_loss)
  eval_iou  = sum(eval_iou)   / len(eval_iou)
  eval_acc  = sum(eval_acc)   / len(eval_acc)

  eval_metrics = { "loss": eval_loss, "mean_iou": eval_iou, "mean_acc": eval_acc }
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
def main():

  # config
  MODEL_OUTPUT_PATH = "./tmp/models"
  MODEL_NAME = "model1"
  DATASET_ROOT_PATH = "./tmp/BEVDataset"
  DATASET_VERSION = "mini"

  # Hyperparameters
  batch_size = 5
  epochs = 100
  learning_rate = 0.00006

  # Logger
  logger = TrainingLogger(model_out_path=MODEL_OUTPUT_PATH, 
                          model_name=MODEL_NAME, 
                          overwrite=True,
                          pretrained="",
                          train_dataset=DATASET_ROOT_PATH,
                          eval_dataset=DATASET_ROOT_PATH
                          )
  logger.set_hyperparams({
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs
    })
  
  # Metric
  metric = evaluate.load("mean_iou")

  # Dataset and Dataloader
  image_processor = SegformerImageProcessor(reduce_labels=False)

  train_dataset = BEVFeatureExtractionDataset(dataroot=DATASET_ROOT_PATH, version=DATASET_VERSION, image_processor=image_processor)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  eval_dataset = BEVFeatureExtractionDataset(dataroot=DATASET_ROOT_PATH, version=DATASET_VERSION, image_processor=image_processor)
  eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

  # Model
  model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                          num_labels=len(train_dataset.id2label),
                                                          id2label=train_dataset.id2label,
                                                          label2id=train_dataset.label2id)

  # Move model to GPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    checkpoint_path = save_checkpoint(model, MODEL_OUTPUT_PATH, MODEL_NAME)
    
    logger.log_epoch(epoch=epoch,
      checkpoint_path=checkpoint_path,
      train_metrics=train_metrics, 
      eval_metrics=eval_metrics
      )
    
  logger.finish_training()


if __name__ == "__main__":
  main()
