from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from oldatasets.BEV import BEVFeatureExtractionDataset

import argparse

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
        ignore_index=0,
        reduce_labels=False,
    )
    
    # # add per category metrics as individual key-value pairs
    # per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    # per_category_iou = metrics.pop("per_category_iou").tolist()

    # metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    # metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
    return metrics

########################## MAIN ##########################
def main(model_output_path, model_name, dataset_root_path, dataset_version):
    # Hyperparams
    initial_lr = 2e-5
    batch_size = 16
    train_epochs = 3

    # Dataset and Dataloader
    image_processor = SegformerImageProcessor(reduce_labels=True)

    train_dataset = BEVFeatureExtractionDataset(dataroot=dataset_root_path, version=dataset_version, image_processor=image_processor)
    eval_dataset = BEVFeatureExtractionDataset(dataroot=dataset_root_path, version=dataset_version, image_processor=image_processor)
    global num_labels
    num_labels = len(train_dataset.id2label)

    # Model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                          num_labels=len(train_dataset.id2label),
                                                          id2label=train_dataset.id2label,
                                                          label2id=train_dataset.label2id)
    
    
    # Training Args
    training_args = TrainingArguments(
        # output_dir: directory where the model checkpoints will be saved.
        output_dir=model_output_path,
        # evaluation_strategy (default "no"):
        # Possible values are:
        # "no": No evaluation is done during training.
        # "steps": Evaluation is done (and logged) every eval_steps.
        # "epoch": Evaluation is done at the end of each epoch.
        eval_strategy="epoch",
        # logging_strategy (default: "steps"): The logging strategy to adopt during
        # training (used to log training loss for example). Possible values are:
        # "no": No logging is done during training.
        # "epoch": Logging is done at the end of each epoch.
        # "steps": Logging is done every logging_steps.
        logging_strategy="epoch",
        # save_strategy (default "steps"):
        # The checkpoint save strategy to adopt during training. Possible values are:
        # "no": No save is done during training.
        # "epoch": Save is done at the end of each epoch.
        # "steps": Save is done every save_steps (default 500).
        save_strategy="epoch",
        # learning_rate (default 5e-5): The initial learning rate for AdamW optimizer.
        # Adam algorithm with weight decay fix as introduced in the paper
        # Decoupled Weight Decay Regularization.
        learning_rate=initial_lr,
        # per_device_train_batch_size: The batch size per GPU/TPU core/CPU for training.
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size: The batch size per GPU/TPU core/CPU for evaluation.
        per_device_eval_batch_size=batch_size,
        # num_train_epochs (default 3.0): Total number of training epochs to perform
        # (if not an integer, will perform the decimal part percents of the last epoch
        # before stopping training).
        num_train_epochs=train_epochs,
        # load_best_model_at_end (default False): Whether or not to load the best model
        # found during training at the end of training.
        load_best_model_at_end=True,
        # metric_for_best_model:
        # Use in conjunction with load_best_model_at_end to specify the metric to use
        # to compare two different models. Must be the name of a metric returned by
        # the evaluation with or without the prefix "eval_".
        metric_for_best_model="mean_accuracy",
        # report_to:
        # The list of integrations to report the results and logs to. Supported
        # platforms are "azure_ml", "comet_ml", "mlflow", "tensorboard" and "wandb".
        # Use "all" to report to all integrations installed, "none" for no integrations.
        report_to="tensorboard"
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
  parser.add_argument('model_output_path', type=str, help="Output path for runs of the model")
  parser.add_argument('model_name', type=str, help="Name of the model")
  parser.add_argument('dataset_root_path', type=str, help="Path of the dataset.")
  parser.add_argument('dataset_version', type=str, help="Dataset version.")
  args = parser.parse_args()

  main(args.model_output_path, args.model_name, args.dataset_root_path, args.dataset_version)
