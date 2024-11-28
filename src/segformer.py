from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation

from datasets.BEV import BEVFeatureExtractionDataset
from datasets.common import display_images

import torch
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

# Hyperparameters
batch_size = 5
epochs = 100
learning_rate = 0.00006

# Dataset and Dataloader
image_processor = SegformerImageProcessor(reduce_labels=False)
train_dataset = BEVFeatureExtractionDataset(dataroot='./tmp/BEVDataset', version='mini', image_processor=image_processor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=len(train_dataset.id2label),
                                                         id2label=train_dataset.id2label,
                                                         label2id=train_dataset.label2id,
)

# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
def train_loop()

model.train()
for e in range(epochs):  # loop over the dataset multiple times
   print("Epoch:", e)
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

        # Evaluation
        with torch.no_grad():
          upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
          predicted = upsampled_logits.argmax(dim=1)

          # note that the metric expects predictions + labels as numpy arrays
          metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches
        if idx % 100 == 0:
          # currently using _compute instead of compute
          # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
          metrics = metric._compute(
                  predictions=predicted.cpu(),
                  references=labels.cpu(),
                  num_labels=len(id2label),
                  ignore_index=255,
                  reduce_labels=False, # we've already reduced the labels ourselves
              )

          print("Loss:", loss.item())
          print("Mean_iou:", metrics["mean_iou"])
          print("Mean accuracy:", metrics["mean_accuracy"])


