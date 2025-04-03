import torch
from transformers import SegformerForSemanticSegmentation


mit     = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")
model   = SegformerForSemanticSegmentation.from_pretrained("./models/segformer_bev/raw2bevseg_mit-b0_v0.5")


# Freeze encoder for training
for p in model.segformer.parameters():
    p.requires_grad = False



encoder_frozen = all(not param.requires_grad for param in model.segformer.encoder.parameters())
