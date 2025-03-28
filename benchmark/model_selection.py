

import numpy as np

import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms

import cv2

from PIL import Image

from tqdm import tqdm
import time
import os

CITYSCAPES_DATASET  = "/run/user/17937/gvfs/smb-share:server=gpfs-cluster,share=databases/GeneralDatabases/Cityscapes"
SEGFORMER_PATH = "./benchmark/inference/segformer"

debug_ = False

def check_folder(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"La carpeta '{path}' no existe.")
    
    if not os.listdir(path): 
        print(f"La carpeta '{path}' está vacía.")
        return True
    else:
        print(f"La carpeta '{path}' contiene archivos.")
        return False

def target2image(target, colormap: dict) -> np.ndarray:
    """
     Converts target (seg mask) into BGR image!!!
    INPUT:
        target: torch.tensor (H, W)
        colormap: dict {0: RGB, 1: RGB...}
    OUTPUT:
        np.array BGR Image
    """
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    
    res_mask = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    for label, rgb_col in colormap.items():
        res_mask[target == label] = rgb_col

    # Convert to BGR (for display with opencv)
    res_mask = cv2.cvtColor(res_mask, cv2.COLOR_RGB2BGR) 
    return res_mask


def segformer(segformer_out_path:str):
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

    if not check_folder(segformer_out_path):
        print("The folder is not empty")
        return
    
    model_path="nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    # model.eval()

    cs_eval_dataset = Cityscapes(root=CITYSCAPES_DATASET, split="val", mode="fine", target_type="semantic")
    cs_name2id      = { cs.name:cs.id for cs in cs_eval_dataset.classes }
    cs_color_map    = { cs_class.id:cs_class.color  for cs_class in cs_eval_dataset.classes }
    cs_num_images = len(cs_eval_dataset)
    
    model2gt_labels = { model_id:cs_name2id[model_label] for model_id, model_label in model.segformer.config.id2label.items() }

    # Measure FPS
    inference_times = []
    fps_values = []
    
    with torch.no_grad():
        for idx in tqdm(range(cs_num_images), desc="Evaluating"):
            image_pil, label_pil = cs_eval_dataset.__getitem__(idx)
            image_path = cs_eval_dataset.images[idx]
            image_name = os.path.basename(image_path).replace("_leftImg8bit.png", "_gtFine_labelIds.png") 
            print(image_name)
            # Image preprocessing
            encoding = feature_extractor(images=image_pil, return_tensors="pt")  
            encoding = {k: v.to(device) for k, v in encoding.items()}  
            
            # Inference
            start_time = time.time()
            outputs = model(**encoding)  # Forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None  # Sync CUDA if using GPU
            end_time = time.time()

            # Get the mask from logits
            label_ = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[np.array(label_pil).shape])[0]
            
            # Map pred labels to GT format
            cs_like_labels_ = np.zeros(label_.shape, dtype=np.uint8)
            for model_label, gt_label in model2gt_labels.items():
                cs_like_labels_[label_ == model_label] = gt_label

            # Save inference
            l_, l  = cs_like_labels_, np.array(label_pil)
            Image.fromarray(l_).save(os.path.join(segformer_out_path, image_name))

            # Debug
            global debug_
            if debug_:
                colored_l   = target2image(l,   cs_color_map)
                colored_l_  = target2image(l_,  cs_color_map)

                cv2.imshow("GT mask", colored_l)
                cv2.imshow("Pred mask", colored_l_)
                cv2.waitKey(0)

            # Compute inference time
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Compute FPS for this frame
            fps = 1 / inference_time if inference_time > 0 else 0
            fps_values.append(fps)

    mean_time   = sum(inference_times)  / len(inference_times)
    mean_fps    = sum(fps_values)       / len(fps_values)

    print(f"Mean inference time: {mean_time} | mean inference FPS: {mean_fps}")

def main():
    cv2.namedWindow("GT mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pred mask", cv2.WINDOW_NORMAL)
    segformer(SEGFORMER_PATH)

if __name__ == "__main__":
    main()
