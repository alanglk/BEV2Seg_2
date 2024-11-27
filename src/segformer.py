from transformers import SegformerFeatureExtractor
from torch.utils.data import DataLoader

from datasets.BEV import BEVFeatureExtractionDataset


feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor.reduce_labels = False
feature_extractor.size = 128


dataset = BEVFeatureExtractionDataset("./tmp/BEVDataset", "mini", feature_extractor)
dataloader = DataLoader(dataset, batch_size=5, shuffle= True)

for batch in dataloader:
    print(batch.size())

