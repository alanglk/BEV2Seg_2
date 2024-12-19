from oldatasets.NuImages import NuImagesFormattedDataset, NuImagesFormattedFeatureExtractionDataset
from oldatasets.common import display_images

import numpy as np

TMP_DIR = "./tmp/NuImagesFormatted"
DISPLAY_IMAGES = True

def test_load_formatted_dataset():
    """
    Test for loading a previously generated BEVDataset
    """
    dataset = NuImagesFormattedDataset(dataroot=TMP_DIR, version='mini')
    assert len(dataset) > 0
    
    for i in range(len(dataset)):
        image, target = dataset.__getitem__(i)
        if DISPLAY_IMAGES:
            target = dataset.target2image(target)
            image = image.numpy()
            display_images("test_load_formatted_dataset", [image, target])

def test_segformer_feature_extraction_dataset():
    from transformers import SegformerImageProcessor
    image_processor = SegformerImageProcessor(reduce_labels=True)

    dataset_nu = NuImagesFormattedDataset(TMP_DIR, 'mini')
    dataset_fe  = NuImagesFormattedFeatureExtractionDataset(TMP_DIR, 'mini', image_processor)

    assert len(dataset_nu) > 0
    assert len(dataset_fe) > 0

    for i in range(len(dataset_nu)):
        image_bev, target_bev = dataset_nu[i]
        original_labels = target_bev.squeeze().unique()
        print("Vanilla NuImagesFormatted")
        print(image_bev.shape)
        print(target_bev.shape)
        print(original_labels)

        encoded = dataset_fe[i]
        encoded_labels = encoded['labels'].squeeze().unique()
        print("FeatureExtraction NuImagesFormatted")
        print(encoded['pixel_values'].shape)
        print(encoded['labels'].shape)
        print(encoded_labels)

        print("LABEL DIFFERENCES")
        only_in_original = np.setdiff1d(original_labels, encoded_labels)
        only_in_encoded = np.setdiff1d(encoded_labels, original_labels)
        diff = np.concatenate((only_in_original, only_in_encoded))
        print(f"Estan en el original pero no después de feature extraction:")
        print(only_in_original)
        print("Estan en el feature extraction pero no en el original:")
        print(only_in_encoded)
        print()

        if DISPLAY_IMAGES:
            target_bev = dataset_nu.target2image(target_bev)
            target_fe = dataset_fe.target2image(encoded['labels'])
            display_images("Encoded Targets", [target_bev, target_fe])
        



