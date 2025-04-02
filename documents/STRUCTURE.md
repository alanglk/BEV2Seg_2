
# Thesis Structure

Estructura principal de la memoria del TFM:

## 1. Introducción

- **Contexto y motivación**: Importancia de la segmentación semántica en BEV (Bird’s Eye View) en el ámbito de percepción vehicular.

- **Planteamiento del problema**: Explicación del desafío de identificar superficies planares en BEV.

- **Pregunta de investigación**: Presentar la pregunta principal y su relevancia.

- **Objetivos**:
    Evaluar la segmentación semántica en BEV con dos enfoques distintos.
    Implementar un pipeline para la generación de máscaras BEV locales.

- **Metodología general**: Breve resumen de la aproximación empleada (experimentos con modelos + desarrollo del pipeline).

- **Estructura del documento**: Breve explicación de los capítulos.

## 2. Estado del arte

- **Segmentación semántica**:
    CNN, ViT, modelos fundacionales...

- **Segmentación semántica en BEV**:
    Técnicas actuales y modelos utilizados en percepción vehicular.
    Diferencias entre entrenar en BEV directamente vs. reproyectar imágenes segmentadas.

- **Conclusión del estado del arte**: Identificación de la brecha en la literatura que tu trabajo aborda.

## 3. Metodología

- **Diseño del experimento de segmentación**:
    - Modelo utilizado (Segformer), arquitectura y pipelines segmentacion.
    - Dataset de entrenamiento -> generación de BEVDataset y NuImagesFormatted a partir de NuImages
     - Transformación para la generación de datos. Explicar cómo se pasa a BEV (IPM)
    - Data augmentations
    - Proceso de entrenamiento y validación en ambos casos.
    - Metodología de evaluación / comparación

- **Estructura del pipeline de fusión BEV**:
    Descripción de cada etapa:
        Captura de imágenes y depthmaps.
        Generación de detecciones 3D.
        Fusión de información.
        Creación de la máscara BEV local.
    Diagrama de flujo del sistema.

- **Métricas de evaluación**:
    Qué métricas se usan para comparar la calidad de segmentación (IoU, precisión, recall, etc.).

## 4. Evaluación y Resultados

- **Análisis comparativo de la segmentación BEV**:
    Desempeño de ambos enfoques con gráficos y tablas.
    Ventajas y desventajas de cada método.

- **Validación del pipeline BEV**:
    Pruebas en distintos escenarios.
    Errores detectados y mejoras potenciales.

- **Discusión**:
    Interpretación de los resultados en contexto.
    Comparación con estudios previos.

## 5. Conclusiones y Trabajo Futuro

- **Resumen de hallazgos clave**:
    ¿Cuál enfoque de segmentación BEV funciona mejor y por qué?
    ¿Cómo contribuye el pipeline al problema de percepción vehicular?
- **Limitaciones del estudio: Datos, modelos, restricciones computacionales**.
- **Futuras líneas de investigación**:
    Mejoras en la segmentación BEV (más datos, modelos más avanzados).
    Integración con sensores LiDAR o radares.
    Optimización del pipeline en tiempo real.



# 1. Introduction
...

# 2. State of The Art
...

## 2.1 Image Semantic Segmentation
The goal of semantic segmentation is to assign a semantic category to each pixel in an input image based on a predefined label set. The main difference between image classification and semantic segmentation is that semantic segmentation works in the pixel-level of the image. Instance segmentation is another task where a multiobject detector detects multiple objects and then this objects are segmented to obtain the pixel level boundaries of the detected instances. Then, a fusion between semantic and instance segmentation tasks is made and is named as panoptic segmentation, where the input image gets not only a pixel level semantic label but also differentiates between different instances of the same classes.

This thesis focuses on the semantic segmentation task, for which there has been a lot of research about.

Early methods relied on hand-crafted features but the introduction of Fully Convolutional Networks (FCNs) boosted semantic segmentation as convolutional layers in FCNs enables a data-driven feature extraction from images enabling pixel-wise predictions suitable for segmentation tasks. Additionaly, U-Net further impoved this by incorporating an encoder-decoder architecture with skip connections to refine segmented object boundaries and preserve spatial information.

Furthermore, the introduction of image classification backbones like VGG and ResNet into segmentation models further boosted performance. This backbones are known by their strong feature extraction capabilities, and also the residual connections in ResNets helped mitigate the vanishing gradient problem, improving the learning of deep features and enhancing segmentation results.

From that point, semantic segmentation started to been framed as a structured prediction problem and the development of specific modules and architectures for this task was made. That's the case of dilated convolution (atrous convolution), which increase the receptive field by "inflating" the CNN kernels with holes allowing models to capture more spatial context in their prediction.

Also, Transformers have been adapted from NLP to vision tasks, starting with Vision Transformers (ViTs), which model images as patch sequences using self-attention. For semantic segmentation, architectures like SETR replace CNN backbones with transformer for a better global understangin, while hybrid models like Swin Transformer and MaskFormer balance efficiency and context modeling. These advancements enable improved segmentation performance, particularly in tasks requiring both fine details and long-range dependencies.


## 2.2 BEV Semantic Segmentation
Traditional methods (3d_traffic_scene_understanding) estimate local BEV maps using camera inputs under the flat-ground assumption applying IPM. However, these methods require accurate camera parameters which has led to research focusing on camera parameter estimation for BEV transformation (BEV_params_estimation1, BEV_params_estimation2). Another key challenge in BEV map generation is handling object occupancy and occlusion. ADS must be aware of objects dimensions and should account for uncertainty in vehicular scenes. However, estimating vehicle occupancy is non-trivial as the necessary perspective views of objects are often unavailable. In this context, there are many reseachs that addresses the local semantic map estimation with different approaches.

Cam2BEV applies IPM to transform multi-camera segmented input images into the BEV domain which is feeded into the model to refine the BEV representation. To handle occlusions, Cam2BEV introduces an additional semantic class that explicitly marks occluded areas from all vehicle-mounted cameras. As the input of the model are already segmented images, an extra CNN was employed to test the method in real-world data.  

HDMapNet generates high-definition semantic maps from multi-camera input by employing a feature projection module that transforms image features into BEV space. The model first extracts image features and transforms them into the camera coordinate system with a shared MLP backbone, and then projects them into BEV using camera extrinsics. Finally a semantic segmentation decoder is used.

PYVA introduces a cross-view transformer that projects features from the front-view domain to the BEV domain. While similar to HDMapNet, PYVA differs in that it does not rely on camera parameters in the second transformation stage as the model is cappable of learning this transformation. Different to other methods, this work uses a GAN-based framework to manage occluions by estimating the vehicle's top view masks. However, this method is not suitable for multi-camera fusion.

Other approaches propose different architectures for BEV semantic segmentation. VPN introduces a two-layer MLP module for multi-camera feature fusion, followed by a decoder for semantic segmentation in indoor scenes. LSS proposes a unified framework that lifts 2D images into a 3D space by learning an implicit depth distribution and shows that their method is suitable for end-to-end motion plannig. M²BEV transforms 2D image features into 3D voxels along projection rays and obtains an efficient BEV representation which supports multiple end tasks such as semantic segmentation or object 3D detection.

MonoLayout tackles occlusion estimation by employing a standard encoder-decoder framework combined with adversarial training, making it suittable for predicting amodal layouts of the driving scene. BEVFormer similarly enhances occlusion reasoning by leveraging attention mechanisms to fuse multi-view spatial-temporal features from historical BEV maps.

There are also methods that combines information form multiple sensor such as FishingNet which extends VPN to use multiple sensors, or HDMapNet which is also cappable of using LiDAR sensors.  

In summary, existing approaches to local BEV map estimation typically follow one of two strategies: (1) performing early-stage segmentation on input images before refining the BEV representation, or (2) learning to embed image features into BEV space before passing them through a semantic segmentation decoder. However, to the best of our knowledge, no previous work directly trains a model on already-reprojected BEV images.

Instead of applying semantic segmentation before transforming images into BEV space, we study whether training a segmentation model directly on BEV images improves the representation of planar elements compared to the traditional segmentation-first-then-IPM approach. Furthermore, we treat occupancy and occlusion mask generation as a post-processing step applied to BEV semantic segmentation, where 3D object detection is performed. This is integrated into a pre-annotation pipeline for vehicle scenes, contributing to advancements in monocular 3D object detection. 


# 3. Methodology

This section details the experiments and implementations made to address the problems described in section (Introduction). On the first hand, the experimental design is introduced, tackling the model and dataset selection, how data augmentations are performed and the training and validation processes. Afterward, the implementation of the occupancy and occlusion masks preannotation pipeline is presented. Finally, the evaluation strategy of the pipeline is discussed to measure the quality of the resulting semantic masks and the monocular 3D detections for estimating the object's dimensions.

Thus, this project can be divided into three main blocks: (BEV) semantic segmentation experimentation, design and implementation of the preannotation pipeline and a thrid block of how the system is evaluated.

## 3.1 Segementation experiment design: BEV2Seg_2




### 3.1.2 Dataset

NuImages contains **93,000** samples, with approximately **80%** allocated for the training set and **20%** for validation. Additionally, NuImages includes a private test set reserved for benchmark evaluations, whose annotations are not publicly available.

To train the models in the pipeline, a parser has been developed to convert NuImages into a sub-dataset called **BEVDataset**. This dataset includes all front-facing images with NuImages annotations. Since the test annotations in NuImages are private, the validation set has been further split to ensure fair comparisons between models from different pipelines.

The conversion process is performed using a custom parser named **"oldatasets"**, which transforms NuImages into the structured **OpenLABEL** format, where metadata for each frame or sample is stored. In the case of BEVDataset, images are reprojected into the **BEV** domain using the **Video Content Descriptor (VCD)** library. This library provides tools to handle OpenLABEL annotations and manage both **2D** and **3D** data efficiently.

The "oldatasets" parser extracts the camera parameters for each sample and computes a **lookup table (LUT)** to apply **Inverse Perspective Mapping (IPM)** reprojection. Using this data, semantic pixel masks are generated and reprojected along with the original images into the BEV space. Since this reprojection involves **image warping**, the interpolation method must be carefully chosen:
- **Linear interpolation** is applied to images.
- **Nearest neighbor interpolation** is used for masks to preserve pixel class integrity.

The virtual BEV camera parameters remain fixed, as shown in Table \ref{tab:bev_camera_params}.

The final BEVDataset contains a total of **16,427 images**, distributed as follows:
- **80%** for training,
- **10%** for validation,
- **10%** for testing.

### 3.1.3 Data Augmentations
Data augmentations are commonly used in deep learning models to mitigate overfittings during training and improve model generalization. There exists multiple types of data augmentation on the image domain: from pixel-based transformations, such as color space modifications, histogram equalization or filtering operations; to geometric transformations, including translations, rotations, shearings and homographies. These techniques have been widely applied in computer vision tasks and have shown to enhance model performance. However, performing data augmentation in BEV is not an easy task, as IPM images are already homographies of camera images, resulting in inherent distorsions.

Filtering operations can be applied to both standard and \aclink{BEV} images, though they come with certain drawbacks. For camera domain images, geometric transformations were selected as the primary data augmentation method, following the strategies employed in training the SegFormer model \cite{segformer}. Accordingly, random resizing, random cropping, and horizontal flipping were chosen as augmentation operations for perspective images.  

Regarding \aclink{BEV} data augmentations, some multi-view methods implement strategies such as random flipping and random scaling, while others operate in the frequency domain \cite{HSDA}. However, these approaches apply augmentations to perspective images before the BEV transformation. Performing random cropping on a \aclink{BEV} image can lead to significant information loss, as large portions of the image may consist of unlabeled background data, potentially resulting in crops with insufficient information for effective training (Figure \ref{fig:bev_cropping}). Similarly, applying random horizontal flipping to a \aclink{BEV} image would merely create the illusion that the image was captured from a rear-facing camera rather than providing meaningful variation.  

In light of these challenges, a different approach was considered: applying geometric transformations by modifying the camera's extrinsic parameters before reprojecting to \aclink{BEV} space. The objective is to introduce random transformations along one of the camera’s rotation axes, generating diverse \aclink{BEV} reprojections with varying degrees of distortion. This technique enables the model to adapt to different extrinsic camera configurations, improving its robustness to variations in camera placement and orientation (Figure \ref{fig:bev_data_aug}).