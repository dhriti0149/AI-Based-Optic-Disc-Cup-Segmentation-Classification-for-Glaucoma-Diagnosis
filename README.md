# AI-Based-Optic-Disc-Cup-Segmentation-Classification-for-Glaucoma-Diagnosis
Here, I have applied and compared UNet variants on fundus images to segment the optic disc and optic cup. Then, I have extracted clinical data for glaucoma detection and evaluated Logistic Regression, ResNet18 &amp; ShuffleNetV2  by 5-fold cross-validation. 

Automated Glaucoma Screening from Retinal Fundus Images

An end-to-end glaucoma screening pipeline that combines deep learning-based optic nerve head segmentation with clinically interpretable classification. The project compares multiple segmentation and classification architectures and demonstrates how retinal biomarkers can be leveraged for explainable glaucoma detection.

Overview

Glaucoma is a progressive optic neuropathy and a leading cause of irreversible blindness worldwide. One of the most important structural indicators of glaucoma is the Vertical Cup-to-Disc Ratio (VCDR), which is derived from the optic disc and optic cup regions of retinal fundus images.

This project develops a 2-stage framework:

Fundus Image
      ↓
Optic Disc & Cup Segmentation
      ↓
Biomarker Extraction
(Disc Area, Cup Area, VCDR)
      ↓
Glaucoma Classification


The study compares:

Segmentation Models
U-Net
U-Net++
Attention U-Net

Classification Models
Logistic Regression
ResNet18 (5-channel input)
ShuffleNetV2 (5-channel input)

Dataset
Dataset: Drishti-GS

The dataset contains retinal fundus photographs along with expert annotations of:

Optic Disc (OD)
Optic Cup (OC)

These annotations serve as ground truth masks for segmentation training and evaluation.

Methodology
Stage 1: Image Preprocessing

Average intensity per channel across sample is checked.
Red channel mean is 85.93
Green channel mean is 42.18
Blue channel mean is 13.63
Data check  has been passed.
 Channel hierarchy looks as expected for retinal fundus images
Intensity hierarchy correctly shows the Red channel as the brightest and Blue as the darkest (R > G > B).
This perfectly matches the natural, haemoglobin-rich colour signature of a standard retinal fundus photograph.


Preprocessing steps include:

Image resizing
Normalization
Data augmentation
Fundus-mask pairing
Tensor conversion
Interpolation Strategy
Data Type	InterpolationRGB Image	INTER_AREA for Area Interpolation
Segmentation Mask	INTER_NEAREST for Nearest Neighbour Interpolation

Nearest-neighbor interpolation was used for masks to preserve binary labels during resizing.
Area interpolation (often called box resampling or pixel-area relation) resizes digital images by computing each output pixel from the area overlap or integrated intensity over a corresponding region of the source image.

Fundus Images are resized by Area Interpolation to reduce down sampling aliasing and retain structural background detail.
Segmentation Masks are resized by Nearest Neighbour Interpolation to protect discrete binary pixel boundaries from mathematical blurring or edge distortion.
A fixed random seed is set to 42 globally across PyTorch, NumPy, Python random, and CUDA.
Ensured reproducibility through deterministic computation and fixed random seeds.
Prepared image-mask pairs and extracted separate optic disc and optic cup masks.
Applied synchronized augmentation and converted all inputs into normalized tensors for model training.



Stage 2: Optic Disc and Cup Segmentation

Three segmentation architectures were evaluated.

U-Net uses encoder-decoder architecture, skip connections & multi-scale feature representation
Attention U-Net introduces attention gates, region-focused feature filtering & background suppression
U-Net++ introduces nested dense skip pathways, progressive feature refinement,segmentation Evaluation & reduced semantic gap between encoder and decoder


Validation Loss and Classification Accuracy:
Here, the first graph shows how well the segmentation models learn & generalize by validation loss, while the second graph shows how the quality of those segmentations affects downstream glaucoma classification. U-Net++ performs best in both graphs, indicating that it produces the most clinically reliable optic disc-cup segmentations and the most informative features for glaucoma diagnosis.

<img width="1631" height="602" alt="image" src="https://github.com/user-attachments/assets/68ce4504-c11c-4d33-a604-4a8a30ec3631" />

Segmentation Models Test Performance Comparison:

Here, we see that UNet++ has the highest overall performance outperforming Standard UNet & Attention UNet as nested dense skip connections capture fine-grained spatial details, that are critical for delineating precise boundaries like the smaller optic cup. 

Attention Unet successfully scored Optic Disc segmentation to match U-Net++, but struggles with the smaller Optic Cup yielding the lowest OC metrics across all evaluated architectures as the attention gates need substantial training data to learn accurate maps for small structures.

<img width="776" height="495" alt="image" src="https://github.com/user-attachments/assets/5ef40289-ae01-4b70-8b17-a33e399ed257" />
<img width="744" height="495" alt="image" src="https://github.com/user-attachments/assets/6a2ed7c5-01bd-41f0-82aa-15ae80965fd4" />

Morphometric Method Comparison: Actual vs Predicted Cup Disc Ratio:
Here, we see that U-Net++ got the lowest Mean Absolute Error, the highest Pearson correlation & a non-significant paired t-test result. 
This means that the CDR values predicted by U-Net++ were closest to the ground-truth ophthalmologist annotations and showed no statistically significant systematic bias. 
In contrast, both Standard U-Net & Attention U-Net showed significant deviations from the ground truth.
<img width="527" height="83" alt="image" src="https://github.com/user-attachments/assets/fb07d3db-89bf-4811-81a2-5326839f573a" />

 <img width="2000" height="421" alt="image" src="https://github.com/user-attachments/assets/bb2a7d0c-3416-4359-9b4e-7aaab2be65c4" />

Analysis of ROC-AUC for CDR Classification:
Standard UNet is giving AUC score of 0.7893 that despite being an acceptable score can trigger a high rate of false positives.

UNet++ gives the highest diagnostic power with an AUC of 0.8982, establishing it as the most reliable model because boundary capture directly translates into reliable clinical classification.

Attention UNet gives a terrible AUC of 0.5857 that is due to not segmenting optic cup accurately 

<img width="741" height="692" alt="image" src="https://github.com/user-attachments/assets/f5246acd-6389-4c37-8f01-1d0c641ee4d2" />


Stage 4: Glaucoma Classification

Two different classification paradigms were compared.

Approach 1: Feature Engineering
U-Net++ + Logistic Regression

Workflow:

Fundus Image
      ↓
U-Net++
      ↓
Disc Mask + Cup Mask
      ↓
Disc Area
Cup Area
VCDR
      ↓
Logistic Regression


Advantages:

Clinically interpretable
Lightweight
Uses established ophthalmic biomarkers
Approach 2: Feature Learning
5-Channel Tensor

The RGB fundus image was augmented with segmentation masks:

Channel 1 → Red
Channel 2 → Green
Channel 3 → Blue
Channel 4 → Disc Mask
Channel 5 → Cup Mask


Tensor shape:

[5 × 224 × 224]


This representation was used by:

ResNet18
ShuffleNetV2
Cross-Validation

Stratified 5-Fold Cross Validation was employed for classifier evaluation.

Purpose:

Reduce split dependency
Estimate model robustness
Measure performance stability

Accuracy Comparison:
Resnet18 and the custom pipeline(UNet++ & Logistic Regression) gets the same validation accuracy.
Shufflenetv2 gets a lower accuracy due to lightweight architecture limitations.
This shows that accuracy alone could not differentiate model performance of 2 classifiers alone, motivating the use of additional metrics like F1-score &  ROC-AUC for a more better evaluation.
<img width="741" height="609" alt="image" src="https://github.com/user-attachments/assets/83779838-0342-4a1e-ba75-1bcb81d88477" />



F1-Score Stability Comparison:
The custom pipeline (U-Net++ & LogReg) achieves the optimal clinical balance with a stable Mean F1-Score and low variance, outperforming a highly volatile ResNet-18 and matching the stability of ShuffleNet-V2 without its underlying diagnostic metric failures.
<img width="843" height="357" alt="image" src="https://github.com/user-attachments/assets/f5198aaf-4f8f-4f4f-8786-290ecfcd2adb" />
<img width="841" height="234" alt="image" src="https://github.com/user-attachments/assets/d8ccd205-0e60-4c4b-8474-844b81fd2c7a" />

<img width="825" height="501" alt="image" src="https://github.com/user-attachments/assets/7bc9ed8a-389c-4b09-babd-70f5b910e5c7" />

ROC-AUC  Curve Analysis:
ShuffleNetV2 achieved the highest ROC-AUC (0.9091), indicating the strongest class discrimination.
The custom pipeline outperformed ResNet18 but remained below ShuffleNetV2 in ROC-AUC.
The custom pipeline achieved comparable F1-score stability while using clinically meaningful features.
This provides a more interpretable decision process based on optic disc area, optic cup area, and VCDR.
<img width="844" height="845" alt="image" src="https://github.com/user-attachments/assets/95d15ee0-403f-4b5a-a5e4-ec4d52aca13f" />

ROC-AUC  Variance Comparison:
Shufflenetv2 shows the highest roc-auc variance indicating despite a low f1 variance, this model is very unstable  at confidence despite choosing the right class
Resnet18 gets the lowest roc-auc variance but it’s high f1 variance shows the model is inconsistent with decision boundaries despite being stable at ranking samples.
The custom pipeline’s performance in roc-auc variance and f1 variance shows that this is the most balanced most predictable model.
<img width="921" height="675" alt="image" src="https://github.com/user-attachments/assets/4779de9a-2ef3-49d6-bc68-ba906f97e13f" />

Master Performance Analysis: 
Custom Pipeline achieved the best overall balance across all evaluation metrics.

ResNet18 matched the validation accuracy but showed lower F1-score and higher fold-to-fold variation.

ShuffleNetV2 achieved the highest F1-score, but lower accuracy and unstable ROC-AUC reduced its overall effectiveness.

Therefore, the Custom Pipeline was selected as the final framework.
<img width="1064" height="471" alt="image" src="https://github.com/user-attachments/assets/dc28850c-15ff-46e6-8fdd-887b5db918d0" />
<img width="1064" height="395" alt="image" src="https://github.com/user-attachments/assets/64207e4e-0618-430c-8bfa-a7edc7473518" />




Key Findings:
U-Net++
Achieved the best optic cup segmentation.
Produced the most accurate VCDR estimation.
Demonstrated the strongest morphometric agreement.
ResNet18
Highest model complexity.
Showed greater fold-to-fold variability.
Did not outperform the feature-engineering approach.
ShuffleNetV2
Achieved the highest mean F1-score.
Showed strong threshold-specific performance.
Exhibited higher ROC-AUC variability.
Custom Pipeline
Consistently ranked near the top across all metrics.
Achieved the highest mean ROC-AUC.
Provided clinically interpretable predictions.

Final Framework:
Fundus Image
      ↓
U-Net++
      ↓
Disc Mask + Cup Mask
      ↓
Disc Area
Cup Area
VCDR
      ↓
Logistic Regression
      ↓
Glaucoma Screening


Author:
IIT Madras - AI & Data Science Capstone Project
