# HeadPoseEstimation
This project involves head pose estimation using CNN + attention. The task is to estimate Yaw, Pitch and Roll as a Regression task (Currently in the development stage).

**HeadPoseEstimation_V3** is the most recent notebook showing our progress in which we have obtained a training loss of around 4 and validstion loss of around 7.5
We are using L1 loss keeping the training batch size of 16 and validation batch size of 4. 
We have trained the model on efficient-net b3.

**Next Step:** Try augmentation along with attention added to our model. Train with AFLW dataset

# Dataset
We have trained the model on BIWI. We have acquired the AFLW2000 dataset. Training is yet to be done with AFLW dataset.
We have manually found out the ideal crop size for each person and have cropped out the faces and created the dataset on which training has been done.
# Cropped Images Link
https://drive.google.com/open?id=13pca-FDOiFsGKEo1Z9YL73LdYNBorNdm

# Evaluation protocl
Follow the evaluation protocol in FSA-Net https://github.com/shamangary/FSA-Net (CVPR 2019)
