# Waste-management
Multi-Object Detection for Recyclable Waste Management

Introduction:
The misclassification of waste and overuse of landfills pose a significant threat to the
ecosystem and have bred sanitary and health problems for mankind and wildlife alike. Recycling
solid waste is a critical step in reducing the harmful impacts of waste accumulation; however,
recycling first requires sorting waste material, which is complex and costly. To facilitate the
sorting process, this study proposes an intelligent waste classification system that uses deep
learning technology to detect and classify objects in waste images into six categories: cardboard,
glass, metal, paper, plastic, and trash.

Datasets:
TrashNet - Trashnet, is approximately 3.5 GB in size and contains 2,527 images (Yang
and Thung, 2016). The dataset contains the following number of images for each of the six
defined garbage classes, as shown in Figure 28: 501 glass, 594 paper, 403 cardboard, 482 plastic,
410 metal, and 137 other trash. The curators of the dataset used the following devices to capture
the images: Apple iPhone 7 Plus, Apple iPhone 5s, and Apple iPhone SE. The images were taken
by placing the trash objects against a white poster board under sunlight and/or room lighting. The
lighting and angle of each image vary, as Figure 29 shows. Since the size of each class was
relatively small, Yang and Thung (2016) used data augmentation techniques on each image,
including random rotation, random scaling, random brightness control, random translation, and
random shearing of the images. They also performed mean subtraction and normalization
![image](https://github.com/SadakhyaNarnur/Waste-management/assets/111921205/bea2ab15-f96b-4e89-a484-ab4b70649390)

TrashBox - TrashBox data has been extracted from the web by a comprehensive search. A
batch-downloading software named WFdownloader has been used to complete the process. This
tool facilitated bulk downloading of images from Google, Bing Images, and other sources. After
downloading, the images were cropped and segregated into their respective categories. Through
this process, they collected 17,785 images making it a unique benchmark dataset in this research
area. The dataset has seven categories: cardboard, paper, plastic, glass, medical waste, and
e-waste. The TrashBox dataset provides more types of waste images when compared to the
existing benchmark datasets in this field. As shown in Table 2, each of the categories has various
sub-classes defined. Also, Figure 26 showcases the distribution of the dataset among the
categories available, and Figure shows sample images from the TrashBox dataset.
![image](https://github.com/SadakhyaNarnur/Waste-management/assets/111921205/1a8dbe91-f938-473d-8d2a-7b42b7f1c47f)

Data Cleaning:
It is the first step for any type of data in the data engineering process. Data Cleaning
refers to the process of removing any noise, inconsistencies, and discrepancies in the data. In this
image data Inconsistent, Missing, Noisy, Incomplete, and Redundancy are handled.

Data Annotation:
Data annotation plays an important role in our object detection. Labeling the objects with
the category name in the image by creating the bounded boxes is a process of annotation. In this
project, labeling is done into six categories which are cardboard, paper, plastic, glass, trash, and
metal.

Data Transformation:
This is considered one of the crucial steps in data engineering as transforming our
pre-processed dataset to a format suitable for the use case is important. Data Transformation
includes Smoothing, Aggregation, Standardization and many other methods. The project uses
Normalization, Regularization, and Data Reduction techniques for transforming our data.

Data Normalization:
To standardize data, normalization is a pre-processing technique. Having data from
different sources within the same range, a lack of normalization before training can cause issues
with our network, making learning harder and slower. By normalizing data, one can get the data
within a range and reduce the skewness, which increases the models performance. By
normalizing data, it was able to get the data within a range and reduce the skewness, which helps
models learn faster and better. By normalizing the images, it means transforming them into
values with mean and standard deviation equal to 0.00 and 1.00, respectively.

Data Preparation
The entire image dataset is split into 80% training, 10% validation, and 10% for testing.
The final dataset that is left after all the Data Preprocessing, transformations and augmentations
has 5879 images distributed over the six classes. The split is done in Roboflow.

Modeling:
Figure shows the data flow and architecture for Faster R-CNN. The
dataset versions—with and without SVD—created during the data preparation stage were
divided into 80% training, 10% validation, and 10% testing each and are hosted on Roboflow.
During each training phase, the label map and TFRecord files for the training, validation, and
testing sets are imported into the Google Colab environment using a cURL provided by
Roboflow. Images resized to 640x640 were used to train this model.
![image](https://github.com/SadakhyaNarnur/Waste-management/assets/111921205/133cbc11-6a62-49d7-8577-071a6a5fc504)

Next, the training configuration of the model was performed by setting hyperparameter
values—namely, batch size, number of training steps, and number of evaluation steps. A training
step is equivalent to processing one batch of training images, while an evaluation step is
equivalent to evaluating at random one batch of validation images. The greater the batch size, the
greater the memory that is required to train. Similarly, the greater the number of training steps
specified, the longer the training but the higher the accuracy. For batch sizes of 16 and 8,
out-of-memory error resulted during the training job. A batch size of 4 along with 5000 training
steps and 10 evaluation steps were selected for both with and without SVD training runs.
Faster R-CNN Data Flow - After configuring the training hyperparameters, the pretrained weights of Faster R-CNN
were downloaded from the TensorFlow2 repository, as well as the base training configuration
file. The training configuration file was modified to reflect Pascal VOC detection metrics instead
of COCO detection metrics so that the evaluation metrics can yield class-specific performance
results.
Once the training process begins, the data flow within the network begins with the
Region Proposal Network, which outputs candidate regions that have a high probability of
containing objects within them. The number of regions generated in this stage is typically several
thousand. Features are then extracted from each region proposal. In this study, ResNet 50 was
selected as the feature extractor for Faster R-CNN. The feature extractor extracts a feature vector
of a predefined length from each region proposal using the Region of Interest (ROI) pooling layer. This feature vector is then used to assign the proposed region to one of the object classes
or to the background class by the Faster R-CNN network. Through the mechanism of
backpropagation discussed earlier, the model is iteratively trained to minimize the loss function.
Once the training process is complete, the model evaluation is run, which uses log files
stored in the model directory during the training process to generate the Pascal VOC detection
metrics. Finally, an inference test is conducted to test the model on testing images the model has
not seen before and evaluate results.

Summary:
The preprocessing steps include SVD and image resizing to
increase model speed, normalization to improve model accuracy, and various data augmentation
techniques, such as adding noise and rotation, to increase model robustness. The resulting data
was fed to supervised Deep Learning object detection algorithms: Faster R-CNN. 
Results show YOLOv5 had the highest mAP rate at 86.2% while also being
the fastest, with inference speed of 7 ms per image. This study provides a mechanism to
categorize recyclable waste with high average precision and high speed that could reduce both
the toxic waste in our landfills and the cost of waste management.
