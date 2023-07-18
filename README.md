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

Summary:
The preprocessing steps include SVD and image resizing to
increase model speed, normalization to improve model accuracy, and various data augmentation
techniques, such as adding noise and rotation, to increase model robustness. The resulting data
was fed to supervised Deep Learning object detection algorithms: Faster R-CNN. 
Results show YOLOv5 had the highest mAP rate at 86.2% while also being
the fastest, with inference speed of 7 ms per image. This study provides a mechanism to
categorize recyclable waste with high average precision and high speed that could reduce both
the toxic waste in our landfills and the cost of waste management.
